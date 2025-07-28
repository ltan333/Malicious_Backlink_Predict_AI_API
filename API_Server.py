# --- Import all libraries and frameworks ---
import const as config
import json
import os
import logging
import asyncio
import httpx
import urllib3
import torch
from urllib.parse import urlparse
from pathlib import Path
from playwright.async_api import async_playwright
import fitz
from asyncio import Lock as AsyncLock
from typing import List
from contextlib import nullcontext
from jose import JWTError, jwt, ExpiredSignatureError
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime, timezone, timedelta
from bs4 import BeautifulSoup
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

# --- LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.FileHandler("Logs/api_server.log"), logging.StreamHandler()]
)

# --- CACHE ---
homepage_cache = {}
cache_lock = AsyncLock()
cache_updated = False

# --- MODEL ---
model = None
tokenizer = None

# --- Define device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autocast_ctx = torch.amp.autocast(device_type="cuda") if torch.cuda.is_available() else nullcontext()

label2id = {
    "gambling": 0, "movies": 1, "ecommerce": 2, "government": 3, "education": 4, "technology": 5,
    "tourism": 6, "health": 7, "finance": 8, "media": 9, "nonprofit": 10, "realestate": 11,
    "services": 12, "industries": 13, "agriculture": 14
}
id2label = {v: k for k, v in label2id.items()}

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=64):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze()
        }

def load_cache():
    global homepage_cache
    if os.path.exists(config.CACHE_FILE):
        try:
            with open(config.CACHE_FILE, "r", encoding="utf-8") as f:
                homepage_cache = json.load(f)
            logging.info(f"Cache loaded from {config.CACHE_FILE}")
        except (json.JSONDecodeError, OSError):
            logging.error("Failed to load cache file, starting with an empty cache.")
            print("[WARN] Cache file is empty or corrupted. Starting fresh.")
            homepage_cache = {}
    else:
        homepage_cache = {}

async def save_cache():
    try:
        async with cache_lock:
            with open(config.CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(homepage_cache, f, ensure_ascii=False, indent=2)
        logging.info(f"Cache saved to {config.CACHE_FILE}")
    except Exception as e:
        logging.error(f"Failed to save cache: {e}")
        print("[ERROR] Failed to save cache. Check logs for details.")

def load_model():
    global model, tokenizer
    # Update new Model version 6
    model_path = "Models/phobert_base_v6"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True).to(device).eval()
        logging.info(f"Model loaded from {model_path}")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        print("[ERROR] Failed to load model. Check logs for details.")

def classify_batch(texts: List[str]):
    dataset = TextDataset(texts, tokenizer)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    results = []
    with torch.no_grad(), autocast_ctx:
        for batch in loader:
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1)
            scores, preds = torch.max(probs, dim=-1)
            for i in range(len(scores)):
                results.append((id2label[preds[i].item()], scores[i].item()))
    return results

async def classify_text_async(text: str):
    return classify_batch([text])[0]

# --- SCRAPER ---
def is_accessible(url, timeout=5):
    try:
        response = httpx.head(url, timeout=timeout, follow_redirects=True, verify=False)
        return response.status_code < 400
    except:
        return False

def extract_pdf_text(pdf_bytes):
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        return "\n".join(page.get_text() for page in doc).strip()
    except:
        return ""

def fetch_pdf_httpx(url, timeout=10):
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/pdf",
        "Referer": url,
    }
    try:
        with httpx.Client(follow_redirects=True, timeout=timeout, headers=headers, verify=False) as client:
            r = client.get(url)
            if r.status_code == 200 and r.content.startswith(b"%PDF"):
                return r.content
    except:
        pass
    return None

async def fetch_pdf_playwright(url, timeout=20):
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            try:
                context = await browser.new_context(ignore_https_errors=True)
                page = await context.new_page()
                response = await page.goto(url, timeout=timeout * 1000)
                if response and response.ok:
                    data = await response.body()
                    if data.startswith(b"%PDF"):
                        return data
            finally:
                await browser.close()
    except:
        pass
    return None

async def handle_pdf(url):
    pdf_data = fetch_pdf_httpx(url) or await fetch_pdf_playwright(url)
    if pdf_data:
        text = extract_pdf_text(pdf_data)
        return text if text else "inaccessible"
    return "inaccessible"

def fetch_static_html(url, timeout=10):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        with httpx.Client(follow_redirects=True, verify=False, timeout=timeout, headers=headers) as client:
            r = client.get(url)
            return r.text
    except:
        return ""

async def fetch_dynamic_html(url, timeout=10):
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            try:
                context = await browser.new_context(ignore_https_errors=True)
                page = await context.new_page()
                await page.goto(url, timeout=timeout * 1000)
                await page.wait_for_timeout(4000)
                html = await page.content()
                return html
            finally:
                await browser.close()
    except:
        return ""

async def handle_html(url):
    static_html = fetch_static_html(url)
    if not static_html:
        return "inaccessible"

    soup = BeautifulSoup(static_html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    for tag in soup.select('[style*="display:none"], [style*="display: none"]'):
        tag.decompose()

    text = soup.get_text("\n", strip=True)

    has_fw = any(f in static_html.lower() for f in ["vue", "react", "angular", "webpack", "require.js"])
    has_ns = "enable javascript" in static_html.lower()

    if has_fw or has_ns:
        dynamic_html = await fetch_dynamic_html(url)
        if not dynamic_html:
            return text or "inaccessible"
        soup = BeautifulSoup(dynamic_html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        for tag in soup.select('[style*="display:none"], [style*="display: none"]'):
            tag.decompose()
        text = soup.get_text("\n", strip=True)

    return text or "inaccessible"

async def get_website_context_async(url):
    if not is_accessible(url):
        return "inaccessible"
    ext = Path(urlparse(url).path).suffix.lower()
    if not ext:
        ext = ".html"
    if ext == ".pdf":
        return await handle_pdf(url)
    else:
        return await handle_html(url)
    
def overlap_check(text1, text2):
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    overlap = words1.intersection(words2)
    ratio = len(overlap) / len(words2) if words2 else 0
    return (ratio > 0.85)

async def check_subpage_context(domain, backlink, text, final_label, score):
    subpage_context = await get_website_context_async(backlink)
    if subpage_context == "inaccessible":
        return OutputEntry(domain=domain, backlink=backlink, label=final_label, score=score)
    
    if overlap_check(subpage_context, text):
        return OutputEntry(domain=domain, backlink=backlink, label="An toàn", score=score)
    
    return OutputEntry(domain=domain, backlink=backlink, label=final_label, score=score)


# --- FASTAPI APP ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info(f"Running on device with: {device}")
    load_cache()
    load_model()
    yield

app = FastAPI(lifespan=lifespan)

# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="get-access-token")
http_bearer = HTTPBearer()

# Payload token
class Token(BaseModel):
    access_token: str
    token_type: str

# Input token
class SecretKeyInput(BaseModel):
    api_key: str

# Input model
class InputEntry(BaseModel):
    domain: str
    backlink: str
    title: str
    description: str

# Output model
class OutputEntry(BaseModel):
    domain: str
    backlink: str
    label: str
    score: float

async def create_access_token(data: dict, expires_delta: timedelta):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + expires_delta
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, config.API_SECRET_KEY, algorithm=config.ALGORITHM)
    return encoded_jwt

# async def verify_access_token(token: str = Depends(oauth2_scheme)):
async def verify_access_token(credentials: HTTPAuthorizationCredentials = Depends(http_bearer)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate token",
    )
    try:
        token = credentials.credentials
        payload = jwt.decode(token, config.API_SECRET_KEY, algorithms=[config.ALGORITHM])
        print(f"Token payload: {payload}")
        return {
            "sub": payload.get("sub"),
            "exp": payload.get("exp")
        }
    except ExpiredSignatureError:
        logging.warning("Token expired")
        raise HTTPException(status_code=401, detail="Token has expired")
    except JWTError:
        logging.warning("Invalid token")
        raise credentials_exception

# --- ROUTES ---
# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the THD AI Model API!", "version": config.SERVER_VERSION}

# Endpoint to get access token
@app.post("/get-access-token", response_model=Token)
async def login_for_access_token(secret_input: SecretKeyInput):
    if secret_input.api_key != config.API_SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    access_token_expires = timedelta(seconds=config.ACCESS_TOKEN_EXPIRE_SECOND)
    access_token = await create_access_token(
        data={"sub": "authenticated_user"}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# Endpoint to get token
@app.get("/get-token-info")
async def get_token_info(current_user: str = Depends(verify_access_token)):
    return {"message": "Token is valid", "exp": datetime.fromtimestamp(current_user.get("exp")), "user": current_user.get("sub")}

# Endpoint to predict
@app.post("/predict", response_model=List[OutputEntry], dependencies=[Depends(verify_access_token)])
async def predict(input_data: List[InputEntry]):
    # Load Models checking
    if model is None or tokenizer is None:
        logging.error("Model or tokenizer not loaded. Cannot process prediction.")
        raise HTTPException(status_code=503, detail="Model not loaded. Please try again later.")
    
    # Limit input to avoid overload RAM/GPU - with max batchsize MAX_BATCH_SIZE
    MAX_BATCH_SIZE = config.MAX_BATCH_SIZE
    if len(input_data) > MAX_BATCH_SIZE:
        logging.warning(f"Input batch size {len(input_data)} exceeds {MAX_BATCH_SIZE}")
        raise HTTPException(status_code=413, detail=f"Batch size exceeds (max={MAX_BATCH_SIZE})")
    results = []
    async with httpx.AsyncClient(verify=False, timeout=20.0) as client:
        for entry in input_data:
            try:
                domain, backlink = entry.domain, entry.backlink
                description, title = entry.description, entry.title
                content = f"{title}, {description}".strip()

                # Classify first 15 characters of title
                title_label, title_score = await classify_text_async(title[:15])
                if title_label == "gambling":
                    results.append(OutputEntry(domain=domain, backlink=backlink, label="Cờ bạc", score=title_score))
                    continue
                if title_label == "movies":
                    result = await check_subpage_context(domain, backlink, title[:15], "Phim lậu", title_score)
                    if result:
                        results.append(result)
                        continue

                # Classify content
                content_label, content_score = await classify_text_async(content)
                if content_label == "gambling":
                    results.append(OutputEntry(domain=domain, backlink=backlink, label="Cờ bạc", score=content_score))
                    continue
                if content_label == "movies":
                    result = await check_subpage_context(domain, backlink, content, "Phim lậu", content_score)
                    if result:
                        results.append(result)
                        continue
                
                # Mark safe if not ecommerce
                if content_label != "ecommerce":
                    results.append(OutputEntry(domain=domain, backlink=backlink, label="An toàn", score=content_score))
                    continue

                homepage_url = f"https://{domain}/"
                if domain in homepage_cache:
                    homepage_label = homepage_cache[domain]
                else:
                    homepage_context = await get_website_context_async(homepage_url)
                    if homepage_context == "inaccessible":
                        results.append(OutputEntry(domain=domain, backlink=backlink, label="An toàn", score=content_score))
                        continue
                    homepage_label, homepage_score = await classify_text_async(homepage_context)
                    homepage_cache[domain] = homepage_label
                    await save_cache()

                if homepage_label not in ["education", "government"]:
                    results.append(OutputEntry(domain=domain, backlink=backlink, label="An toàn", score=content_score))
                    continue
                result = await check_subpage_context(domain, backlink, content, "Quảng cáo bán hàng", content_score)
                if result:
                    results.append(result)
                    continue

            # --- Exception for each entry ---
            except Exception as e:
                logging.error(f"Error processing entry {getattr(entry, 'domain', None)}: {e}")
                results.append(OutputEntry(
                    domain=getattr(entry, 'domain', ''),
                    backlink=getattr(entry, 'backlink', ''),
                    label="Predict error",
                    score=0.0
                ))
    return results

# --- RUN SERVER ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host=config.SERVER_HOST, port=config.SERVER_PORT)