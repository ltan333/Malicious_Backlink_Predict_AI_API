# app.py
import const as config
import json
import os
import logging
import asyncio
import httpx
import urllib3
import torch
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
    handlers=[logging.FileHandler("api_server.log"), logging.StreamHandler()]
)

# --- CACHE ---
homepage_cache = {}
cache_lock = AsyncLock()
cache_updated = False

# --- MODEL ---
model = None
tokenizer = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autocast_ctx = torch.amp.autocast(device_type="cuda") if torch.cuda.is_available() else nullcontext()

label2id = {
    "government": 0, "education": 1, "technology": 2, "tourism": 3,
    "ecommerce": 4, "delivery": 5, "health": 6, "finance": 7,
    "media": 8, "nonprofit": 9, "gambling": 10, "movies": 11
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
    model_path = "Models/phobert_base_v4"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True).to(device).eval()
        logging.info(f"Model loaded from {model_path}")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        print("[ERROR] Failed to load model. Check logs for details.")

def classify_batch(texts: List[str]):
    dataset = TextDataset(texts, tokenizer)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
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
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

async def get_website_context_async(url: str, client: httpx.AsyncClient, max_retries=2):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/114.0.0.0 Safari/537.36"
        )
    }
    # Scan 10s timeout
    timeout = httpx.Timeout(10.0, connect=5.0)
    for attempt in range(1, max_retries + 1):
        try:
            response = await client.get(url, headers=headers, timeout=timeout)
            if response.status_code != 200:
                raise httpx.HTTPStatusError(f"Status {response.status_code}", request=response.request, response=response)

            raw_html = response.text
            soup = BeautifulSoup(raw_html, "html.parser")
            title = soup.find("title").text.strip() if soup.find("title") else ""
            meta_desc = ""
            for meta in soup.find_all("meta"):
                if "description" in meta.get("name", "").lower():
                    meta_desc = meta.get("content", "").strip()
                    break

            combined = f"{title}. {meta_desc}".strip()
            logging.info(f"Scraped context: {combined}")
            return combined if combined else "inaccessible"

        except (httpx.TimeoutException, httpx.RequestError) as e:
            logging.warning(f"Timeout or network error scraping {url} (attempt {attempt}): {e}")
        except Exception as e:
            logging.error(f"Error scraping {url} (attempt {attempt}): {e}")
        if attempt == max_retries:
            return "inaccessible"
        await asyncio.sleep(1.5 * (2 ** (attempt - 1)))

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

# Input khi lấy token
class SecretKeyInput(BaseModel):
    api_key: str

class InputEntry(BaseModel):
    domain: str
    backlink: str
    title: str
    description: str

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

# Endpoint để lấy access token
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

# Endpoint để lấy thông tin token
@app.get("/get-token-info")
async def get_token_info(current_user: str = Depends(verify_access_token)):
    return {"message": "Token is valid", "exp": datetime.fromtimestamp(current_user.get("exp")), "user": current_user.get("sub")}

# Endpoint để dự đoán
@app.post("/predict", response_model=List[OutputEntry], dependencies=[Depends(verify_access_token)])
async def predict(input_data: List[InputEntry]):
    # Kiểm tra model/tokenizer đã load thành công chưa, có trường hợp model load lỗi nhưng endpoint vẫn lên
    if model is None or tokenizer is None:
        logging.error("Model or tokenizer not loaded. Cannot process prediction.")
        raise HTTPException(status_code=503, detail="Model not loaded. Please try again later.")
    # Giới hạn số lượng input tránh quá tải RAM/GPU - Chỉ nhận các gói tin nhỏ hơn MAX_BATCH_SIZE
    MAX_BATCH_SIZE = config.MAX_BATCH_SIZE
    if len(input_data) > MAX_BATCH_SIZE:
        logging.warning(f"Input batch size {len(input_data)} exceeds {MAX_BATCH_SIZE}")
        raise HTTPException(status_code=413, detail=f"Batch size exceeds (max={MAX_BATCH_SIZE})")
    results = []
    async with httpx.AsyncClient(verify=False, timeout=20.0) as client:
        for entry in input_data:
            try:
                domain, backlink = entry.domain, entry.backlink
                desc_text, title_text = entry.description, entry.title

                # Classify description
                desc_label, desc_score = await classify_text_async(desc_text)
                if desc_label in ["gambling", "movies"]:
                    label_map = {"gambling": "Cờ bạc", "movies": "Phim lậu"}
                    results.append(OutputEntry(domain=domain, backlink=backlink, label=label_map[desc_label], score=desc_score))
                    continue

                # Classify title
                title_label, title_score = await classify_text_async(title_text)
                if title_label in ["gambling", "movies"]:
                    label_map = {"gambling": "Cờ bạc", "movies": "Phim lậu"}
                    results.append(OutputEntry(domain=domain, backlink=backlink, label=label_map[title_label], score=title_score))
                    continue

                # Safe categories
                safe_categories = ["government", "education", "technology", "health", "finance", "media", "nonprofit"]
                if desc_label in safe_categories:
                    results.append(OutputEntry(domain=domain, backlink=backlink, label="An toàn", score=desc_score))
                    continue

                homepage_url = f"https://{domain}/"
                if domain in homepage_cache:
                    homepage_label = homepage_cache[domain]
                    homepage_score = 1.0
                else:
                    homepage_context = await get_website_context_async(homepage_url, client)
                    if homepage_context == "inaccessible":
                        results.append(OutputEntry(domain=domain, backlink=backlink, label="An toàn", score=desc_score))
                        continue
                    homepage_label, homepage_score = await classify_text_async(homepage_context)
                    homepage_cache[domain] = homepage_label
                    await save_cache()

                if homepage_label in [desc_label, "media", "An toàn"]:
                    results.append(OutputEntry(domain=domain, backlink=backlink, label="An toàn", score=desc_score))
                    continue

                subpage_context = await get_website_context_async(backlink, client)
                if subpage_context == "inaccessible":
                    results.append(OutputEntry(domain=domain, backlink=backlink, label="Quảng cáo bán hàng", score=desc_score))
                    continue

                subpage_label, subpage_score = await classify_text_async(subpage_context)
                if subpage_label in [desc_label, "media"]:
                    results.append(OutputEntry(domain=domain, backlink=backlink, label="An toàn", score=desc_score))
                else:
                    results.append(OutputEntry(domain=domain, backlink=backlink, label="Quảng cáo bán hàng", score=desc_score))
            # --- Xử lý lỗi --- Trong trường hợp có lỗi trong quá trình xử lý từng entry
            except Exception as e:
                logging.error(f"Error processing entry {getattr(entry, 'domain', None)}: {e}")
                results.append(OutputEntry(
                    domain=getattr(entry, 'domain', ''),
                    backlink=getattr(entry, 'backlink', ''),
                    label="Predict error",
                    score=0.0
                ))
    return results

# Expose API: uvicorn app:app --host 0.0.0.0 --port 8000
# http://localhost:8000/docs#/default/predict_predict_post

# --- RUN SERVER ---
# API_Server:app - phải trùng với tên file
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("API_Server:app", host=config.SERVER_HOST, port=config.SERVER_PORT, reload=True)