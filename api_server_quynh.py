# --- IMPORT LIBRARIES AND FRAMEWORKS ---
import const as config
import json
import os
import re
import logging
import asyncio
import httpx
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
import tldextract

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
domain_dict = {}

# --- MODEL ---
model = None
tokenizer = None

# --- DEVICE ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autocast_ctx = torch.amp.autocast(device_type="cuda") if torch.cuda.is_available() else nullcontext()

# --- LABEL MAPPING ---
label2id = {
    "gambling": 0, "movies": 1, "ecommerce": 2, "government": 3, "education": 4, "technology": 5,
    "tourism": 6, "health": 7, "finance": 8, "media": 9, "nonprofit": 10, "realestate": 11,
    "services": 12, "industries": 13, "agriculture": 14
}
id2label = {v: k for k, v in label2id.items()}

# --- CUSTOM DATASET ---
class TextDataset(Dataset):
    """
    A custom Dataset class for tokenizing text input to feed into a transformer model.
    Used for batch inference.
    """
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

# --- HOMECACHE ---
def load_cache():
    """
    Load cached homepage labels from a JSON file.
    This helps avoid redundant website fetch/classification.
    """
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

# --- SAVE HOMECACHE ---
async def save_cache():
    """
    Save the current homepage label cache into a JSON file.
    Uses async lock to ensure thread safety during writing.
    """
    try:
        async with cache_lock:
            with open(config.CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(homepage_cache, f, ensure_ascii=False, indent=2)
        logging.info(f"Cache saved to {config.CACHE_FILE}")
    except Exception as e:
        logging.error(f"Failed to save cache: {e}")
        print("[ERROR] Failed to save cache. Check logs for details.")

# --- LOAD MODEL ---
def load_model():
    """
    Load the pre-trained classification model and tokenizer from local directory.
    This is called once during app startup.
    """
    global model, tokenizer
    model_path = r"Models\phobert_base_v7"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True).to(device).eval()
        logging.info(f"Model loaded from {model_path}")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        print("[ERROR] Failed to load model. Check logs for details.")

# --- CLASSIFY BATCH ---
async def classify_batch(texts: List[str]):
    """
    Perform batch classification of text inputs using the loaded model.
    Returns a list of (predicted_label, confidence_score) tuples.
    """
    dataset = TextDataset(texts, tokenizer)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, pin_memory=torch.cuda.is_available())
    results = []
    with torch.inference_mode(), autocast_ctx:
        for batch in loader:
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1)
            scores, preds = torch.max(probs, dim=-1)
            for i in range(len(scores)):
                results.append((id2label[preds[i].item()], scores[i].item()))
    return results

# --- CLEAN TEXT ---
def clean_text(text):   
    text = text.lower()

    # Preserve domain dots, decimal dots, and URL hyphens
    text = re.sub(r'(\w)\.(?=\w)', r'\1<DOMAIN>', text)
    text = re.sub(r'(\d)\.(?=\d)', r'\1<DECIMAL>', text)
    text = re.sub(r'(\w)-(?=\w)', r'\1<HYPHEN>', text)

    # Remove remaining dots and hyphens
    text = text.replace('.', '')
    text = text.replace('-', '')

    # Replace one or more underscores with a single space
    text = re.sub(r'_+', ' ', text)

    # Restore preserved characters
    text = text.replace('<DOMAIN>', '.')
    text = text.replace('<DECIMAL>', '.')
    text = text.replace('<HYPHEN>', '-')

    # Handle commas
    text = re.sub(r'(?<=[a-z0-9]),(?=[a-z])', ' ', text)
    text = re.sub(r'(?<=[a-z]),(?=[0-9])', ' ', text)
    text = re.sub(r',(?=\D)|(?<=\D),', '', text)

    # Remove unwanted punctuation (keep quotes, %, /)
    text = re.sub(r'[^\w\s\.,/%"]', '', text)

    # Normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# --- SCRAPER ---
# --- CHECK URL ---
def is_accessible(url, timeout=5):
    """
    Check if a given URL is accessible (status code < 400).
    Sends a HEAD request using httpx.
    """
    try:
        response = httpx.head(url, timeout=timeout, follow_redirects=True, verify=False)
        return response.status_code < 400
    except:
        return False

# --- EXTRACT PDF TEXT ---
def extract_pdf_text(pdf_bytes):
    """
    Extract plain text from a PDF file given its byte content.
    Uses PyMuPDF (fitz) to read the PDF pages.
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        return "\n".join(page.get_text() for page in doc).strip()
    except:
        return ""
    
# --- FETCH PDF HTTPX---
def fetch_pdf_httpx(url, timeout=10):
    """
    Attempt to fetch a PDF file from a URL using httpx.
    Returns the PDF bytes if valid, else None.
    """
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

# --- FETCH PDF PLAYWRIGHT ---
async def fetch_pdf_playwright(url, timeout=20):
    """
    Use Playwright to render the webpage and fetch PDF content if available.
    Useful if PDF is dynamically generated via JS.
    """
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            try:
                context = await browser.new_context(ignore_https_errors=True)
                page = await context.new_page()
                await page.goto(url, timeout=timeout * 1000)
                try:
                    await page.wait_for_load_state("networkidle", timeout=5000)
                except:
                    await page.wait_for_timeout(4000)
                html = await page.content()
                return html
            finally:
                await browser.close()
    except:
        pass
    return None

# --- HANDLE PDF ---
async def handle_pdf(url):
    """
    Try to fetch PDF content from a URL using httpx or Playwright.
    Extract and return the text content, or 'inaccessible' if it fails.
    """
    pdf_data = fetch_pdf_httpx(url) or await fetch_pdf_playwright(url)
    if pdf_data:
        text = extract_pdf_text(pdf_data)
        return text if text else "inaccessible"
    return "inaccessible"

# --- FETCH STATIC HTML ---
def fetch_static_html(url, timeout=10):
    """
    Fetch static HTML content of a webpage using httpx.
    Returns the raw HTML or an empty string on failure.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        with httpx.Client(follow_redirects=True, verify=False, timeout=timeout, headers=headers) as client:
            r = client.get(url)
            return r.text
    except:
        return ""
    
playwright_semaphore = asyncio.Semaphore(2)

# --- FETCH DYNAMIC HTML ---
async def fetch_dynamic_html(url, timeout=10):
    """
    Fetch HTML content rendered dynamically (SPA) using Playwright.
    Waits for content to load before returning.
    """
    try:
        async with playwright_semaphore:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await (await browser.new_context(ignore_https_errors=True)).new_page()
                await page.goto(url, timeout=timeout * 1000)
                await page.wait_for_timeout(4000)
                html = await page.content()
                await browser.close()
                return html
    except:
        return ""

# --- HANDLE HTML ---
async def handle_html(url):
    """
    Extract clean text from HTML content.
    Fallbacks to dynamic rendering if site uses frameworks like React/Vue.
    Removes scripts, styles, and hidden elements.
    """
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

# --- GET WEBSITE CONTEXT ---
async def get_website_context_async(url):
    """
    Unified function to return website's textual content.
    Decides whether the URL is a PDF or HTML and handles accordingly.
    """
    if not is_accessible(url):
        return "inaccessible"

    ext = Path(urlparse(url).path).suffix.lower()
    if not ext:
        ext = ".html"

    if ext == ".pdf":
        result = await handle_pdf(url)
    else:
        result = await handle_html(url)

    return result
    
# --- OVERLAP CHECK ---
def overlap_check(text1, text2):
    """
    Check how much overlap there is between two texts.
    If the overlap ratio is greater than 85%, treat them as similar.
    """
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    overlap = words1.intersection(words2)
    non_overlap = words2 - words1
    ratio = len(overlap) / len(words2) if words2 else 0
    print(ratio, words2)
    print("non_overlap:", non_overlap)
    print(words1)
    return (ratio > 0.85)

# --- CHECK SUBPAGE CONTEXT ---
async def check_subpage_context(domain, backlink, text, final_label, score):
    """
    Fetch the backlink page and compare its content with the predicted text.
    If content is similar, mark as 'safe', else keep original label.
    """
    subpage_context = clean_text(await get_website_context_async(backlink))
    if subpage_context == "inaccessible":
        return OutputEntry(domain=domain, backlink=backlink, label=final_label, score=score)
    
    if overlap_check(subpage_context, text):
        return OutputEntry(domain=domain, backlink=backlink, label="An toàn", score=score)
    
    return OutputEntry(domain=domain, backlink=backlink, label=final_label, score=score)

# --- PROCESS DOAMIN ---
async def process_domain(domain):
    if domain in domain_dict:
        return domain_dict[domain]
    ext = tldextract.extract(domain)
    normalized = f"{ext.domain}.{ext.suffix}"
    domain_dict[domain] = normalized
    return normalized

# --- FASTAPI APP ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI's lifespan context to initialize resources on startup.
    Loads model and cache before serving any requests.
    """
    logging.info(f"Running on device with: {device}")
    load_cache()
    load_model()
    yield

app = FastAPI(lifespan=lifespan)

# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="get-access-token")
http_bearer = HTTPBearer()

# --- SCHEMAS MODELS ---
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

# --- CREATE ACCESS TOKEN ---
async def create_access_token(data: dict, expires_delta: timedelta):
    """
    Generate a JWT access token with custom expiration.
    Encodes user data and expiry time into token.
    """
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + expires_delta
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, config.API_SECRET_KEY, algorithm=config.ALGORITHM)
    return encoded_jwt

# --- VERIFY ACCESS TOKEN ---
# async def verify_access_token(token: str = Depends(oauth2_scheme)):
async def verify_access_token(credentials: HTTPAuthorizationCredentials = Depends(http_bearer)):
    """
    Decode and validate the JWT access token.
    Raises error if the token is invalid or expired.
    """
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
# --- ROOT ---
# Root endpoint
@app.get("/")
async def root():
    """
    Root API route. Returns a welcome message and current server version.
    """
    return {"message": "Welcome to the THD AI Model API!", "version": config.SERVER_VERSION}

# --- LOGIN FOR ACCESS TOKEN ---
# Endpoint to get access token
@app.post("/get-access-token", response_model=Token)
async def login_for_access_token(secret_input: SecretKeyInput):
    """
    Login endpoint to obtain an access token.
    Checks if the provided API key matches the server's key.
    """
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

# --- GET TOKEN INFO ---
# Endpoint to get token
@app.get("/get-token-info")
async def get_token_info(current_user: str = Depends(verify_access_token)):
    """
    Verifies the provided token and returns user information and expiration time.
    """
    return {"message": "Token is valid", "exp": datetime.fromtimestamp(current_user.get("exp")), "user": current_user.get("sub")}

# --- PREDICT ENDPOINT ---
# Endpoint to predict
@app.post("/predict", response_model=List[OutputEntry], dependencies=[Depends(verify_access_token)])
async def predict(input_data: List[InputEntry]):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please try again later.")

    MAX_BATCH_SIZE = config.MAX_BATCH_SIZE
    if len(input_data) > MAX_BATCH_SIZE:
        raise HTTPException(status_code=413, detail=f"Batch size exceeds (max={MAX_BATCH_SIZE})")

    # --- STEP 1: Prepare domains and tasks concurrently ---
    unique_domains = await asyncio.gather(*(process_domain(entry.domain) for entry in input_data))
    unique_domains_set = set(unique_domains)

    homepage_tasks = {
        domain: asyncio.create_task(get_website_context_async(f"https://{domain}/"))
        for domain in unique_domains_set if domain not in homepage_cache
    }

    # Start classification tasks for titles and contents concurrently
    titles_all = [clean_text(entry.title) for entry in input_data]
    contents_all = [f"{clean_text(entry.title)} {clean_text(entry.description)}".strip() for entry in input_data]

    title_classify_task = asyncio.create_task(classify_batch(titles_all))
    content_classify_task = asyncio.create_task(classify_batch(contents_all))

    # Await homepage fetch concurrently
    homepage_results = await asyncio.gather(*homepage_tasks.values(), return_exceptions=True)
    homepage_context_map = dict(zip(homepage_tasks.keys(), homepage_results))

    # --- STEP 2: Classify and cache accessible homepages in one batch ---
    homepages_to_classify = [
        (domain, context)
        for domain, context in homepage_context_map.items()
        if not isinstance(context, Exception) and context != "inaccessible"
    ]

    if homepages_to_classify:
        homepage_labels = await classify_batch([ctx for _, ctx in homepages_to_classify])
        for (domain, _), (label, _) in zip(homepages_to_classify, homepage_labels):
            homepage_cache[domain] = label

    await save_cache()

    # --- STEP 3: Finish title and content classification ---
    title_results, content_results = await asyncio.gather(title_classify_task, content_classify_task)

    results = [None] * len(input_data)

    # Map homepage gambling labels
    for i, (entry, domain) in enumerate(zip(input_data, unique_domains)):
        label = homepage_cache.get(domain)
        if label == "gambling":
            results[i] = OutputEntry(domain=domain, backlink=entry.backlink, label="Cờ bạc", score=1.0)

    # --- STEP 4: Use classification results for remaining entries ---
    subpage_promises = []
    for i, (entry, domain) in enumerate(zip(input_data, unique_domains)):
        if results[i] is not None:
            continue

        title_label, title_score = title_results[i]
        content_label, content_score = content_results[i]

        if title_label == "gambling":
            subpage_promises.append(check_subpage_context(domain, entry.backlink, titles_all[i], "Cờ bạc", title_score))
            continue
        if title_label == "movies":
            subpage_promises.append(check_subpage_context(domain, entry.backlink, titles_all[i], "Phim lậu", title_score))
            continue

        if content_label == "gambling":
            subpage_promises.append(check_subpage_context(domain, entry.backlink, contents_all[i], "Cờ bạc", content_score))
            continue
        if content_label == "movies":
            subpage_promises.append(check_subpage_context(domain, entry.backlink, contents_all[i], "Phim lậu", content_score))
            continue

        results[i] = OutputEntry(domain=domain, backlink=entry.backlink, label="An toàn", score=content_score)

    if subpage_promises:
        subpage_results = await asyncio.gather(*subpage_promises, return_exceptions=True)
        for res in subpage_results:
            if isinstance(res, OutputEntry):
                for j in range(len(results)):
                    if results[j] is None:
                        results[j] = res
                        break

    for i, res in enumerate(results):
        if res is None:
            entry = input_data[i]
            domain = unique_domains[i]
            results[i] = OutputEntry(domain=domain, backlink=entry.backlink, label="Predict error", score=0.0)

     # --- STEP 5: Adjust labels for low spam ratio ---
    spam_labels = ["Cờ bạc", "Phim lậu", "Quảng cáo bán hàng"]
    spam_count = sum(1 for r in results if r.label in spam_labels)

    print("spam count:", spam_count)
    print("length of input_data:", len(input_data))
    print("spam ratio:", spam_count / len(input_data))

    if spam_count / len(input_data) <= 0.15:
        for r in results:
            # Do not override gambling that skipped title/content checks
            domain = await process_domain(r.domain)
            homepage_label = homepage_cache.get(domain)
            if homepage_label != "gambling" and r.label in spam_labels:
                r.label = "An toàn"

    return results

# --- RUN SERVER ---
if __name__ == "__main__":
    """
    Run the FastAPI server using Uvicorn.
    """
    import uvicorn
    uvicorn.run("api_server_quynh:app", host=config.SERVER_HOST, port=config.SERVER_PORT)