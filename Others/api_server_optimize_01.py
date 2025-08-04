# --- IMPORT LIBRARIES AND FRAMEWORKS ---
import logging
import re
import asyncio
import torch

from typing import List
from datetime import datetime, timezone, timedelta
from contextlib import nullcontext, asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from jose import jwt, JWTError, ExpiredSignatureError

import const as config

# --- LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.FileHandler("Logs/api_server.log"), logging.StreamHandler()]
)

# --- DEVICE ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autocast_ctx = torch.amp.autocast(device_type="cuda") if torch.cuda.is_available() else nullcontext()

# --- MODEL ---
model = None
tokenizer = None

# --- LABEL MAPPING ---
label2id = {
    "gambling": 0, "movies": 1, "ecommerce": 2, "government": 3, "education": 4, "technology": 5,
    "tourism": 6, "health": 7, "finance": 8, "media": 9, "nonprofit": 10, "realestate": 11,
    "services": 12, "industries": 13, "agriculture": 14
}
id2label = {v: k for k, v in label2id.items()}

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

# --- FASTAPI APP ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI's lifespan context to initialize resources on startup.
    Loads model and cache before serving any requests.
    """
    logging.info(f"Running on device with: {device}")
    load_model()
    yield

app = FastAPI(lifespan=lifespan)

# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="get-access-token")
http_bearer = HTTPBearer()

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

# --- ROUTES ---
# --- ROOT ENDPOINT ---
@app.get("/")
async def root():
    """
    Root API route. Returns a welcome message and current server version.
    """
    return {
        "message": "Welcome to the THD AI Model API!",
        "version": config.SERVER_VERSION
    }

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
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(input_data) > config.MAX_BATCH_SIZE:
        raise HTTPException(status_code=413, detail=f"Max batch size exceeded ({config.MAX_BATCH_SIZE})")

    titles = [clean_text(e.title) for e in input_data]
    contents = [clean_text(f"{e.title} {e.description}") for e in input_data]

    title_results, content_results = await asyncio.gather(
        classify_batch(titles),
        classify_batch(contents)
    )

    results = []
    for entry, (label, score) in zip(input_data, content_results):
        results.append(OutputEntry(
            domain=entry.domain,
            backlink=entry.backlink,
            label="Cờ bạc" if label == "gambling" else "Phim lậu" if label == "movies" else "An toàn",
            score=score
        ))
    return results

# --- RUN SERVER ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server_optimize_01:app", host=config.SERVER_HOST, port=config.SERVER_PORT)