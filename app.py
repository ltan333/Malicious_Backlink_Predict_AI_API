# app.py
import json
import os
import asyncio
import httpx
from asyncio import Lock as AsyncLock
from typing import List
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from bs4 import BeautifulSoup
import urllib3
from fastapi import FastAPI
from pydantic import BaseModel

# --- CACHE ---
CACHE_FILE = os.path.join("Homepage_Cache", "homepage_cache.json")
# homepage_cache = {}
# cache_lock = AsyncLock()
# cache_updated = False

# # --- MODEL ---
# model = None
# tokenizer = None

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# autocast_ctx = torch.amp.autocast(device_type="cuda") if torch.cuda.is_available() else nullcontext()

# label2id = {
#     "government": 0, "education": 1, "technology": 2, "tourism": 3,
#     "ecommerce": 4, "delivery": 5, "health": 6, "finance": 7,
#     "media": 8, "nonprofit": 9, "gambling": 10, "movies": 11
# }
# id2label = {v: k for k, v in label2id.items()}

# class TextDataset(Dataset):
#     def __init__(self, texts, tokenizer, max_length=64):
#         self.texts = texts
#         self.tokenizer = tokenizer
#         self.max_length = max_length

#     def __len__(self):
#         return len(self.texts)

#     def __getitem__(self, idx):
#         encoding = self.tokenizer(
#             self.texts[idx],
#             max_length=self.max_length,
#             padding="max_length",
#             truncation=True,
#             return_tensors="pt"
#         )
#         return {
#             "input_ids": encoding["input_ids"].squeeze(),
#             "attention_mask": encoding["attention_mask"].squeeze()
#         }

def load_cache():
    global homepage_cache
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            homepage_cache = json.load(f)
    else:
        homepage_cache = {}

async def save_cache():
    async with cache_lock:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(homepage_cache, f, ensure_ascii=False, indent=2)

def load_model():
    global model, tokenizer
    model_path = "Models/phobert_base_v4"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device).eval()

# def classify_batch(texts: List[str]):
#     dataset = TextDataset(texts, tokenizer)
#     loader = DataLoader(dataset, batch_size=32, shuffle=False)
#     results = []
#     with torch.no_grad(), autocast_ctx:
#         for batch in loader:
#             input_ids = batch["input_ids"].to(model.device)
#             attention_mask = batch["attention_mask"].to(model.device)
#             outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#             probs = torch.softmax(outputs.logits, dim=-1)
#             scores, preds = torch.max(probs, dim=-1)
#             for i in range(len(scores)):
#                 results.append((id2label[preds[i].item()], scores[i].item()))
#     return results

# async def classify_text_async(text: str):
#     return classify_batch([text])[0]

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
    for attempt in range(1, max_retries + 1):
        try:
            response = await client.get(url, headers=headers)
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
            return combined if combined else "inaccessible"

        except Exception:
            if attempt == max_retries:
                return "inaccessible"
            await asyncio.sleep(1.5 * (2 ** (attempt - 1)))

# --- FASTAPI APP ---
app = FastAPI()

# class InputEntry(BaseModel):
#     domain: str
#     backlink: str
#     title: str
#     description: str

# class OutputEntry(BaseModel):
#     domain: str
#     backlink: str
#     label: str
#     score: float

@app.on_event("startup")
async def startup_event():
    print(f"Running on device: {device}")
    load_cache()
    load_model()

@app.post("/predict", response_model=List[OutputEntry])
async def predict(input_data: List[InputEntry]):
    results = []
    async with httpx.AsyncClient(verify=False, timeout=5.0) as client:
        for entry in input_data:
            domain, backlink = entry.domain, entry.backlink
            desc_text, title_text = entry.description, entry.title

            # Classify description
            desc_label, desc_score = await classify_text_async(desc_text)
            # Classify title
            title_label, title_score = await classify_text_async(title_text)

            # Shortcut for gambling or movies
            if desc_label in ["gambling", "movies"]:
                label_map = {"gambling": "Cờ bạc", "movies": "Phim lậu"}
                results.append(OutputEntry(domain=domain, backlink=backlink, label=label_map[desc_label], score=desc_score))
                continue
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
                    homepage_label = "An toàn"
                    homepage_score = 1.0
                    homepage_cache[domain] = homepage_label
                    await save_cache()
                    results.append(OutputEntry(domain=domain, backlink=backlink, label="An toàn", score=homepage_score))
                    continue
                homepage_label, homepage_score = await classify_text_async(homepage_context)
                homepage_cache[domain] = homepage_label
                await save_cache()

            if homepage_label in [desc_label, "media"]:
                results.append(OutputEntry(domain=domain, backlink=backlink, label="An toàn", score=homepage_score))
                continue

            subpage_context = await get_website_context_async(backlink, client)
            if subpage_context == "inaccessible":
                results.append(OutputEntry(domain=domain, backlink=backlink, label="Quảng cáo bán hàng", score=1.0))
                continue

            subpage_label, subpage_score = await classify_text_async(subpage_context)
            if subpage_label in [desc_label, "media"]:
                results.append(OutputEntry(domain=domain, backlink=backlink, label="An toàn", score=subpage_score))
            else:
                results.append(OutputEntry(domain=domain, backlink=backlink, label="Quảng cáo bán hàng", score=subpage_score))
    return results

# Expose API: uvicorn app:app --host 0.0.0.0 --port 8000
# http://localhost:8000/docs#/default/predict_predict_post