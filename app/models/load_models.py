# --- IMPORT LIBRARIES AND FRAMEWORKS ---
import torch
from typing import List
from contextlib import nullcontext
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from app.cores import config
from app.cores.logging import logging
from app.datasets.custom_dataset import TextDataset

# --- MODEL ---
model = None
tokenizer = None
rdrsegmenter = None

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

# --- LOAD MODEL ---
def load_model():
    """
    Load the pre-trained classification model and tokenizer from local directory.
    This is called once during app startup.
    """
    global model, tokenizer

    model_path = config.MODEL_PATH

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True).to(device).eval()
        logging.info(f"Loading model from device: {device}")
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