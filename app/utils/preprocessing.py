# --- IMPORT LIBRARIES AND FRAMEWORKS ---
import re
from app.models.load_vncorenlp import rdrsegmenter

# --- SENTINELS ---
PDF_EMPTY_SENTINEL = "<PDF_EMPTY_BUT_ACCESSIBLE>"
PDF_FETCH_SENTINEL = "<PDF_FETCH_FAILED>"
CLOUDFLARE_BLOCKED = "<CLOUDFLARE_BLOCKED>"
NOT_FOUND = "<HTTP_404_NOT_FOUND>"
HTTP_5XX_SAFE = "<HTTP_5XX_SAFE>"
EXTERNAL_REDIRECT = "<EXTERNAL_REDIRECT>"
INACCESSIBLE = "<INACCESSIBLE>"
SENTINELS = {CLOUDFLARE_BLOCKED, PDF_EMPTY_SENTINEL, PDF_FETCH_SENTINEL, NOT_FOUND, HTTP_5XX_SAFE, EXTERNAL_REDIRECT, INACCESSIBLE}

# --- SEGMENT AND CLEAN TEXT ---
def clean_text(text):   
    # Preserve domain dots, decimal dots, and URL hyphens
    text = re.sub(r'(\w)\.(?=\w)', r'\1<DOMAIN>', text)
    text = re.sub(r'(\d)\.(?=\d)', r'\1<DECIMAL>', text)
    text = re.sub(r'(\w)-(?=\w)', r'\1<HYPHEN>', text)
    # Remove remaining dots and hyphens
    text = text.replace('.', ' ')
    text = text.replace('-', ' ')
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

def is_sentinel(text: str) -> bool:
    """Check if text is a sentinel value"""
    return isinstance(text, str) and text in SENTINELS

def preprocess_text(text: str) -> str:
    """Segments Vietnamese text and cleans it for inference."""
    if is_sentinel(text):
        return text
    text = clean_text(text)
    if rdrsegmenter:
        text = ' '.join(rdrsegmenter.word_segment(text))
    return text