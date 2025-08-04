# utils/text_utils.py
import re
import tldextract
from cache import domain_dict

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
    print("non-overlapping words:", non_overlap)
    print(words1)
    return (ratio > 0.85)

async def process_domain(domain):
    if domain in domain_dict:
        return domain_dict[domain]
    ext = tldextract.extract(domain)
    normalized = f"{ext.domain}.{ext.suffix}"
    domain_dict[domain] = normalized
    return normalized