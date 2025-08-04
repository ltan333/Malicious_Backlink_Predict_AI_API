# utils/cache.py
import json
import os
import logging
from asyncio import Lock as AsyncLock
from config import const as config

# --- CACHE ---
homepage_cache = {}
cache_lock = AsyncLock()
cache_updated = False
domain_dict = {}

def load_cache():
    """
    Load cached homepage labels ('government', 'education') from a JSON file.
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