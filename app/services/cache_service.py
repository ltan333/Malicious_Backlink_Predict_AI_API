# Cache handling logic
import os
import json
from asyncio import Lock as AsyncLock

from app.cores.logging import logging
from app.cores import config

# --- CACHE ---
homepage_cache = {}
cache_lock = AsyncLock()

# --- HOMECACHE ---
def load_cache():
    """
    Load cached homepage labels from a JSON file.
    This helps avoid redundant website fetch/classification.
    """
    global homepage_cache

    cache_file = config.HOME_CACHE_PATH
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            homepage_cache.clear()
            homepage_cache.update(data)
            logging.info(f"Cache loaded from {cache_file}")
        except (json.JSONDecodeError, OSError) as e:
            logging.error(f"Failed to load cache file: {e}. Starting with an empty cache.")
            homepage_cache.clear()
    else:
        cache_dir = os.path.dirname(cache_file)
        os.makedirs(cache_dir, exist_ok=True)
        homepage_cache.clear()
        logging.info(f"No cache file found at {cache_file}, starting with empty cache.")

# --- SAVE HOMECACHE ---
async def save_cache():
    """
    Save the current homepage label cache into a JSON file.
    Uses async lock to ensure thread safety during writing.
    """
    cache_file = config.HOME_CACHE_PATH
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)

    try:
        async with cache_lock:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(homepage_cache, f, ensure_ascii=False, indent=2)
        logging.info(f"Cache saved to {cache_file}")
    except Exception as e:
        logging.error(f"Failed to save cache: {e}")