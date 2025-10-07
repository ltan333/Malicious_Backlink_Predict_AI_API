# --- IMPORT LIBRARIES AND FRAMEWORKS ---
import logging
import sys
from app.cores import config  

# --- LOGGING ---
# Reconfigure console streams to UTF-8
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(config.LOGS_PATH, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
    force=True
)
