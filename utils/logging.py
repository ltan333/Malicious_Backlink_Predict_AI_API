# utils/logging.py
import logging

# --- LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.FileHandler("Logs/api_server.log"), logging.StreamHandler()]
)