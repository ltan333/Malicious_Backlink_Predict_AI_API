# --- IMPORT LIBRARIES AND FRAMEWORKS ---
import os
from app.cores.logging import logging
from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.models.load_models import load_model
from app.models.load_vncorenlp import load_vncorenlp
from app.services.cache_service import load_cache
from app.services import scraping_service as scraping
from app.routers import routes_auth, routes_predict

# --- FASTAPI APP ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: load model/cache, init optional segmenter, warm up Playwright.
    Shutdown: close Playwright (and anything else).
    """
    # Startup
    try:
        logging.info(f"[startup] running file: {os.path.abspath(__file__)}")
        logging.info("Startup: determining device...")

        logging.info("Loading cache...")
        load_cache()
        logging.info("Cache loaded")

        logging.info("Loading model...")
        load_model()
        logging.info("Model loaded")

        logging.info("Loading segmenter...")
        load_vncorenlp()
        logging.info("Segmenter loaded")

        scraping.init_static_client()
        logging.info("Static HTTP client initialized")
        await scraping.init_browser()
        logging.info("Browser launched")
    except Exception as e:
        logging.exception("Startup failed: %s", e)
        # re-raise to prevent serving in a bad state
        raise

    # Hand control to FastAPI to serve requests
    yield

    # Shutdown
    try:
        await scraping.close_browser()
        logging.info("Browser closed")
        logging.info("Static HTTP client closed")
    except Exception as e:
        logging.exception("Shutdown cleanup failed: %s", e)

app = FastAPI(lifespan=lifespan)
# Mount routers
app.include_router(routes_auth.router)
app.include_router(routes_predict.router)