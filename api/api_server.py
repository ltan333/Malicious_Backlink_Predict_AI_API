# api/main.py
from fastapi import FastAPI
from contextlib import asynccontextmanager
from ..utils.logging import logging
from ..config import const as config
from ..utils.cache import load_cache
from ..utils.model import load_model, device
from .routes import login_for_access_token, get_token_info, predict

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI's lifespan context to initialize resources on startup.
    Loads model and cache before serving any requests.
    """
    logging.info(f"Running on device with: {device}")
    load_cache()
    load_model()
    yield

app = FastAPI(lifespan=lifespan)

# Root endpoint
@app.get("/")
async def root():
    """
    Root API route. Returns a welcome message and current server version.
    """
    return {"message": "Welcome to the THD AI Model API!", "version": config.SERVER_VERSION}

# Register routes
app.post("/get-access-token")(login_for_access_token)
app.get("/get-token-info")(get_token_info)
app.post("/predict")(predict)