# main.py
import uvicorn
from config import const as config

# --- RUN SERVER ---
if __name__ == "__main__":
    """
    Run the FastAPI server using Uvicorn.
    """
    uvicorn.run("api.api_server:app", host=config.SERVER_HOST, port=config.SERVER_PORT)