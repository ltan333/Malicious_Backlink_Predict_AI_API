# --- MAIN ENTRY POINT FOR FASTAPI SERVER ---
def main():
    """
    Run the FastAPI server using Uvicorn.
    """
    # Adjust sys.path when run directly by path
    from pathlib import Path
    import sys
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    # --- IMPORT LIBRARIES AND FRAMEWORKS ---
    import uvicorn
    from app.cores import config
    from app.api.api_server import app
    
    # Run the Uvicorn server with specified host and port from config
    uvicorn.run(app, host=config.SERVER_HOST, port=config.SERVER_PORT)

# --- RUN SERVER ---
if __name__ == "__main__":
    main()