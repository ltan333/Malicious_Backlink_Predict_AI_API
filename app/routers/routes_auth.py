# --- IMPORT LIBRARIES AND FRAMEWORKS ---
from datetime import datetime, timedelta
from fastapi import Depends, HTTPException, status, APIRouter

from app.cores import config
from app.schemas.schemas import Token, SecretKeyInput
from app.cores.security import create_access_token, verify_access_token

router = APIRouter(prefix="", tags=["auth"])

# --- ROUTES ---
# --- ROOT ---
# Root endpoint
@router.get("/")
async def root():
    """
    Root API route. Returns a welcome message and current server version.
    """
    return {"message": "Welcome to the THD AI Model API!", "version": config.SERVER_VERSION}

# --- LOGIN FOR ACCESS TOKEN ---
# Endpoint to get access token
@router.post("/get-access-token", response_model=Token)
async def login_for_access_token(secret_input: SecretKeyInput):
    """
    Login endpoint to obtain an access token.
    Checks if the provided API key matches the server's key.
    """
    if secret_input.api_key != config.API_SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    access_token_expires = timedelta(seconds=config.ACCESS_TOKEN_EXPIRE_SECOND)
    access_token = await create_access_token(
        data={"sub": "authenticated_user"}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# --- GET TOKEN INFO ---
# Endpoint to get token
@router.get("/get-token-info")
async def get_token_info(current_user: str = Depends(verify_access_token)):
    """
    Verifies the provided token and returns user information and expiration time.
    """
    return {"message": "Token is valid", "exp": datetime.fromtimestamp(current_user.get("exp")), "user": current_user.get("sub")}
