# Security and authentication logic
from datetime import datetime, timezone, timedelta
from jose import jwt, JWTError, ExpiredSignatureError
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.cores import config
from app.cores.logging import logging

# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="get-access-token")
http_bearer = HTTPBearer()

# --- CREATE ACCESS TOKEN ---
async def create_access_token(data: dict, expires_delta: timedelta):
    """
    Generate a JWT access token with custom expiration.
    Encodes user data and expiry time into token.
    """
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + expires_delta
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, config.API_SECRET_KEY, algorithm=config.ALGORITHM)
    return encoded_jwt

# --- VERIFY ACCESS TOKEN ---
# async def verify_access_token(token: str = Depends(oauth2_scheme)):
async def verify_access_token(credentials: HTTPAuthorizationCredentials = Depends(http_bearer)):
    """
    Decode and validate the JWT access token.
    Raises error if the token is invalid or expired.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate token",
    )
    try:
        token = credentials.credentials
        payload = jwt.decode(token, config.API_SECRET_KEY, algorithms=[config.ALGORITHM])
        print(f"Token payload: {payload}")
        return {
            "sub": payload.get("sub"),
            "exp": payload.get("exp")
        }
    except ExpiredSignatureError:
        logging.warning("Token expired")
        raise HTTPException(status_code=401, detail="Token has expired")
    except JWTError:
        logging.warning("Invalid token")
        raise credentials_exception