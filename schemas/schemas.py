# models/schemas.py
from pydantic import BaseModel

# Payload token
class Token(BaseModel):
    access_token: str
    token_type: str

# Input token
class SecretKeyInput(BaseModel):
    api_key: str

# Input model
class InputEntry(BaseModel):
    domain: str
    backlink: str
    title: str
    description: str

# Output model
class OutputEntry(BaseModel):
    domain: str
    backlink: str
    label: str
    score: float