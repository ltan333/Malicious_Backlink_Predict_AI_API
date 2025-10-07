# Pydantic schemas
from pydantic import BaseModel

class InputEntry(BaseModel):
    domain: str
    backlink: str
    title: str
    description: str

class OutputEntry(BaseModel):
    domain: str
    backlink: str
    label: str
    score: float

class Token(BaseModel):
    access_token: str
    token_type: str

class SecretKeyInput(BaseModel):
    api_key: str
