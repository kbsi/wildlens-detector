from pydantic import BaseModel
from typing import Optional

class Footprint(BaseModel):
    id: Optional[int]
    species_name: str
    confidence: float
    image_url: str
    created_at: Optional[str] = None

class User(BaseModel):
    id: Optional[int]
    username: str
    email: str
    created_at: Optional[str] = None