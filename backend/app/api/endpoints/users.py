from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List

router = APIRouter()

class User(BaseModel):
    id: int
    username: str
    email: str

users_db = []

@router.post("/users/", response_model=User)
def create_user(user: User):
    if any(u.username == user.username for u in users_db):
        raise HTTPException(status_code=400, detail="Username already registered")
    users_db.append(user)
    return user

@router.get("/users/", response_model=List[User])
def get_users():
    return users_db

@router.get("/users/{user_id}", response_model=User)
def get_user(user_id: int):
    user = next((u for u in users_db if u.id == user_id), None)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user