from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session
from ..db.database import get_db
from ..api.models import Footprint, User

def get_footprint(footprint_id: int, db: Session = Depends(get_db)) -> Footprint:
    footprint = db.query(Footprint).filter(Footprint.id == footprint_id).first()
    if footprint is None:
        raise HTTPException(status_code=404, detail="Footprint not found")
    return footprint

def get_user(user_id: int, db: Session = Depends(get_db)) -> User:
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user