from sqlalchemy.orm import Session
from typing import List
from ..db.models import Footprint, User

class FootprintRepository:
    def __init__(self, db: Session):
        self.db = db

    def create_footprint(self, footprint: Footprint) -> Footprint:
        self.db.add(footprint)
        self.db.commit()
        self.db.refresh(footprint)
        return footprint

    def get_footprint(self, footprint_id: int) -> Footprint:
        return self.db.query(Footprint).filter(Footprint.id == footprint_id).first()

    def get_all_footprints(self) -> List[Footprint]:
        return self.db.query(Footprint).all()

    def delete_footprint(self, footprint_id: int) -> None:
        footprint = self.get_footprint(footprint_id)
        if footprint:
            self.db.delete(footprint)
            self.db.commit()

class UserRepository:
    def __init__(self, db: Session):
        self.db = db

    def create_user(self, user: User) -> User:
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user

    def get_user(self, user_id: int) -> User:
        return self.db.query(User).filter(User.id == user_id).first()

    def get_all_users(self) -> List[User]:
        return self.db.query(User).all()

    def delete_user(self, user_id: int) -> None:
        user = self.get_user(user_id)
        if user:
            self.db.delete(user)
            self.db.commit()