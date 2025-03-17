from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Footprint(Base):
    __tablename__ = 'footprints'

    id = Column(Integer, primary_key=True, index=True)
    species_name = Column(String, index=True)
    confidence = Column(Float)
    image_url = Column(String)

class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)