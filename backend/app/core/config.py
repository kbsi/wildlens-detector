import os

class Config:
    PROJECT_NAME = "Wildlife Footprint Identifier"
    API_V1_STR = "/api/v1"
    SECRET_KEY = os.getenv("SECRET_KEY", "your_secret_key")
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./app.db")
    ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "*").split(",")
    DEBUG = os.getenv("DEBUG", "false").lower() in ("true", "1", "t")