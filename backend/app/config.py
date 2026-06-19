"""
Configuration module for re-skin.
Loads settings from environment variables using python-dotenv.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings:
    """Application settings loaded from environment."""

    # kie.ai (Seedance)
    KIE_API_KEY: str = os.getenv("KIE_API_KEY", "")

    # Google Drive
    GOOGLE_SERVICE_ACCOUNT_FILE: str = os.getenv(
        "GOOGLE_SERVICE_ACCOUNT_FILE", "./secrets/gdrive-sa.json"
    )
    GDRIVE_DEFAULT_FOLDER_ID: str = os.getenv("GDRIVE_DEFAULT_FOLDER_ID", "")

    # Web access
    BASIC_AUTH_USER: str = os.getenv("BASIC_AUTH_USER", "reskin")
    BASIC_AUTH_PASS: str = os.getenv("BASIC_AUTH_PASS", "")

    # Infrastructure
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://redis:6379/0")
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./data/app.db")

    # App configuration
    DEFAULT_RESOLUTION: str = os.getenv("DEFAULT_RESOLUTION", "480p")
    MAX_REFERENCE_IMAGES: int = int(os.getenv("MAX_REFERENCE_IMAGES", "2"))
    SEGMENT_MAX_SECONDS: int = int(os.getenv("SEGMENT_MAX_SECONDS", "15"))


settings = Settings()
