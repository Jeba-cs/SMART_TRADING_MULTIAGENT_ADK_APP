import os
from datetime import datetime, timezone
from typing import Optional

# Google Cloud Platform configuration for ADK Smart Trader

GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "adk-smart-trader-2025")
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
GOOGLE_GENAI_USE_VERTEXAI = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "true").lower() == "true"

GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", None)

FIRESTORE_EMULATOR_HOST = os.getenv("FIRESTORE_EMULATOR_HOST", None)
FIRESTORE_DATABASE_ID = os.getenv("FIRESTORE_DATABASE_ID", "(default)")

BIGQUERY_DATASET_ID = os.getenv("BIGQUERY_DATASET_ID", "smart_trader_analytics")
BIGQUERY_LOCATION = os.getenv("BIGQUERY_LOCATION", GOOGLE_CLOUD_LOCATION)

CLOUD_STORAGE_BUCKET = os.getenv("CLOUD_STORAGE_BUCKET", f"{GOOGLE_CLOUD_PROJECT}-smart-trader-data")

VERTEX_AI_MODEL_NAME = os.getenv("VERTEX_AI_MODEL_NAME", "gemini-2.0-flash-exp")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")
YAHOO_FINANCE_API_KEY = os.getenv("YAHOO_FINANCE_API_KEY", "")

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")

DEBUG = os.getenv("DEBUG", "false").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
MAX_CONCURRENT_AGENTS = int(os.getenv("MAX_CONCURRENT_AGENTS", "8"))
WORKFLOW_TIMEOUT = int(os.getenv("WORKFLOW_TIMEOUT", "600"))
AGENT_RETRY_COUNT = int(os.getenv("AGENT_RETRY_COUNT", "3"))
RATE_LIMIT_REQUESTS_PER_MINUTE = int(os.getenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "50"))

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "")
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", "")
SESSION_TIMEOUT = int(os.getenv("SESSION_TIMEOUT", "7200"))

CACHE_TTL = int(os.getenv("CACHE_TTL", "300"))
MAX_PORTFOLIO_SIZE = float(os.getenv("MAX_PORTFOLIO_SIZE", "1000000"))
MIN_TRADE_AMOUNT = float(os.getenv("MIN_TRADE_AMOUNT", "100"))
RISK_TOLERANCE_LEVELS = os.getenv("RISK_TOLERANCE_LEVELS", "conservative,moderate,aggressive").split(",")


class GCPConfig:
    """GCP Configuration Class"""

    def __init__(self):
        self.project_id = GOOGLE_CLOUD_PROJECT
        self.location = GOOGLE_CLOUD_LOCATION
        self.use_vertex_ai = GOOGLE_GENAI_USE_VERTEXAI
        self.credentials_path = GOOGLE_APPLICATION_CREDENTIALS
        self.gemini_api_key = GEMINI_API_KEY
        self.debug = DEBUG
        self.created_at = datetime.now(timezone.utc)

    def get_firestore_config(self) -> dict:
        """Get Firestore configuration"""
        return {
            "project_id": self.project_id,
            "database_id": FIRESTORE_DATABASE_ID,
            "emulator_host": FIRESTORE_EMULATOR_HOST
        }

    def get_bigquery_config(self) -> dict:
        """Get BigQuery configuration"""
        return {
            "project_id": self.project_id,
            "dataset_id": BIGQUERY_DATASET_ID,
            "location": BIGQUERY_LOCATION
        }

    def get_storage_config(self) -> dict:
        """Get Cloud Storage configuration"""
        return {
            "project_id": self.project_id,
            "bucket_name": CLOUD_STORAGE_BUCKET,
            "location": self.location
        }

    def get_vertex_ai_config(self) -> dict:
        """Get Vertex AI configuration"""
        return {
            "project_id": self.project_id,
            "location": self.location,
            "model_name": VERTEX_AI_MODEL_NAME,
            "use_vertex_ai": self.use_vertex_ai
        }

    def get_api_keys(self) -> dict:
        """Get all API keys configuration"""
        return {
            "alpaca": {
                "api_key": ALPACA_API_KEY,
                "secret_key": ALPACA_SECRET_KEY,
                "base_url": ALPACA_BASE_URL
            },
            "finnhub": FINNHUB_API_KEY,
            "alpha_vantage": ALPHA_VANTAGE_API_KEY,
            "news_api": NEWS_API_KEY,
            "reddit": {
                "client_id": REDDIT_CLIENT_ID,
                "client_secret": REDDIT_CLIENT_SECRET
            },
            "gemini": GEMINI_API_KEY
        }

    @classmethod
    def validate_config(cls) -> bool:
        """Validate essential configuration parameters"""
        required_vars = [GOOGLE_CLOUD_PROJECT]
        has_auth = GEMINI_API_KEY or GOOGLE_APPLICATION_CREDENTIALS

        missing_vars = [var for var in required_vars if not var]

        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")

        if not has_auth:
            raise ValueError("Either GEMINI_API_KEY or GOOGLE_APPLICATION_CREDENTIALS must be provided")

        return True

    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.debug or os.getenv("ENVIRONMENT", "development") == "development"

    def get_logging_config(self) -> dict:
        """Get logging configuration"""
        return {
            "level": LOG_LEVEL,
            "debug": self.debug,
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }


def validate_config() -> bool:
    """Validate essential configuration parameters"""
    return GCPConfig.validate_config()


__all__ = [
    'GOOGLE_CLOUD_PROJECT', 'GOOGLE_CLOUD_LOCATION', 'GOOGLE_GENAI_USE_VERTEXAI',
    'GCPConfig', 'validate_config'
]
