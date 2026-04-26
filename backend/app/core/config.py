from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path
 
 
class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    Pydantic automatically reads from .env file and validates types.
    """
 
    # ── Application ────────────────────────────────────────────
    APP_NAME: str = "AA-LSTM-AEA Pest Prediction API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    API_PREFIX: str = "/api"
 
    # ── Database ───────────────────────────────────────────────
    # TimescaleDB = PostgreSQL with time-series extension
    POSTGRES_HOST: str = "db"          # Docker service name
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "pest_prediction"
    POSTGRES_USER: str = "pest_admin"
    POSTGRES_PASSWORD: str             # Required — no default
 
    @property
    def DATABASE_URL(self) -> str:
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:"
            f"{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:"
            f"{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )
 
    @property
    def DATABASE_URL_SYNC(self) -> str:
        """Synchronous URL for Alembic migrations."""
        return (
            f"postgresql://{self.POSTGRES_USER}:"
            f"{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:"
            f"{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )
 
    # ── Redis ──────────────────────────────────────────────────
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
 
    @property
    def REDIS_URL(self) -> str:
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
 
    # ── Celery ─────────────────────────────────────────────────
    @property
    def CELERY_BROKER_URL(self) -> str:
        return self.REDIS_URL
 
    @property
    def CELERY_RESULT_BACKEND(self) -> str:
        return self.REDIS_URL
 
    # ── Model paths ────────────────────────────────────────────
    MODEL_PATH: str = "/app/weights/aa_lstm_aea_final.keras"
    SCALER_X_PATH: str = "/app/weights/scaler_X.pkl"
    SCALER_Y_PATH: str = "/app/weights/scaler_y.pkl"
    FEATURE_META_PATH: str = "/app/weights/feature_meta.json"
 
    # ── Inference settings ─────────────────────────────────────
    SEQUENCE_LENGTH: int = 14          # 14-day lookback window
    N_FEATURES: int = 84               # Features per timestep
    N_TARGETS: int = 6                 # Number of pest risk outputs
    ANOMALY_WINDOW: int = 7            # Rolling window for anomaly scores
    INFERENCE_BATCH_SIZE: int = 1      # Single-sample inference
 
    # Redis cache TTL for feature windows (24 hours)
    FEATURE_CACHE_TTL_SECONDS: int = 86400
 
    # ── Risk tier thresholds (0–100 scale) ────────────────────
    RISK_TIER_LOW: float = 25.0
    RISK_TIER_MODERATE: float = 50.0
    RISK_TIER_HIGH: float = 75.0
    # Above HIGH = CRITICAL
 
    # ── Weather API (Open-Meteo — free, no key required) ──────
    OPENMETEO_BASE_URL: str = "https://api.open-meteo.com/v1/forecast"
    OPENMETEO_HISTORICAL_URL: str = "https://archive-api.open-meteo.com/v1/archive"
 
    # ── Notification services ──────────────────────────────────
    SMTP_HOST: str = "smtp-relay.brevo.com"
    SMTP_PORT: int = 587
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""
    EMAIL_FROM: str = "noreply@pest-prediction.id"
 
    FONNTE_API_KEY: str = ""           # WhatsApp via Fonnte
    FONNTE_API_URL: str = "https://api.fonnte.com/send"
 
    # ── Security ───────────────────────────────────────────────
    SECRET_KEY: str                    # Required — generate with: openssl rand -hex 32
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
 
    # ── CORS ───────────────────────────────────────────────────
    # Comma-separated list of allowed origins
    CORS_ORIGINS: str = "http://localhost:3000"
 
    @property
    def CORS_ORIGINS_LIST(self) -> list[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]
 
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
 
 
@lru_cache()
def get_settings() -> Settings:
    """
    Cached settings instance.
    Use this function everywhere instead of instantiating Settings directly.
 
    Usage:
        from app.core.config import get_settings
        settings = get_settings()
    """
    return Settings()