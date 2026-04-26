import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, date
from typing import Optional
 
import redis.asyncio as aioredis
from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
import numpy as np
 
from app.core.config import get_settings
from app.core.database import get_db, engine
from app.models.domain import (
    Base, Location, WeatherObservation, PredictionLog
)
from app.services.inference import PestPredictionService, FeatureWindowCache
 
logger   = logging.getLogger(__name__)
settings = get_settings()
 
# ── Global Redis client (shared across requests) ──────────────────────────────
redis_client: Optional[aioredis.Redis] = None
 
 
# ==============================================================================
# LIFESPAN — startup and shutdown
# ==============================================================================
 
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    Runs startup logic before yield, shutdown logic after yield.
 
    Startup:
        1. Create database tables (if not exist)
        2. Connect to Redis
        3. Load AA-LSTM-AEA model into RAM (once)
 
    Shutdown:
        4. Close Redis connection
    """
    global redis_client
 
    # 1. Database tables
    logger.info("Creating database tables if not exist...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables ready.")
 
    # 2. Redis connection
    logger.info(f"Connecting to Redis at {settings.REDIS_URL}...")
    redis_client = aioredis.from_url(
        settings.REDIS_URL,
        encoding="utf-8",
        decode_responses=False,  # Keep bytes for numpy serialization
    )
    await redis_client.ping()
    logger.info("Redis connection established.")
 
    # 3. Load model into RAM
    logger.info("Loading AA-LSTM-AEA model into RAM...")
    PestPredictionService.initialize()
    logger.info("Model ready. Application startup complete.")
 
    yield
 
    # 4. Shutdown
    logger.info("Shutting down — closing Redis connection...")
    await redis_client.close()
    logger.info("Shutdown complete.")
 
 
# ==============================================================================
# FASTAPI APPLICATION
# ==============================================================================
 
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)
 
# CORS — allow Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS_LIST,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)
 
 
# ==============================================================================
# RESPONSE SCHEMAS (Pydantic)
# ==============================================================================
 
class RiskValues(BaseModel):
    bph_risk  : float = Field(..., ge=0, le=100, description="Brown Plant Hopper risk (0–100)")
    ysb_risk  : float = Field(..., ge=0, le=100, description="Yellow Stem Borer risk (0–100)")
    rlf_risk  : float = Field(..., ge=0, le=100, description="Rice Leaf Folder risk (0–100)")
    wst_risk  : float = Field(..., ge=0, le=100, description="White Stem Borer risk (0–100)")
    rat_risk  : float = Field(..., ge=0, le=100, description="Rat risk (0–100)")
    snail_risk: float = Field(..., ge=0, le=100, description="Golden Apple Snail risk (0–100)")
 
 
class RiskTiers(BaseModel):
    bph_risk  : str = Field(..., description="LOW / MODERATE / HIGH / CRITICAL")
    ysb_risk  : str
    rlf_risk  : str
    wst_risk  : str
    rat_risk  : str
    snail_risk: str
 
 
class AnomalyContribution(BaseModel):
    """How much anomaly enhancement changed each prediction vs lambda=0."""
    bph_risk  : float = Field(..., description="Delta risk from anomaly enhancement")
    ysb_risk  : float
    rlf_risk  : float
    wst_risk  : float
    rat_risk  : float
    snail_risk: float
 
 
class PredictionResponse(BaseModel):
    """
    Complete prediction response from AA-LSTM-AEA.
    Includes all explainability outputs for frontend visualization.
    """
    location_id          : int
    prediction_date      : date
    risks                : RiskValues
    risk_tiers           : RiskTiers
    dominant_pest        : str = Field(..., description="Pest with highest risk")
    dominant_risk        : float = Field(..., ge=0, le=100)
    attention_weights    : list[float] = Field(
        ..., description="14 attention weights (oldest→newest day)")
    anomaly_scores       : list[float] = Field(
        ..., description="14 anomaly scores (oldest→newest day)")
    anomaly_contribution : AnomalyContribution
    lambda_anomaly       : float = Field(
        ..., description="Learned anomaly influence weight")
    recommendation       : str = Field(
        ..., description="Plain-language action guidance")
    cache_hit            : bool = Field(
        ..., description="True if feature window served from Redis cache")
    inference_ms         : float = Field(
        ..., description="Model inference latency in milliseconds")
 
 
class HealthResponse(BaseModel):
    status       : str
    model_loaded : bool
    lambda_value : float
    version      : str
 
 
class LocationResponse(BaseModel):
    id        : int
    name      : str
    latitude  : float
    longitude : float
    is_active : bool
 
    class Config:
        from_attributes = True
 
 
# ==============================================================================
# HELPER: build recommendation text
# ==============================================================================
 
PEST_NAMES = {
    "bph_risk"  : "Brown Plant Hopper (Wereng Coklat)",
    "ysb_risk"  : "Yellow Stem Borer (Penggerek Batang Kuning)",
    "rlf_risk"  : "Rice Leaf Folder (Penggulung Daun)",
    "wst_risk"  : "White Stem Borer (Penggerek Batang Putih)",
    "rat_risk"  : "Rat (Tikus Sawah)",
    "snail_risk": "Golden Apple Snail (Keong Mas)",
}
 
def build_recommendation(dominant_pest: str, dominant_risk: float, tier: str) -> str:
    pest_name = PEST_NAMES.get(dominant_pest, dominant_pest)
    if tier == "CRITICAL":
        return (
            f"KRITIS: Risiko {pest_name} sangat tinggi ({dominant_risk:.0f}%). "
            f"Segera lakukan pemantauan lapangan dan pertimbangkan tindakan pengendalian."
        )
    elif tier == "HIGH":
        return (
            f"TINGGI: Risiko {pest_name} tinggi ({dominant_risk:.0f}%). "
            f"Pantau populasi hama di perangkap dan siapkan langkah pengendalian."
        )
    elif tier == "MODERATE":
        return (
            f"SEDANG: Risiko {pest_name} sedang ({dominant_risk:.0f}%). "
            f"Lanjutkan pemantauan rutin dan perhatikan perkembangan cuaca."
        )
    return (
        f"RENDAH: Kondisi saat ini kurang mendukung perkembangan hama. "
        f"Tetap lakukan pemantauan berkala."
    )
 
 
# ==============================================================================
# HELPER: assemble feature window from database
# ==============================================================================
 
async def get_feature_window(
    location_id: int,
    target_date: datetime,
    db: AsyncSession,
) -> Optional[np.ndarray]:
    """
    Query TimescaleDB for the 14-day feature window ending on target_date.
 
    Returns np.ndarray of shape (14, 84) or None if insufficient data.
    """
    start_date = target_date - timedelta(days=settings.SEQUENCE_LENGTH)
 
    result = await db.execute(
        select(WeatherObservation)
        .where(WeatherObservation.location_id == location_id)
        .where(WeatherObservation.time >= start_date)
        .where(WeatherObservation.time < target_date)
        .order_by(WeatherObservation.time.asc())
    )
    rows = result.scalars().all()
 
    if len(rows) < settings.SEQUENCE_LENGTH:
        return None
 
    # Take the most recent SEQUENCE_LENGTH rows
    rows = rows[-settings.SEQUENCE_LENGTH:]
 
    # Build feature matrix — column order must match training
    # feature_meta["feature_cols"] defines the exact order
    service      = PestPredictionService.get_instance()
    feature_cols = service.feature_meta.get("feature_cols", [])
 
    if not feature_cols:
        raise RuntimeError("feature_meta.json missing 'feature_cols'")
 
    # Map SQLAlchemy column names to feature order
    feature_matrix = np.zeros(
        (settings.SEQUENCE_LENGTH, len(feature_cols)), dtype=np.float32)
 
    for t, row in enumerate(rows):
        for f_idx, col_name in enumerate(feature_cols):
            val = getattr(row, col_name, None)
            feature_matrix[t, f_idx] = float(val) if val is not None else 0.0
 
    return feature_matrix
 
 
# ==============================================================================
# ENDPOINTS
# ==============================================================================
 
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint for Docker health checks and monitoring.
    Returns model status and learned lambda value.
    """
    try:
        service      = PestPredictionService.get_instance()
        model_loaded = True
        lambda_val   = service.lambda_value
    except RuntimeError:
        model_loaded = False
        lambda_val   = 0.0
 
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        lambda_value=lambda_val,
        version=settings.APP_VERSION,
    )
 
 
@app.get(
    "/api/locations",
    response_model=list[LocationResponse],
    tags=["Locations"],
)
async def list_locations(db: AsyncSession = Depends(get_db)):
    """Return all active monitoring locations (31 Jember subdistricts)."""
    result = await db.execute(
        select(Location).where(Location.is_active == True)
    )
    return result.scalars().all()
 
 
@app.get(
    "/api/locations/{location_id}",
    response_model=LocationResponse,
    tags=["Locations"],
)
async def get_location(
    location_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Return metadata for a single location."""
    result = await db.execute(
        select(Location).where(Location.id == location_id)
    )
    location = result.scalar_one_or_none()
    if not location:
        raise HTTPException(status_code=404, detail="Location not found.")
    return location
 
 
@app.get(
    "/api/locations/{location_id}/predict",
    response_model=PredictionResponse,
    tags=["Predictions"],
    summary="Get pest risk prediction for a location and date",
)
async def get_prediction(
    location_id  : int,
    target_date  : Optional[date] = Query(
        default=None,
        description="Date to predict for (YYYY-MM-DD). Defaults to today.",
    ),
    db           : AsyncSession = Depends(get_db),
):
    """
    Primary prediction endpoint.
 
    Returns AA-LSTM-AEA predictions for all 6 pest risks,
    plus full explainability outputs (attention weights, anomaly scores,
    anomaly contribution delta).
 
    Feature windows are cached in Redis for 24 hours.
    Inference runs in RAM — typical latency: 150–500ms.
    """
    # Default to today if no date provided
    if target_date is None:
        target_date = datetime.utcnow().date()
 
    target_datetime = datetime.combine(target_date, datetime.min.time())
    date_str        = target_date.isoformat()
 
    # ── 1. Check Redis cache ───────────────────────────────────
    cache       = FeatureWindowCache(redis_client)
    cache_hit   = False
    feature_win = await cache.get(location_id, date_str)
 
    if feature_win is not None:
        cache_hit = True
        logger.debug(
            f"Cache HIT — location={location_id} date={date_str}")
    else:
        # ── 2. Query TimescaleDB ───────────────────────────────
        feature_win = await get_feature_window(
            location_id, target_datetime, db)
 
        if feature_win is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Insufficient weather data for location {location_id} "
                    f"on {date_str}. Need {settings.SEQUENCE_LENGTH} days of "
                    f"observations before target date."
                ),
            )
 
        # Store in Redis for next request
        await cache.set(location_id, date_str, feature_win)
        logger.debug(
            f"Cache MISS — assembled from DB and cached. "
            f"location={location_id} date={date_str}"
        )
 
    # ── 3. Run inference ───────────────────────────────────────
    service = PestPredictionService.get_instance()
    result  = service.predict(feature_win)
 
    # ── 4. Log prediction to database ─────────────────────────
    risks = result["risks"]
    tiers = result["risk_tiers"]
 
    log_entry = PredictionLog(
        location_id         = location_id,
        prediction_date     = target_datetime,
        bph_risk            = risks["bph_risk"],
        ysb_risk            = risks["ysb_risk"],
        rlf_risk            = risks["rlf_risk"],
        wst_risk            = risks["wst_risk"],
        rat_risk            = risks["rat_risk"],
        snail_risk          = risks["snail_risk"],
        bph_tier            = tiers["bph_risk"],
        ysb_tier            = tiers["ysb_risk"],
        rlf_tier            = tiers["rlf_risk"],
        wst_tier            = tiers["wst_risk"],
        rat_tier            = tiers["rat_risk"],
        snail_tier          = tiers["snail_risk"],
        dominant_pest       = result["dominant_pest"],
        dominant_risk       = result["dominant_risk"],
        attention_weights   = result["attention_weights"],
        anomaly_scores      = result["anomaly_scores"],
        lambda_anomaly      = result["lambda_anomaly"],
        anomaly_contribution= result["anomaly_contribution"],
        inference_latency_ms= result["inference_ms"],
        model_version       = "aa_lstm_aea_v1",
    )
    db.add(log_entry)
    await db.commit()
 
    # ── 5. Build recommendation ────────────────────────────────
    recommendation = build_recommendation(
        result["dominant_pest"],
        result["dominant_risk"],
        tiers[result["dominant_pest"]],
    )
 
    # ── 6. Return structured response ─────────────────────────
    return PredictionResponse(
        location_id          = location_id,
        prediction_date      = target_date,
        risks                = RiskValues(**risks),
        risk_tiers           = RiskTiers(**tiers),
        dominant_pest        = result["dominant_pest"],
        dominant_risk        = result["dominant_risk"],
        attention_weights    = result["attention_weights"],
        anomaly_scores       = result["anomaly_scores"],
        anomaly_contribution = AnomalyContribution(
            **result["anomaly_contribution"]),
        lambda_anomaly       = result["lambda_anomaly"],
        recommendation       = recommendation,
        cache_hit            = cache_hit,
        inference_ms         = result["inference_ms"],
    )
 
 
@app.get(
    "/api/locations/{location_id}/history",
    response_model=list[PredictionResponse],
    tags=["Predictions"],
)
async def get_prediction_history(
    location_id: int,
    days       : int = Query(default=30, ge=1, le=90),
    db         : AsyncSession = Depends(get_db),
):
    """
    Return historical predictions for a location.
    Used by the frontend historical analytics page (Phase 4).
    """
    since = datetime.utcnow() - timedelta(days=days)
    result = await db.execute(
        select(PredictionLog)
        .where(PredictionLog.location_id == location_id)
        .where(PredictionLog.timestamp >= since)
        .order_by(PredictionLog.prediction_date.desc())
    )
    logs = result.scalars().all()
    if not logs:
        raise HTTPException(
            status_code=404,
            detail=f"No prediction history for location {location_id}.",
        )
    return logs