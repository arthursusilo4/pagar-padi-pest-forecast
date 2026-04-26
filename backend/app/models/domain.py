from sqlalchemy import (
    Column, Integer, String, Float, DateTime,
    ForeignKey, Index, Boolean, Text, JSON
)
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime
 
Base = declarative_base()
 
 
# ==============================================================================
# STATIC METADATA TABLES (standard PostgreSQL)
# ==============================================================================
 
class Location(Base):
    """
    One row per Jember subdistrict (31 total).
    Coordinates used for Open-Meteo API weather ingestion.
    """
    __tablename__ = "locations"
 
    id            = Column(Integer, primary_key=True, index=True)
    name          = Column(String(100), nullable=False, unique=True)
    district_code = Column(String(20), nullable=True)   # BPS kode wilayah
    latitude      = Column(Float, nullable=False)
    longitude     = Column(Float, nullable=False)
    is_active     = Column(Boolean, default=True)
    created_at    = Column(DateTime(timezone=True), default=datetime.utcnow)
 
    # Relationships
    weather_observations = relationship(
        "WeatherObservation", back_populates="location")
    predictions          = relationship(
        "PredictionLog", back_populates="location")
    alert_configs        = relationship(
        "AlertConfig", back_populates="location")
    crop_calendars       = relationship(
        "CropCalendar", back_populates="location")
 
    def __repr__(self):
        return f"<Location id={self.id} name={self.name}>"
 
 
class CropCalendar(Base):
    """
    Tracks planting dates per location per season.
    Used to compute growth_stage and days_since_planting features
    which are required inputs for the AA-LSTM-AEA model.
    """
    __tablename__ = "crop_calendars"
 
    id              = Column(Integer, primary_key=True, index=True)
    location_id     = Column(
        Integer, ForeignKey("locations.id"), nullable=False, index=True)
    planting_date   = Column(DateTime(timezone=True), nullable=False)
    season_label    = Column(String(50), nullable=True)  # e.g. "MH 2025/2026"
    variety         = Column(String(100), nullable=True)
    is_current      = Column(Boolean, default=True)
    created_at      = Column(DateTime(timezone=True), default=datetime.utcnow)
 
    location = relationship("Location", back_populates="crop_calendars")
 
 
class AlertConfig(Base):
    """
    Per-location alert thresholds and notification preferences.
    Each pest gets its own threshold (0–100 scale).
    """
    __tablename__ = "alert_configs"
 
    id                    = Column(Integer, primary_key=True, index=True)
    location_id           = Column(
        Integer, ForeignKey("locations.id"), nullable=False, index=True)
 
    # Per-pest thresholds (0–100 scale, matching UI display)
    bph_threshold         = Column(Float, default=50.0)
    ysb_threshold         = Column(Float, default=50.0)
    rlf_threshold         = Column(Float, default=50.0)
    wst_threshold         = Column(Float, default=50.0)
    rat_threshold         = Column(Float, default=50.0)
    snail_threshold       = Column(Float, default=50.0)
 
    # Anomaly alert threshold (0–1 scale, anomaly score)
    anomaly_threshold     = Column(Float, default=0.70)
 
    # Notification channels
    notify_browser_push   = Column(Boolean, default=True)
    notify_email          = Column(Boolean, default=False)
    notify_whatsapp       = Column(Boolean, default=False)
    contact_email         = Column(String(255), nullable=True)
    contact_whatsapp      = Column(String(20), nullable=True)
 
    is_active             = Column(Boolean, default=True)
    created_at            = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at            = Column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )
 
    location = relationship("Location", back_populates="alert_configs")
 
 
class OutbreakReport(Base):
    """
    Ground truth reports submitted by farmers and extension workers.
    Used for model calibration and feedback loop (Phase 4).
    Matched against PredictionLog entries for accuracy scoring.
    """
    __tablename__ = "outbreak_reports"
 
    id                = Column(Integer, primary_key=True, index=True)
    location_id       = Column(
        Integer, ForeignKey("locations.id"), nullable=False, index=True)
    report_date       = Column(DateTime(timezone=True), nullable=False)
    pest_type         = Column(String(20), nullable=False)  # bph/ysb/rlf/wst/rat/snail
    severity          = Column(String(20), nullable=True)   # low/moderate/high/severe
    area_affected_ha  = Column(Float, nullable=True)
    notes             = Column(Text, nullable=True)
    reporter_name     = Column(String(100), nullable=True)
    confirmed         = Column(Boolean, default=False)
    created_at        = Column(DateTime(timezone=True), default=datetime.utcnow)
 
 
# ==============================================================================
# TIME-SERIES TABLE (TimescaleDB hypertable)
# ==============================================================================
 
class WeatherObservation(Base):
    """
    Daily weather observations per location.
    Stores all 84 feature values required by AA-LSTM-AEA.
 
    This table is converted to a TimescaleDB hypertable in the
    Alembic migration. The composite primary key (time, location_id)
    is standard for TimescaleDB hypertables.
 
    Raw weather columns: direct readings from Open-Meteo API.
    Engineered columns:  computed by the feature engineering pipeline
                         and stored here to avoid recomputation.
    """
    __tablename__ = "weather_observations"
 
    # ── Hypertable primary key ────────────────────────────────
    time        = Column(
        DateTime(timezone=True), primary_key=True, nullable=False)
    location_id = Column(
        Integer, ForeignKey("locations.id"),
        primary_key=True, nullable=False)
 
    # ── Raw meteorological features (Open-Meteo API fields) ───
    tempmax             = Column(Float)
    tempmin             = Column(Float)
    temp                = Column(Float)
    feelslikemax        = Column(Float)
    feelslikemin        = Column(Float)
    feelslike           = Column(Float)
    dew                 = Column(Float)
    humidity            = Column(Float)
    precip              = Column(Float)
    precipprob          = Column(Float)
    precipcover         = Column(Float)
    snow                = Column(Float)
    snowdepth           = Column(Float)
    windgust            = Column(Float)
    windspeed           = Column(Float)
    winddir             = Column(Float)
    sealevelpressure    = Column(Float)
    cloudcover          = Column(Float)
    visibility          = Column(Float)
    solarradiation      = Column(Float)
    solarenergy         = Column(Float)
    uvindex             = Column(Float)
    severerisk          = Column(Float)
    moonphase           = Column(Float)
 
    # ── Engineered rolling features ───────────────────────────
    temp_rolling_avg_7d         = Column(Float)
    humidity_rolling_avg_7d     = Column(Float)
    precip_rolling_avg_7d       = Column(Float)
    windspeed_rolling_avg_7d    = Column(Float)
 
    # ── Crop phenology features (computed from CropCalendar) ──
    growth_stage_encoded        = Column(Float)  # label-encoded int as float
    days_since_planting         = Column(Float)
    vpd                         = Column(Float)
    vpd_index                   = Column(Float)
    lwd                         = Column(Float)
    humidity_norm               = Column(Float)
    rhsi                        = Column(Float)
    rhsi_rolling_7d             = Column(Float)
    rainfall_event              = Column(Float)
    rainfall_events_7d          = Column(Float)
    cpi_7d                      = Column(Float)
    cpi_14d                     = Column(Float)
 
    # ── GDD pest development features ────────────────────────
    bph_gdd_daily               = Column(Float)
    bph_gdd_cumulative          = Column(Float)
    bph_gen_progress            = Column(Float)
    ysb_gdd_daily               = Column(Float)
    ysb_gdd_cumulative          = Column(Float)
    ysb_gen_progress            = Column(Float)
    rlf_gdd_daily               = Column(Float)
    rlf_gdd_cumulative          = Column(Float)
    rlf_gen_progress            = Column(Float)
    wst_gdd_daily               = Column(Float)
    wst_gdd_cumulative          = Column(Float)
    wst_gen_progress            = Column(Float)
 
    # ── Risk rolling features ──────────────────────────────────
    bph_risk_rolling_7d         = Column(Float)
    ysb_risk_rolling_7d         = Column(Float)
    rlf_risk_rolling_7d         = Column(Float)
    wst_risk_rolling_7d         = Column(Float)
    rat_risk_rolling_7d         = Column(Float)
    snail_risk_rolling_7d       = Column(Float)
 
    # ── Agronomic context features ────────────────────────────
    stage_suscept               = Column(Float)
    field_age_factor            = Column(Float)
    plant_density               = Column(Float)
    soil_moisture_pct           = Column(Float)
    soil_k_mg_kg                = Column(Float)
 
    # ── Pressure index features ───────────────────────────────
    pest_pressure_index         = Column(Float)
    disease_pressure_index      = Column(Float)
    total_pressure_index        = Column(Float)
 
    # ── Cyclic encoded features (sin/cos) ─────────────────────
    day_of_year_sin             = Column(Float)
    day_of_year_cos             = Column(Float)
    month_sin                   = Column(Float)
    month_cos                   = Column(Float)
    winddir_sin                 = Column(Float)
    winddir_cos                 = Column(Float)
    moonphase_sin               = Column(Float)
    moonphase_cos               = Column(Float)
 
    # ── Precipitation type flags ──────────────────────────────
    precip_is_rain              = Column(Float)
    precip_is_snow              = Column(Float)
    precip_is_none              = Column(Float)
 
    # ── Calendar features ──────────────────────────────────────
    day_of_week                 = Column(Float)
    week_of_year                = Column(Float)
 
    # ── Metadata ───────────────────────────────────────────────
    ingested_at = Column(DateTime(timezone=True), default=datetime.utcnow)
 
    # ── Relationships & indexes ────────────────────────────────
    location = relationship("Location", back_populates="weather_observations")
 
    __table_args__ = (
        Index(
            "ix_weather_location_time",
            "location_id", "time",
            postgresql_using="btree",
        ),
        # Partial index for recent data queries (last 30 days)
        # Most inference queries only touch recent data
        Index(
            "ix_weather_recent",
            "time",
            postgresql_using="btree",
        ),
    )
 
 
# ==============================================================================
# PREDICTION LOGGING TABLE (standard PostgreSQL)
# ==============================================================================
 
class PredictionLog(Base):
    """
    Full prediction record including all 6 pest risk outputs,
    attention weights, anomaly scores, and learned lambda.
 
    This is the complete inference output of AA-LSTM-AEA stored
    for historical analytics, model drift monitoring, and
    outbreak report correlation (Phase 4 feedback loop).
 
    JSON columns store variable-length arrays to avoid
    creating 14 separate Float columns per attention/anomaly day.
    """
    __tablename__ = "predictions_log"
 
    id                      = Column(Integer, primary_key=True, index=True)
    timestamp               = Column(
        DateTime(timezone=True), default=datetime.utcnow, index=True)
    location_id             = Column(
        Integer, ForeignKey("locations.id"), nullable=False, index=True)
    prediction_date         = Column(DateTime(timezone=True), nullable=False)
 
    # ── Six pest risk outputs (0–100 scale for storage) ───────
    bph_risk                = Column(Float, nullable=False)
    ysb_risk                = Column(Float, nullable=False)
    rlf_risk                = Column(Float, nullable=False)
    wst_risk                = Column(Float, nullable=False)
    rat_risk                = Column(Float, nullable=False)
    snail_risk              = Column(Float, nullable=False)
 
    # ── Derived risk tier (LOW/MODERATE/HIGH/CRITICAL) ────────
    bph_tier                = Column(String(10))
    ysb_tier                = Column(String(10))
    rlf_tier                = Column(String(10))
    wst_tier                = Column(String(10))
    rat_tier                = Column(String(10))
    snail_tier              = Column(String(10))
 
    # ── Dominant pest for UI headline ─────────────────────────
    dominant_pest           = Column(String(20))
    dominant_risk           = Column(Float)
 
    # ── Explainability outputs (JSON arrays of 14 values) ─────
    # attention_weights: list[float] — 14 values summing to ~1.0
    # anomaly_scores:    list[float] — 14 values in [0, 1]
    attention_weights       = Column(JSON, nullable=True)
    anomaly_scores          = Column(JSON, nullable=True)
 
    # ── Learned model parameter logged per inference ──────────
    lambda_anomaly          = Column(Float, nullable=True)
 
    # ── Anomaly contribution delta ─────────────────────────────
    # Difference in predictions between full model and lambda=0
    # Used for the "anomaly contribution" UI indicator (Phase 2)
    anomaly_contribution    = Column(JSON, nullable=True)  # dict of 6 deltas
 
    # ── Inference metadata ─────────────────────────────────────
    inference_latency_ms    = Column(Float, nullable=True)
    model_version           = Column(String(50), default="aa_lstm_aea_v1")
 
    location = relationship("Location", back_populates="predictions")
 
    __table_args__ = (
        Index("ix_predictions_location_date",
              "location_id", "prediction_date"),
    )