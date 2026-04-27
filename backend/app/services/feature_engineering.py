import math
import logging
import numpy as np
import pandas as pd
from datetime import datetime, date
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from app.models.domain import WeatherObservation, CropCalendar, PredictionLog
from app.core.config import get_settings

logger   = logging.getLogger(__name__)
settings = get_settings()

# ── Growth stage thresholds (days since planting → stage label) ───────────────
GROWTH_STAGE_MAP = {
    (0,   7):  "germination",
    (8,  14):  "seedling",
    (15, 30):  "tillering",
    (31, 55):  "active_tillering",
    (56, 70):  "panicle_initiation",
    (71, 85):  "booting",
    (86, 95):  "heading",
    (96, 110): "flowering",
    (111, 130): "grain_filling",
    (131, 999): "maturity",
}

# Growth stage label → encoded integer (must match training encoding)
GROWTH_STAGE_ENCODING = {
    "germination"      : 0,
    "seedling"         : 1,
    "tillering"        : 2,
    "active_tillering" : 3,
    "panicle_initiation": 4,
    "booting"          : 5,
    "heading"          : 6,
    "flowering"        : 7,
    "grain_filling"    : 8,
    "maturity"         : 9,
    "unknown"          : 5,  # fallback to booting (mid-season default)
}

# Stage susceptibility index (biological vulnerability per growth stage)
STAGE_SUSCEPTIBILITY = {
    "germination"      : 0.3,
    "seedling"         : 0.6,
    "tillering"        : 0.8,
    "active_tillering" : 0.9,
    "panicle_initiation": 0.85,
    "booting"          : 0.7,
    "heading"          : 0.75,
    "flowering"        : 0.65,
    "grain_filling"    : 0.5,
    "maturity"         : 0.2,
    "unknown"          : 0.5,
}

# GDD base temperatures per pest (degrees Celsius)
GDD_BASE_TEMPS = {
    "bph": 15.0,   # Brown Plant Hopper
    "ysb": 14.0,   # Yellow Stem Borer
    "rlf": 13.5,   # Rice Leaf Folder
    "wst": 14.5,   # White Stem Borer
}


def get_growth_stage(days_since_planting: int) -> str:
    """Map days since planting to growth stage label."""
    for (start, end), stage in GROWTH_STAGE_MAP.items():
        if start <= days_since_planting <= end:
            return stage
    return "unknown"


def compute_vpd(temp_c: float, humidity_pct: float) -> float:
    """
    Compute Vapour Pressure Deficit (kPa).
    VPD = SVP × (1 - RH/100)
    SVP = 0.6108 × exp(17.27 × T / (T + 237.3))
    """
    svp = 0.6108 * math.exp(17.27 * temp_c / (temp_c + 237.3))
    return round(svp * (1.0 - humidity_pct / 100.0), 4)


def compute_gdd_daily(temp_max: float, temp_min: float,
                      base_temp: float) -> float:
    """Daily Growing Degree Days above base temperature."""
    avg_temp = (temp_max + temp_min) / 2.0
    return max(0.0, avg_temp - base_temp)


def cyclic_encode(value: float, period: float) -> tuple[float, float]:
    """Convert a cyclic value to sin/cos pair."""
    angle = 2 * math.pi * value / period
    return math.sin(angle), math.cos(angle)


class FeatureEngineer:
    """
    Transforms raw WeatherObservation rows into the 84-feature matrix
    expected by AA-LSTM-AEA.

    Usage:
        engineer = FeatureEngineer()
        feature_matrix = await engineer.build_feature_window(
            location_id=1,
            target_date=date.today(),
            db=db_session,
        )
        # Returns np.ndarray shape (14, 84) or None if insufficient data
    """

    async def build_feature_window(
        self,
        location_id  : int,
        target_date  : date,
        db           : AsyncSession,
    ) -> Optional[np.ndarray]:
        """
        Assemble a complete 14-day × 84-feature matrix for inference.

        Queries:
          - WeatherObservation rows (last 14 days of raw weather)
          - CropCalendar (current planting date for growth stage)
          - PredictionLog (last 7 days of risk values for rolling features)

        Returns None if fewer than 14 days of weather data are available.
        """
        # ── Query 14 days of raw weather ──────────────────────
        from datetime import timedelta
        start_dt = datetime.combine(
            target_date - timedelta(days=settings.SEQUENCE_LENGTH),
            datetime.min.time()
        )
        end_dt = datetime.combine(target_date, datetime.min.time())

        weather_result = await db.execute(
            select(WeatherObservation)
            .where(WeatherObservation.location_id == location_id)
            .where(WeatherObservation.time >= start_dt)
            .where(WeatherObservation.time < end_dt)
            .order_by(WeatherObservation.time.asc())
        )
        weather_rows = weather_result.scalars().all()

        if len(weather_rows) < settings.SEQUENCE_LENGTH:
            logger.warning(
                f"Insufficient weather data for location {location_id} "
                f"on {target_date}. Have {len(weather_rows)}, "
                f"need {settings.SEQUENCE_LENGTH}."
            )
            return None

        weather_rows = weather_rows[-settings.SEQUENCE_LENGTH:]

        # ── Query current crop calendar ────────────────────────
        cal_result = await db.execute(
            select(CropCalendar)
            .where(CropCalendar.location_id == location_id)
            .where(CropCalendar.is_current == True)
        )
        crop_cal = cal_result.scalar_one_or_none()

        # ── Query last 7 days of predictions (risk rolling avg) ─
        risk_start = datetime.combine(
            target_date - timedelta(days=7),
            datetime.min.time()
        )
        pred_result = await db.execute(
            select(PredictionLog)
            .where(PredictionLog.location_id == location_id)
            .where(PredictionLog.prediction_date >= risk_start)
            .where(PredictionLog.prediction_date < end_dt)
            .order_by(PredictionLog.prediction_date.asc())
        )
        recent_preds = pred_result.scalars().all()

        # ── Build feature matrix row by row ───────────────────
        rows = []
        cumulative_gdd = {pest: 0.0 for pest in ["bph", "ysb", "rlf", "wst"]}

        for t, obs in enumerate(weather_rows):
            row = self._compute_row_features(
                obs=obs,
                obs_date=obs.time.date() if obs.time else target_date,
                crop_cal=crop_cal,
                cumulative_gdd=cumulative_gdd,
                recent_preds=recent_preds,
                t_idx=t,
            )
            rows.append(row)

        feature_matrix = np.array(rows, dtype=np.float32)
        return feature_matrix

    def _compute_row_features(
        self,
        obs            : WeatherObservation,
        obs_date       : date,
        crop_cal       : Optional[CropCalendar],
        cumulative_gdd : dict,
        recent_preds   : list,
        t_idx          : int,
    ) -> list[float]:
        """
        Compute all 84 features for a single observation day.
        Feature order MUST match feature_meta["feature_cols"].
        """
        # Safe getters with fallback
        def safe(val, default=0.0):
            return float(val) if val is not None else default

        temp       = safe(obs.temp, 28.0)
        tempmax    = safe(obs.tempmax, 30.0)
        tempmin    = safe(obs.tempmin, 25.0)
        humidity   = safe(obs.humidity, 80.0)
        precip     = safe(obs.precip, 0.0)
        windspeed  = safe(obs.windspeed, 5.0)
        winddir    = safe(obs.winddir, 180.0)
        moonphase  = safe(obs.moonphase, 0.5)
        solarrad   = safe(obs.solarradiation, 15.0)
        dew        = safe(obs.dew, 22.0)

        # ── Crop phenology ─────────────────────────────────────
        if crop_cal and crop_cal.planting_date:
            days_since = (obs_date - crop_cal.planting_date.date()).days
            days_since = max(0, days_since)
        else:
            days_since = 60   # Default: mid-season estimate

        growth_stage_label   = get_growth_stage(days_since)
        growth_stage_encoded = GROWTH_STAGE_ENCODING.get(growth_stage_label, 5)
        stage_suscept        = STAGE_SUSCEPTIBILITY.get(growth_stage_label, 0.5)
        field_age_factor     = min(1.0, days_since / 120.0)

        # ── VPD & RHSI ─────────────────────────────────────────
        vpd       = compute_vpd(temp, humidity)
        vpd_index = min(1.0, vpd / 2.0)   # Normalized to [0,1]
        rhsi      = max(0.0, (humidity - 60.0) / 40.0)  # Risk above 60%

        # ── GDD per pest ───────────────────────────────────────
        gdd_values = {}
        for pest, base in GDD_BASE_TEMPS.items():
            daily_gdd              = compute_gdd_daily(tempmax, tempmin, base)
            cumulative_gdd[pest]  += daily_gdd
            gen_progress           = min(1.0, cumulative_gdd[pest] / 300.0)
            gdd_values[pest] = {
                "daily"      : daily_gdd,
                "cumulative" : cumulative_gdd[pest],
                "gen_progress": gen_progress,
            }

        # ── Precipitation indicators ───────────────────────────
        rainfall_event    = 1.0 if precip > 5.0 else 0.0
        precip_is_rain    = 1.0 if precip > 0.0 else 0.0
        precip_is_snow    = 0.0   # No snow in Jember
        precip_is_none    = 1.0 if precip == 0.0 else 0.0

        # ── Cyclic encodings ───────────────────────────────────
        doy_sin, doy_cos    = cyclic_encode(obs_date.timetuple().tm_yday, 365)
        month_sin, month_cos= cyclic_encode(obs_date.month, 12)
        wdir_sin, wdir_cos  = cyclic_encode(winddir, 360)
        moon_sin, moon_cos  = cyclic_encode(moonphase, 1.0)

        # ── Calendar features ──────────────────────────────────
        day_of_week   = float(obs_date.weekday())
        week_of_year  = float(obs_date.isocalendar()[1])

        # ── Pressure index (simplified estimate) ──────────────
        # Full computation requires disease features (excluded from scope)
        # Estimate from weather conditions
        humidity_norm    = humidity / 100.0
        pest_pressure    = (humidity_norm * 0.4 + vpd_index * 0.3
                            + rainfall_event * 0.3)
        disease_pressure = humidity_norm * 0.5 + (precip > 10) * 0.3
        total_pressure   = (pest_pressure + disease_pressure) / 2.0

        # ── Risk rolling averages (from recent predictions) ────
        # Use 0 as default if no predictions yet (system bootstrapping)
        def rolling_avg(pest_key):
            vals = [getattr(p, pest_key, 0.0) or 0.0
                    for p in recent_preds]
            return float(np.mean(vals)) if vals else 0.0

        # ── Assemble feature vector ────────────────────────────
        # ORDER MUST EXACTLY MATCH feature_meta["feature_cols"]
        # Cross-check against your saved feature_meta.json
        features = [
            # Raw weather
            safe(obs.tempmax), safe(obs.tempmin), temp,
            safe(obs.feelslikemax), safe(obs.feelslikemin), safe(obs.feelslike),
            dew, humidity,
            precip, safe(obs.precipprob), safe(obs.precipcover),
            safe(obs.snow), safe(obs.snowdepth),
            safe(obs.windgust), windspeed, winddir,
            safe(obs.sealevelpressure), safe(obs.cloudcover), safe(obs.visibility),
            solarrad, safe(obs.solarenergy), safe(obs.uvindex), safe(obs.severerisk),
            moonphase,

            # Rolling averages (placeholders — updated by ingestion pipeline)
            safe(obs.temp_rolling_avg_7d, temp),
            safe(obs.humidity_rolling_avg_7d, humidity),
            safe(obs.precip_rolling_avg_7d, precip),
            safe(obs.windspeed_rolling_avg_7d, windspeed),

            # Crop phenology
            float(growth_stage_encoded),
            float(days_since),
            vpd, vpd_index,
            safe(obs.lwd, 0.0),
            safe(obs.n_applied, 0.0),
            safe(obs.nsf, 0.0),
            humidity_norm, rhsi,
            safe(obs.rhsi_rolling_7d, rhsi),
            rainfall_event,
            safe(obs.rainfall_events_7d, rainfall_event),
            safe(obs.cpi_7d, 0.0),
            safe(obs.cpi_14d, 0.0),

            # GDD features
            gdd_values["bph"]["daily"], gdd_values["bph"]["cumulative"],
            gdd_values["bph"]["gen_progress"],
            gdd_values["ysb"]["daily"], gdd_values["ysb"]["cumulative"],
            gdd_values["ysb"]["gen_progress"],
            gdd_values["rlf"]["daily"], gdd_values["rlf"]["cumulative"],
            gdd_values["rlf"]["gen_progress"],
            gdd_values["wst"]["daily"], gdd_values["wst"]["cumulative"],
            gdd_values["wst"]["gen_progress"],

            # Risk rolling averages
            rolling_avg("bph_risk"),
            rolling_avg("ysb_risk"),
            rolling_avg("rlf_risk"),
            rolling_avg("wst_risk"),
            rolling_avg("rat_risk"),
            rolling_avg("snail_risk"),

            # Agronomic context
            stage_suscept, field_age_factor,
            safe(obs.plant_density, 25.0),
            safe(obs.soil_moisture_pct, 60.0),
            safe(obs.soil_k_mg_kg, 120.0),

            # Pressure indices
            pest_pressure, disease_pressure, total_pressure,

            # Cyclic encodings
            doy_sin, doy_cos,
            month_sin, month_cos,
            wdir_sin, wdir_cos,
            moon_sin, moon_cos,

            # Precipitation flags
            precip_is_rain, precip_is_snow, precip_is_none,

            # Calendar
            day_of_week, week_of_year,
        ]

        return features