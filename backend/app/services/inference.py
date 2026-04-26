import json
import time
import logging
import numpy as np
import joblib
from pathlib import Path
from typing import Optional
 
import redis.asyncio as aioredis
import tensorflow as tf
from tensorflow import keras
 
from app.core.config import get_settings
 
logger = logging.getLogger(__name__)
settings = get_settings()
 
# ── Target pest names (must match training order) ─────────────────────────────
TARGET_COLS = ["bph_risk", "ysb_risk", "rlf_risk",
               "wst_risk", "rat_risk", "snail_risk"]
 
# ── Risk tier thresholds ──────────────────────────────────────────────────────
def get_risk_tier(value: float) -> str:
    """Convert 0–100 risk score to tier label."""
    if value < settings.RISK_TIER_LOW:
        return "LOW"
    elif value < settings.RISK_TIER_MODERATE:
        return "MODERATE"
    elif value < settings.RISK_TIER_HIGH:
        return "HIGH"
    return "CRITICAL"
 
 
# ==============================================================================
# ANOMALY SCORE COMPUTATION
# ==============================================================================
 
def compute_anomaly_scores(
    X: np.ndarray,
    window: int = 7,
) -> np.ndarray:
    """
    Compute per-timestep anomaly scores for a batch of sequences.
 
    Replicates the exact anomaly computation used during AA-LSTM-AEA
    training (preprocessing_pipeline.py Stage 6 equivalent).
 
    Method: Z-score deviation from rolling mean/std within sequence window.
    Scores are normalized to [0, 1] per sequence.
 
    Parameters
    ----------
    X      : np.ndarray  shape (n_samples, seq_len, n_features)
    window : int         rolling window in days (default: 7)
 
    Returns
    -------
    anomaly_scores : np.ndarray  shape (n_samples, seq_len, 1)  float32
    """
    n_samples, seq_len, n_features = X.shape
    anomaly_scores = np.zeros((n_samples, seq_len, 1), dtype=np.float32)
    eps = 1e-8
 
    for i in range(n_samples):
        scores = np.zeros(seq_len, dtype=np.float32)
        for t in range(seq_len):
            start    = max(0, t - window)
            window_X = X[i, start:t + 1, :]
            if window_X.shape[0] < 2:
                scores[t] = 0.0
                continue
            roll_mean = window_X.mean(axis=0)
            roll_std  = window_X.std(axis=0)
            z = np.abs(X[i, t, :] - roll_mean) / (roll_std + eps)
            scores[t] = float(z.mean())
 
        s_min, s_max = scores.min(), scores.max()
        if s_max > s_min:
            scores = (scores - s_min) / (s_max - s_min + eps)
        anomaly_scores[i, :, 0] = scores
 
    return anomaly_scores
 
 
# ==============================================================================
# SINGLETON INFERENCE SERVICE
# ==============================================================================
 
class PestPredictionService:
    """
    Singleton service that holds the AA-LSTM-AEA model in RAM.
 
    Loaded once at FastAPI startup via the lifespan context manager.
    All subsequent inference calls reuse the same model instance,
    guaranteeing sub-3-second response times.
 
    Usage:
        # In main.py lifespan:
        PestPredictionService.initialize()
 
        # In endpoint:
        service = PestPredictionService.get_instance()
        result  = await service.predict(feature_window)
    """
 
    _instance: Optional["PestPredictionService"] = None
 
    def __init__(self):
        self.model: Optional[keras.Model] = None
        self.scaler_X = None
        self.scaler_y = None
        self.feature_meta: dict = {}
        self._loaded: bool = False
 
    @classmethod
    def initialize(cls) -> "PestPredictionService":
        """Load model and scalers. Call once at application startup."""
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._load()
        return cls._instance
 
    @classmethod
    def get_instance(cls) -> "PestPredictionService":
        """Retrieve the loaded singleton. Raises if not initialized."""
        if cls._instance is None or not cls._instance._loaded:
            raise RuntimeError(
                "PestPredictionService not initialized. "
                "Call PestPredictionService.initialize() in lifespan."
            )
        return cls._instance
 
    def _load(self) -> None:
        """Internal: load model, scalers, and feature metadata."""
        logger.info("Loading AA-LSTM-AEA model...")
        t0 = time.perf_counter()
 
        # Validate file paths
        for path_str, label in [
            (settings.MODEL_PATH,       "Keras model"),
            (settings.SCALER_X_PATH,    "Feature scaler"),
            (settings.SCALER_Y_PATH,    "Target scaler"),
            (settings.FEATURE_META_PATH,"Feature metadata"),
        ]:
            if not Path(path_str).exists():
                raise FileNotFoundError(
                    f"{label} not found at: {path_str}\n"
                    "Ensure /app/weights/ is mounted correctly in docker-compose.yml"
                )
 
        # Load Keras model
        self.model = keras.models.load_model(
            settings.MODEL_PATH,
            compile=False,  # No need to recompile for inference
        )
 
        # Load scalers (saved during preprocessing pipeline)
        self.scaler_X = joblib.load(settings.SCALER_X_PATH)
        self.scaler_y = joblib.load(settings.SCALER_Y_PATH)
 
        # Load feature metadata (feature column names, n_features, etc.)
        with open(settings.FEATURE_META_PATH) as f:
            self.feature_meta = json.load(f)
 
        # Extract learned lambda from attention layer
        try:
            attn_layer    = self.model.get_layer("anomaly_enhanced_attention")
            self._lambda  = float(attn_layer.lambda_anomaly.numpy()[0])
            logger.info(f"Learned lambda (anomaly weight): {self._lambda:.4f}")
        except Exception:
            self._lambda = 1.0
            logger.warning("Could not extract lambda — using default 1.0")
 
        elapsed = time.perf_counter() - t0
        logger.info(
            f"Model loaded in {elapsed:.2f}s | "
            f"Parameters: {self.model.count_params():,} | "
            f"Lambda: {self._lambda:.4f}"
        )
        self._loaded = True
 
    @property
    def lambda_value(self) -> float:
        return self._lambda
 
    # ── Core inference ────────────────────────────────────────────────────────
 
    def _run_inference(
        self,
        X_scaled: np.ndarray,
        anomaly_scores: np.ndarray,
    ) -> np.ndarray:
        """
        Run a single forward pass through the model.
        Inputs are already scaled and shaped.
 
        Parameters
        ----------
        X_scaled       : (1, seq_len, n_features)  float32
        anomaly_scores : (1, seq_len, 1)            float32
 
        Returns
        -------
        y_scaled : (1, n_targets)  float32
        """
        return self.model.predict(
            [X_scaled, anomaly_scores],
            batch_size=1,
            verbose=0,
        )
 
    def predict(self, raw_feature_window: np.ndarray) -> dict:
        """
        Full inference pipeline: scale → anomaly → predict → decode.
 
        Parameters
        ----------
        raw_feature_window : np.ndarray  shape (seq_len, n_features)
            Raw (unscaled) feature values for the 14-day window.
            Column order must match feature_meta["feature_cols"].
 
        Returns
        -------
        dict with keys:
            risks              : dict[str, float]  — 0–100 scale per pest
            risk_tiers         : dict[str, str]    — LOW/MODERATE/HIGH/CRITICAL
            dominant_pest      : str
            dominant_risk      : float
            attention_weights  : list[float]       — 14 values
            anomaly_scores     : list[float]       — 14 values
            anomaly_contribution: dict[str, float] — delta vs no anomaly
            lambda_anomaly     : float
            inference_ms       : float
        """
        t_start = time.perf_counter()
 
        # ── 1. Scale features ─────────────────────────────────
        X = raw_feature_window.astype(np.float32)
        X_scaled = self.scaler_X.transform(
            X.reshape(-1, X.shape[-1])
        ).reshape(1, settings.SEQUENCE_LENGTH, settings.N_FEATURES)
 
        # ── 2. Compute anomaly scores ─────────────────────────
        A = compute_anomaly_scores(
            X_scaled,
            window=settings.ANOMALY_WINDOW,
        )  # shape: (1, seq_len, 1)
 
        # ── 3. Full model inference ───────────────────────────
        y_scaled = self._run_inference(X_scaled, A)
 
        # ── 4. Anomaly contribution: re-run with lambda = 0 ───
        # Temporarily zero the anomaly scores to simulate no enhancement
        A_zero = np.zeros_like(A)
        y_scaled_no_anomaly = self._run_inference(X_scaled, A_zero)
 
        # ── 5. Extract attention weights ──────────────────────
        try:
            attn_extractor = keras.Model(
                inputs=self.model.input,
                outputs=[
                    self.model.output,
                    self.model.get_layer(
                        "anomaly_enhanced_attention").output[1],
                ],
            )
            _, attn_weights_raw = attn_extractor.predict(
                [X_scaled, A], batch_size=1, verbose=0)
            attention_weights = attn_weights_raw[0].tolist()  # 14 values
        except Exception as e:
            logger.warning(f"Attention extraction failed: {e}")
            attention_weights = [1 / settings.SEQUENCE_LENGTH] * settings.SEQUENCE_LENGTH
 
        # ── 6. Inverse transform to original scale ────────────
        y_original          = self.scaler_y.inverse_transform(y_scaled)[0]
        y_original_no_anomaly = self.scaler_y.inverse_transform(
            y_scaled_no_anomaly)[0]
 
        # Clip to [0, 100] (model occasionally predicts slightly outside)
        y_original            = np.clip(y_original * 100, 0.0, 100.0)
        y_original_no_anomaly = np.clip(y_original_no_anomaly * 100, 0.0, 100.0)
 
        # ── 7. Build structured output ────────────────────────
        risks = {col: round(float(v), 2)
                 for col, v in zip(TARGET_COLS, y_original)}
 
        risk_tiers = {col: get_risk_tier(v)
                      for col, v in risks.items()}
 
        # Dominant pest = highest risk
        dominant_pest = max(risks, key=risks.get)
        dominant_risk = risks[dominant_pest]
 
        # Anomaly contribution per pest (positive = anomaly raised risk)
        anomaly_contribution = {
            col: round(float(y_original[i] - y_original_no_anomaly[i]), 2)
            for i, col in enumerate(TARGET_COLS)
        }
 
        # Anomaly scores as list (14 values per timestep)
        anomaly_scores_list = A[0, :, 0].tolist()
 
        t_end = time.perf_counter()
        inference_ms = round((t_end - t_start) * 1000, 1)
 
        return {
            "risks"               : risks,
            "risk_tiers"          : risk_tiers,
            "dominant_pest"       : dominant_pest,
            "dominant_risk"       : round(dominant_risk, 2),
            "attention_weights"   : attention_weights,
            "anomaly_scores"      : anomaly_scores_list,
            "anomaly_contribution": anomaly_contribution,
            "lambda_anomaly"      : round(self._lambda, 4),
            "inference_ms"        : inference_ms,
        }
 
 
# ==============================================================================
# REDIS FEATURE WINDOW CACHE
# ==============================================================================
 
class FeatureWindowCache:
    """
    Redis cache for assembled 14-day feature windows.
 
    Cache key: "feature_window:{location_id}:{date_str}"
    TTL: 24 hours (new data arrives once per day)
 
    On cache hit:  skip DB query entirely → fastest path
    On cache miss: assemble from DB → store in Redis → return
    """
 
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
 
    def _cache_key(self, location_id: int, date_str: str) -> str:
        return f"feature_window:{location_id}:{date_str}"
 
    async def get(
        self,
        location_id: int,
        date_str: str,
    ) -> Optional[np.ndarray]:
        """
        Retrieve cached feature window.
 
        Returns
        -------
        np.ndarray of shape (seq_len, n_features) if cached, else None.
        """
        key  = self._cache_key(location_id, date_str)
        data = await self.redis.get(key)
        if data is None:
            return None
        arr = np.frombuffer(data, dtype=np.float32)
        return arr.reshape(
            settings.SEQUENCE_LENGTH, settings.N_FEATURES)
 
    async def set(
        self,
        location_id: int,
        date_str: str,
        feature_window: np.ndarray,
    ) -> None:
        """
        Cache a feature window with 24-hour TTL.
 
        Parameters
        ----------
        feature_window : np.ndarray  shape (seq_len, n_features)
        """
        key  = self._cache_key(location_id, date_str)
        data = feature_window.astype(np.float32).tobytes()
        await self.redis.setex(
            key,
            settings.FEATURE_CACHE_TTL_SECONDS,
            data,
        )
        logger.debug(
            f"Cached feature window for location={location_id} "
            f"date={date_str} TTL={settings.FEATURE_CACHE_TTL_SECONDS}s"
        )
 
    async def invalidate(self, location_id: int, date_str: str) -> None:
        """Remove cached entry (used when new data is ingested)."""
        key = self._cache_key(location_id, date_str)
        await self.redis.delete(key)