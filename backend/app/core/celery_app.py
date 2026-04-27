from celery import Celery
from celery.schedules import crontab
from app.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "pest_prediction",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        "app.tasks.ingestion_tasks",
        "app.tasks.prediction_tasks",
        "app.tasks.alert_tasks",
    ],
)

# ── Celery configuration ──────────────────────────────────────────────────────
celery_app.conf.update(
    task_serializer          = "json",
    result_serializer        = "json",
    accept_content           = ["json"],
    timezone                 = "Asia/Jakarta",   # WIB
    enable_utc               = True,
    task_track_started       = True,
    task_acks_late           = True,             # Ack after completion, not receipt
    worker_prefetch_multiplier = 1,              # One task at a time per worker
    result_expires           = 86400,            # Results expire after 24h
)

# ── Beat schedule (cron-style task scheduling) ────────────────────────────────
celery_app.conf.beat_schedule = {
    # Step 1: Fetch yesterday weather data for all 31 locations
    "ingest-weather-daily": {
        "task"    : "app.tasks.ingestion_tasks.ingest_weather_all_locations",
        "schedule": crontab(hour=17, minute=0),  # 00:00 WIB = 17:00 UTC
    },
    # Step 2: Compute 84 engineered features from raw weather
    "compute-features-daily": {
        "task"    : "app.tasks.ingestion_tasks.compute_features_all_locations",
        "schedule": crontab(hour=17, minute=30),  # 00:30 WIB
    },
    # Step 3: Run AA-LSTM-AEA predictions for all locations
    "run-predictions-daily": {
        "task"    : "app.tasks.prediction_tasks.run_daily_predictions",
        "schedule": crontab(hour=18, minute=0),   # 01:00 WIB
    },
    # Step 4: Evaluate alert thresholds and send notifications
    "evaluate-alerts-daily": {
        "task"    : "app.tasks.alert_tasks.evaluate_alerts",
        "schedule": crontab(hour=18, minute=30),  # 01:30 WIB
    },
    # Step 5: Weekly PDF summary report (Sundays)
    "weekly-summary": {
        "task"    : "app.tasks.alert_tasks.send_weekly_summary",
        "schedule": crontab(hour=0, minute=0, day_of_week=0),  # Sun 07:00 WIB
    },
}