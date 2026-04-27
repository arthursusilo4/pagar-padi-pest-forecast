import asyncio
import logging
from datetime import date, timedelta
 
from app.core.celery_app import celery_app
from app.core.database import AsyncSessionLocal
from app.services.weather_ingestion import WeatherIngestionService
 
logger = logging.getLogger(__name__)
 
 
def run_async(coro):
    """Helper to run async code inside synchronous Celery tasks."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
 
 
@celery_app.task(
    name="app.tasks.ingestion_tasks.ingest_weather_all_locations",
    bind=True,
    max_retries=3,
    default_retry_delay=300,   # Retry after 5 minutes on failure
)
def ingest_weather_all_locations(self, target_date_str: str = None):
    """
    Celery task: fetch yesterday weather data for all 31 locations.
 
    Triggered daily at 00:00 WIB by Celery Beat.
    On failure, retries up to 3 times with 5-minute delays.
 
    Parameters
    ----------
    target_date_str : str  ISO date string "YYYY-MM-DD" (optional).
                           Defaults to yesterday if not provided.
                           Used for manual backfill: .delay("2025-01-15")
    """
    try:
        target_date = (
            date.fromisoformat(target_date_str)
            if target_date_str
            else date.today() - timedelta(days=1)
        )
 
        logger.info(f"Starting weather ingestion for {target_date}")
 
        async def _run():
            async with AsyncSessionLocal() as db:
                service = WeatherIngestionService()
                return await service.ingest_yesterday(
                    db=db, target_date=target_date)
 
        result = run_async(_run())
 
        logger.info(
            f"Weather ingestion complete: "
            f"{result[\'success_count\']}/{result[\'total\']} locations"
        )
        return result
 
    except Exception as exc:
        logger.error(f"Weather ingestion failed: {exc}")
        raise self.retry(exc=exc)
 
 
@celery_app.task(
    name="app.tasks.ingestion_tasks.compute_features_all_locations",
    bind=True,
    max_retries=3,
    default_retry_delay=300,
)
def compute_features_all_locations(self, target_date_str: str = None):
    """
    Celery task: compute 7-day rolling averages and update
    WeatherObservation rows with engineered feature values.
 
    Runs after ingest_weather_all_locations completes.
    """
    try:
        target_date = (
            date.fromisoformat(target_date_str)
            if target_date_str
            else date.today() - timedelta(days=1)
        )
 
        logger.info(f"Computing features for {target_date}")
 
        async def _run():
            from app.models.domain import Location, WeatherObservation
            from sqlalchemy import select, update
            from datetime import datetime
            import numpy as np
 
            async with AsyncSessionLocal() as db:
                # Get all active locations
                loc_result = await db.execute(
                    select(Location).where(Location.is_active == True)
                )
                locations = loc_result.scalars().all()
 
                updated_count = 0
 
                for location in locations:
                    # Fetch last 7 days of observations for rolling avg
                    from datetime import timedelta
                    window_start = datetime.combine(
                        target_date - timedelta(days=7),
                        datetime.min.time()
                    )
                    window_end = datetime.combine(
                        target_date + timedelta(days=1),
                        datetime.min.time()
                    )
 
                    obs_result = await db.execute(
                        select(WeatherObservation)
                        .where(
                            WeatherObservation.location_id == location.id)
                        .where(WeatherObservation.time >= window_start)
                        .where(WeatherObservation.time < window_end)
                        .order_by(WeatherObservation.time.asc())
                    )
                    obs_rows = obs_result.scalars().all()
 
                    if not obs_rows:
                        continue
 
                    # Compute 7-day rolling averages for the target date row
                    temps      = [o.temp      or 0.0 for o in obs_rows]
                    humidities = [o.humidity  or 0.0 for o in obs_rows]
                    precips    = [o.precip    or 0.0 for o in obs_rows]
                    windspeeds = [o.windspeed or 0.0 for o in obs_rows]
 
                    # Update only the target date row
                    target_dt = datetime.combine(
                        target_date, datetime.min.time())
                    await db.execute(
                        update(WeatherObservation)
                        .where(
                            WeatherObservation.location_id == location.id)
                        .where(WeatherObservation.time == target_dt)
                        .values(
                            temp_rolling_avg_7d     = float(np.mean(temps[-7:])),
                            humidity_rolling_avg_7d = float(np.mean(humidities[-7:])),
                            precip_rolling_avg_7d   = float(np.mean(precips[-7:])),
                            windspeed_rolling_avg_7d= float(np.mean(windspeeds[-7:])),
                            rainfall_events_7d      = float(sum(1 for p in precips[-7:] if p > 5.0)),
                        )
                    )
                    updated_count += 1
 
                await db.commit()
                return {"updated_locations": updated_count}
 
        result = run_async(_run())
        logger.info(
            f"Feature computation complete: "
            f"{result[\'updated_locations\']} locations updated"
        )
        return result
 
    except Exception as exc:
        logger.error(f"Feature computation failed: {exc}")
        raise self.retry(exc=exc)
 
 
@celery_app.task(
    name="app.tasks.ingestion_tasks.backfill_weather",
)
def backfill_weather(start_date_str: str, end_date_str: str):
    """
    Manual backfill task for loading historical weather data.
 
    Usage from Python (run once to seed initial data):
        from app.tasks.ingestion_tasks import backfill_weather
        backfill_weather.delay("2019-01-01", "2024-12-31")
 
    Or from command line:
        celery -A app.core.celery_app call
            app.tasks.ingestion_tasks.backfill_weather
            --args=\'["2023-01-01", "2024-12-31"]\'
    """
    from datetime import datetime, timedelta
 
    start = date.fromisoformat(start_date_str)
    end   = date.fromisoformat(end_date_str)
 
    current = start
    results = []
 
    while current <= end:
        logger.info(f"Backfilling {current}...")
 
        async def _run(d):
            async with AsyncSessionLocal() as db:
                service = WeatherIngestionService()
                return await service.ingest_yesterday(db=db, target_date=d)
 
        result = run_async(_run(current))
        results.append(result)
        current += timedelta(days=1)
 
    total_success = sum(r["success_count"] for r in results)
    total_days    = len(results)
    logger.info(
        f"Backfill complete: {total_days} days, "
        f"{total_success} location-days ingested"
    )
    return {"days": total_days, "total_success": total_success}