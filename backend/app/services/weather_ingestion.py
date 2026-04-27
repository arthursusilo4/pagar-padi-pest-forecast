import logging
import httpx
from datetime import date, timedelta
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
import math
 
from app.models.domain import WeatherObservation, Location
from app.core.config import get_settings
 
logger   = logging.getLogger(__name__)
settings = get_settings()
 
# Open-Meteo field mapping: API parameter → WeatherObservation column
OPENMETEO_DAILY_FIELDS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "apparent_temperature_max",
    "apparent_temperature_min",
    "apparent_temperature_mean",
    "dew_point_2m_mean",
    "relative_humidity_2m_mean",
    "precipitation_sum",
    "precipitation_probability_mean",
    "wind_speed_10m_max",
    "wind_direction_10m_dominant",
    "wind_gusts_10m_max",
    "surface_pressure_mean",
    "cloud_cover_mean",
    "visibility_mean",
    "shortwave_radiation_sum",
    "uv_index_max",
    "snowfall_sum",
    "snow_depth_mean",
    "weather_code",
]
 
 
class WeatherIngestionService:
    """
    Fetches and stores daily weather observations for all locations.
    Called by the Celery Beat task every day at 00:00 WIB.
    """
 
    async def ingest_yesterday(
        self,
        db: AsyncSession,
        target_date: Optional[date] = None,
    ) -> dict:
        """
        Fetch and store weather data for all active locations.
 
        Parameters
        ----------
        target_date : date  Date to fetch (defaults to yesterday)
 
        Returns
        -------
        dict with success/failure counts
        """
        if target_date is None:
            target_date = date.today() - timedelta(days=1)
 
        # Fetch all active locations
        result    = await db.execute(
            select(Location).where(Location.is_active == True)
        )
        locations = result.scalars().all()
 
        success_count = 0
        failure_count = 0
 
        async with httpx.AsyncClient(timeout=30.0) as client:
            for location in locations:
                try:
                    obs = await self._fetch_single_location(
                        client=client,
                        location=location,
                        target_date=target_date,
                    )
                    if obs:
                        # Upsert: insert or update if already exists
                        stmt = pg_insert(WeatherObservation).values(
                            **{k: v for k, v in obs.items()}
                        ).on_conflict_do_update(
                            index_elements=["time", "location_id"],
                            set_={k: v for k, v in obs.items()
                                  if k not in ["time", "location_id"]}
                        )
                        await db.execute(stmt)
                        success_count += 1
                        logger.debug(
                            f"Ingested weather for location "
                            f"{location.name} on {target_date}"
                        )
                except Exception as e:
                    failure_count += 1
                    logger.error(
                        f"Failed to ingest weather for location "
                        f"{location.name}: {e}"
                    )
 
        await db.commit()
        logger.info(
            f"Weather ingestion complete for {target_date}. "
            f"Success: {success_count}, Failed: {failure_count}"
        )
        return {
            "date"         : str(target_date),
            "success_count": success_count,
            "failure_count": failure_count,
            "total"        : len(locations),
        }
 
    async def _fetch_single_location(
        self,
        client     : httpx.AsyncClient,
        location   : Location,
        target_date: date,
    ) -> Optional[dict]:
        """Fetch one day of weather data for one location."""
        date_str = target_date.isoformat()
 
        # Use historical archive API for past dates
        url    = settings.OPENMETEO_HISTORICAL_URL
        params = {
            "latitude"        : location.latitude,
            "longitude"       : location.longitude,
            "start_date"      : date_str,
            "end_date"        : date_str,
            "daily"           : ",".join(OPENMETEO_DAILY_FIELDS),
            "timezone"        : "Asia/Jakarta",
            "wind_speed_unit" : "ms",     # metres per second
        }
 
        response = await client.get(url, params=params)
        response.raise_for_status()
        data     = response.json()
 
        daily = data.get("daily", {})
        if not daily or not daily.get("time"):
            return None
 
        idx = 0   # We requested exactly one day
 
        def get_val(field):
            vals = daily.get(field, [None])
            v    = vals[idx] if vals else None
            return float(v) if v is not None else None
 
        # Parse datetime
        from datetime import datetime
        time_val = datetime.fromisoformat(daily["time"][idx]).replace(
            tzinfo=__import__("pytz").timezone("Asia/Jakarta").localize(
                datetime.fromisoformat(daily["time"][idx])
            ).tzinfo
        )
 
        # Compute derived fields
        temp    = get_val("temperature_2m_mean") or 28.0
        humidity = get_val("relative_humidity_2m_mean") or 80.0
        moonphase_val = self._compute_moonphase(target_date)
 
        return {
            "time"               : datetime.combine(
                target_date, datetime.min.time()).replace(
                tzinfo=__import__("pytz").UTC),
            "location_id"        : location.id,
            "tempmax"            : get_val("temperature_2m_max"),
            "tempmin"            : get_val("temperature_2m_min"),
            "temp"               : get_val("temperature_2m_mean"),
            "feelslikemax"       : get_val("apparent_temperature_max"),
            "feelslikemin"       : get_val("apparent_temperature_min"),
            "feelslike"          : get_val("apparent_temperature_mean"),
            "dew"                : get_val("dew_point_2m_mean"),
            "humidity"           : humidity,
            "precip"             : get_val("precipitation_sum"),
            "precipprob"         : get_val("precipitation_probability_mean"),
            "snow"               : get_val("snowfall_sum"),
            "snowdepth"          : get_val("snow_depth_mean"),
            "windgust"           : get_val("wind_gusts_10m_max"),
            "windspeed"          : get_val("wind_speed_10m_max"),
            "winddir"            : get_val("wind_direction_10m_dominant"),
            "sealevelpressure"   : get_val("surface_pressure_mean"),
            "cloudcover"         : get_val("cloud_cover_mean"),
            "visibility"         : get_val("visibility_mean"),
            "solarradiation"     : get_val("shortwave_radiation_sum"),
            "uvindex"            : get_val("uv_index_max"),
            "moonphase"          : moonphase_val,
            "humidity_norm"      : humidity / 100.0 if humidity else None,
        }
 
    @staticmethod
    def _compute_moonphase(d: date) -> float:
        """
        Approximate moon phase (0.0 = new moon, 0.5 = full moon).
        Uses Julian date formula.
        """
        import datetime as dt
        jd = (d.toordinal() + 1721425.5 -
              dt.date(2000, 1, 6).toordinal() - 1721425.5)
        phase = (jd % 29.53059) / 29.53059
        return round(phase, 4)