from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Index
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()

class Location(Base):
    __tablename__ = 'locations'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)

class WeatherObservation(Base):
    __tablename__ = 'weather_observations'
    time = Column(DateTime(timezone=True), primary_key=True, default=datetime.utcnow)
    location_id = Column(Integer, ForeignKey('locations.id'), primary_key=True)
    temperature = Column(Float)
    humidity = Column(Float)
    rainfall = Column(Float)
    
    __table_args__ = (
        Index('ix_weather_location_time', 'location_id', 'time', postgresql_using='btree'),
    )

class PredictionLog(Base):
    __tablename__ = 'predictions_log'
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), default=datetime.utcnow)
    location_id = Column(Integer, ForeignKey('locations.id'))
    predicted_risk_score = Column(Float)