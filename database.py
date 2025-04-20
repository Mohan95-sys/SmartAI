import os
import streamlit as st
import pandas as pd
import numpy as np
import datetime
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# Get database URL from environment variables
DATABASE_URL = os.environ.get("DATABASE_URL")

# Always use in-memory SQLite for this application to avoid connection issues
engine = create_engine('sqlite:///:memory:')

# Create a base class for declarative models
Base = declarative_base()

# Define database models
class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationships
    predictions = relationship("CropPrediction", back_populates="user")
    soil_data = relationship("SoilData", back_populates="user")

class CropData(Base):
    __tablename__ = 'crop_data'
    
    id = Column(Integer, primary_key=True)
    crop_name = Column(String(50), nullable=False)
    n_min = Column(Float)  # Minimum Nitrogen requirement
    n_max = Column(Float)  # Maximum Nitrogen requirement
    p_min = Column(Float)  # Minimum Phosphorus requirement
    p_max = Column(Float)  # Maximum Phosphorus requirement
    k_min = Column(Float)  # Minimum Potassium requirement
    k_max = Column(Float)  # Maximum Potassium requirement
    temperature_min = Column(Float)  # Minimum temperature requirement
    temperature_max = Column(Float)  # Maximum temperature requirement
    rainfall_min = Column(Float)  # Minimum rainfall requirement
    rainfall_max = Column(Float)  # Maximum rainfall requirement
    ph_min = Column(Float)  # Minimum pH requirement
    ph_max = Column(Float)  # Maximum pH requirement
    humidity_min = Column(Float)  # Minimum humidity requirement
    humidity_max = Column(Float)  # Maximum humidity requirement
    growing_season = Column(String(50))
    water_requirement = Column(String(100))
    soil_type = Column(String(100))
    description = Column(Text)

class CropPrice(Base):
    __tablename__ = 'crop_prices'
    
    id = Column(Integer, primary_key=True)
    crop_name = Column(String(50), nullable=False)
    price_usd = Column(Float)  # Price in USD per metric ton
    price_inr = Column(Float)  # Price in INR per quintal
    market = Column(String(50))  # Global or India
    change_percent = Column(Float)  # Price change percentage
    last_updated = Column(DateTime, default=datetime.datetime.utcnow)

class SoilData(Base):
    __tablename__ = 'soil_data'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    location = Column(String(100))
    n_value = Column(Float)  # Nitrogen
    p_value = Column(Float)  # Phosphorus
    k_value = Column(Float)  # Potassium
    ph_value = Column(Float)  # pH
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="soil_data")

class CropPrediction(Base):
    __tablename__ = 'crop_predictions'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    soil_data_id = Column(Integer, ForeignKey('soil_data.id'))
    crop_name = Column(String(50), nullable=False)
    confidence = Column(Float)  # Prediction confidence (0-1)
    n_value = Column(Float)  # Nitrogen input
    p_value = Column(Float)  # Phosphorus input
    k_value = Column(Float)  # Potassium input
    temperature = Column(Float)  # Temperature input
    humidity = Column(Float)  # Humidity input
    ph_value = Column(Float)  # pH input
    rainfall = Column(Float)  # Rainfall input
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="predictions")

# Create tables if they don't exist
def create_tables():
    Base.metadata.create_all(engine)

# Create a session to interact with the database
def get_session():
    Session = sessionmaker(bind=engine)
    return Session()

# Initialize the database
def init_db():
    create_tables()
    initialize_sample_data()

# Database initialization function with sample data
def initialize_sample_data():
    session = get_session()
    
    # Check if crop data already exists
    if session.query(CropData).count() == 0:
        # Add sample crop data
        crops = [
            CropData(
                crop_name="Rice",
                n_min=80, n_max=120,
                p_min=40, p_max=60,
                k_min=40, k_max=60,
                temperature_min=22, temperature_max=32,
                rainfall_min=150, rainfall_max=300,
                ph_min=5.5, ph_max=7.0,
                humidity_min=60, humidity_max=90,
                growing_season="Warm season crop",
                water_requirement="High water requirement (150-300 cm)",
                soil_type="Clay or clay loam soils",
                description="Rice is the staple food for more than half of the world's population"
            ),
            CropData(
                crop_name="Wheat",
                n_min=60, n_max=100,
                p_min=30, p_max=50,
                k_min=20, k_max=40,
                temperature_min=15, temperature_max=24,
                rainfall_min=45, rainfall_max=65,
                ph_min=6.0, ph_max=7.5,
                humidity_min=50, humidity_max=70,
                growing_season="Cool season crop",
                water_requirement="Moderate water requirement (45-65 cm)",
                soil_type="Loam or clay loam soils",
                description="Wheat is the most widely grown crop in the world"
            ),
            CropData(
                crop_name="Maize",
                n_min=80, n_max=120,
                p_min=50, p_max=90,
                k_min=40, k_max=70,
                temperature_min=20, temperature_max=32,
                rainfall_min=50, rainfall_max=75,
                ph_min=5.5, ph_max=7.5,
                humidity_min=50, humidity_max=80,
                growing_season="Warm season crop",
                water_requirement="Moderate water requirement (50-75 cm)",
                soil_type="Well-drained soils",
                description="Maize is used for food, feed, and biofuel production"
            ),
            CropData(
                crop_name="Chickpea",
                n_min=20, n_max=40,
                p_min=40, p_max=60,
                k_min=30, k_max=50,
                temperature_min=18, temperature_max=26,
                rainfall_min=40, rainfall_max=50,
                ph_min=6.5, ph_max=8.0,
                humidity_min=40, humidity_max=70,
                growing_season="Cool season crop",
                water_requirement="Low water requirement (40-50 cm)",
                soil_type="Sandy loam soils",
                description="Chickpea is a good source of protein and fiber"
            ),
            CropData(
                crop_name="Cotton",
                n_min=60, n_max=100,
                p_min=30, p_max=50,
                k_min=70, k_max=100,
                temperature_min=25, temperature_max=35,
                rainfall_min=70, rainfall_max=120,
                ph_min=6.0, ph_max=8.0,
                humidity_min=50, humidity_max=70,
                growing_season="Warm season crop",
                water_requirement="High water requirement (70-120 cm)",
                soil_type="Well-drained soils",
                description="Cotton is the most important natural fiber crop"
            ),
            CropData(
                crop_name="Tea",
                n_min=50, n_max=80,
                p_min=20, p_max=40,
                k_min=40, k_max=70,
                temperature_min=20, temperature_max=30,
                rainfall_min=150, rainfall_max=250,
                ph_min=4.5, ph_max=5.5,
                humidity_min=70, humidity_max=90,
                growing_season="Year-round crop",
                water_requirement="High water requirement (150-250 cm)",
                soil_type="Well-drained acidic soils",
                description="Tea is the most consumed beverage in the world after water"
            ),
            CropData(
                crop_name="Millet",
                n_min=30, n_max=60,
                p_min=20, p_max=40,
                k_min=30, k_max=50,
                temperature_min=25, temperature_max=35,
                rainfall_min=30, rainfall_max=50,
                ph_min=5.5, ph_max=7.5,
                humidity_min=40, humidity_max=60,
                growing_season="Warm season crop",
                water_requirement="Low water requirement (30-50 cm)",
                soil_type="Sandy loam soils",
                description="Millet is drought resistant and can grow in poor soil conditions"
            )
        ]
        
        # Add crops to the database
        for crop in crops:
            session.add(crop)
    
    # Check if crop price data already exists
    if session.query(CropPrice).count() == 0:
        # Add global crop prices
        global_crops = [
            CropPrice(
                crop_name="Rice",
                price_usd=550,
                price_inr=None,
                market="Global",
                change_percent=2.5,
                last_updated=datetime.datetime.utcnow()
            ),
            CropPrice(
                crop_name="Wheat",
                price_usd=320,
                price_inr=None,
                market="Global",
                change_percent=-1.2,
                last_updated=datetime.datetime.utcnow()
            ),
            CropPrice(
                crop_name="Maize",
                price_usd=215,
                price_inr=None,
                market="Global",
                change_percent=0.8,
                last_updated=datetime.datetime.utcnow()
            ),
            CropPrice(
                crop_name="Soybean",
                price_usd=510,
                price_inr=None,
                market="Global",
                change_percent=3.1,
                last_updated=datetime.datetime.utcnow()
            ),
            CropPrice(
                crop_name="Cotton",
                price_usd=92,
                price_inr=None,
                market="Global",
                change_percent=-0.5,
                last_updated=datetime.datetime.utcnow()
            ),
            CropPrice(
                crop_name="Sugar",
                price_usd=18,
                price_inr=None,
                market="Global",
                change_percent=1.7,
                last_updated=datetime.datetime.utcnow()
            ),
            CropPrice(
                crop_name="Coffee",
                price_usd=170,
                price_inr=None,
                market="Global",
                change_percent=-2.1,
                last_updated=datetime.datetime.utcnow()
            ),
            CropPrice(
                crop_name="Tea",
                price_usd=310,
                price_inr=None,
                market="Global",
                change_percent=0.4,
                last_updated=datetime.datetime.utcnow()
            )
        ]
        
        # Add India crop prices
        india_crops = [
            CropPrice(
                crop_name="Rice",
                price_usd=None,
                price_inr=25400,
                market="India",
                change_percent=1.8,
                last_updated=datetime.datetime.utcnow()
            ),
            CropPrice(
                crop_name="Wheat",
                price_usd=None,
                price_inr=26800,
                market="India",
                change_percent=0.7,
                last_updated=datetime.datetime.utcnow()
            ),
            CropPrice(
                crop_name="Maize",
                price_usd=None,
                price_inr=18500,
                market="India",
                change_percent=-0.5,
                last_updated=datetime.datetime.utcnow()
            ),
            CropPrice(
                crop_name="Cotton",
                price_usd=None,
                price_inr=7200,
                market="India",
                change_percent=2.2,
                last_updated=datetime.datetime.utcnow()
            ),
            CropPrice(
                crop_name="Sugarcane",
                price_usd=None,
                price_inr=3500,
                market="India",
                change_percent=1.5,
                last_updated=datetime.datetime.utcnow()
            ),
            CropPrice(
                crop_name="Tea",
                price_usd=None,
                price_inr=38000,
                market="India",
                change_percent=-0.9,
                last_updated=datetime.datetime.utcnow()
            ),
            CropPrice(
                crop_name="Chickpea",
                price_usd=None,
                price_inr=7500,
                market="India",
                change_percent=1.2,
                last_updated=datetime.datetime.utcnow()
            ),
            CropPrice(
                crop_name="Millet",
                price_usd=None,
                price_inr=3200,
                market="India",
                change_percent=-1.5,
                last_updated=datetime.datetime.utcnow()
            )
        ]
        
        # Add prices to the database
        for price in global_crops + india_crops:
            session.add(price)
    
    # Commit the changes
    session.commit()
    session.close()

# Database operations for crop prices
def get_crop_prices_from_db(market=None):
    session = get_session()
    try:
        query = session.query(CropPrice)
        
        if market:
            query = query.filter(CropPrice.market == market)
            
        prices = query.all()
        
        # Convert to DataFrame for easier processing
        if market == "Global":
            df = pd.DataFrame([
                {
                    "Crop": p.crop_name,
                    "Price (USD/MT)": p.price_usd,
                    "Change (%)": p.change_percent,
                    "Last Updated": p.last_updated.strftime('%Y-%m-%d')
                }
                for p in prices
            ])
        elif market == "India":
            df = pd.DataFrame([
                {
                    "Crop": p.crop_name,
                    "Price (INR/Quintal)": p.price_inr,
                    "Change (%)": p.change_percent,
                    "Last Updated": p.last_updated.strftime('%Y-%m-%d')
                }
                for p in prices
            ])
        else:
            # Both markets
            global_prices = [
                {
                    "Crop": p.crop_name,
                    "Price (USD/MT)": p.price_usd,
                    "Change (%)": p.change_percent,
                    "Market": p.market,
                    "Last Updated": p.last_updated.strftime('%Y-%m-%d')
                }
                for p in prices if p.market == "Global"
            ]
            
            india_prices = [
                {
                    "Crop": p.crop_name,
                    "Price (INR/Quintal)": p.price_inr,
                    "Change (%)": p.change_percent,
                    "Market": p.market,
                    "Last Updated": p.last_updated.strftime('%Y-%m-%d')
                }
                for p in prices if p.market == "India"
            ]
            
            return pd.DataFrame(global_prices), pd.DataFrame(india_prices)
        
        return df
        
    finally:
        session.close()

# Database operations for crop data
def get_crop_data_from_db(crop_name=None):
    session = get_session()
    try:
        query = session.query(CropData)
        
        if crop_name:
            query = query.filter(CropData.crop_name == crop_name)
            
        crops = query.all()
        
        if crop_name and crops:
            # Return a single crop as a dictionary
            crop = crops[0]
            return {
                "crop_name": crop.crop_name,
                "n_range": (crop.n_min, crop.n_max),
                "p_range": (crop.p_min, crop.p_max),
                "k_range": (crop.k_min, crop.k_max),
                "temperature_range": (crop.temperature_min, crop.temperature_max),
                "rainfall_range": (crop.rainfall_min, crop.rainfall_max),
                "ph_range": (crop.ph_min, crop.ph_max),
                "humidity_range": (crop.humidity_min, crop.humidity_max),
                "growing_season": crop.growing_season,
                "water_requirement": crop.water_requirement,
                "soil_type": crop.soil_type,
                "description": crop.description
            }
        else:
            # Return all crops as a DataFrame
            df = pd.DataFrame([
                {
                    "Crop": c.crop_name,
                    "Nitrogen (min-max)": f"{c.n_min}-{c.n_max}",
                    "Phosphorus (min-max)": f"{c.p_min}-{c.p_max}",
                    "Potassium (min-max)": f"{c.k_min}-{c.k_max}",
                    "Temperature (°C)": f"{c.temperature_min}-{c.temperature_max}",
                    "Rainfall (mm)": f"{c.rainfall_min}-{c.rainfall_max}",
                    "pH": f"{c.ph_min}-{c.ph_max}",
                    "Humidity (%)": f"{c.humidity_min}-{c.humidity_max}"
                }
                for c in crops
            ])
            return df
        
    finally:
        session.close()

# Save prediction to database
def save_prediction(user_id, n, p, k, temperature, humidity, ph, rainfall, crop_name, confidence=None):
    session = get_session()
    try:
        prediction = CropPrediction(
            user_id=user_id,
            crop_name=crop_name,
            confidence=confidence,
            n_value=n,
            p_value=p,
            k_value=k,
            temperature=temperature,
            humidity=humidity,
            ph_value=ph,
            rainfall=rainfall
        )
        session.add(prediction)
        session.commit()
        return prediction.id
    finally:
        session.close()

# Get user predictions
def get_user_predictions(user_id):
    session = get_session()
    try:
        predictions = session.query(CropPrediction).filter(
            CropPrediction.user_id == user_id
        ).order_by(CropPrediction.created_at.desc()).all()
        
        df = pd.DataFrame([
            {
                "Prediction ID": p.id,
                "Crop": p.crop_name,
                "Nitrogen": p.n_value,
                "Phosphorus": p.p_value,
                "Potassium": p.k_value,
                "Temperature": p.temperature,
                "Humidity": p.humidity,
                "pH": p.ph_value,
                "Rainfall": p.rainfall,
                "Date": p.created_at.strftime('%Y-%m-%d %H:%M:%S')
            }
            for p in predictions
        ])
        return df
    finally:
        session.close()

# Register new user
def register_user(username, email):
    session = get_session()
    try:
        # Check if user already exists
        existing_user = session.query(User).filter(
            (User.username == username) | (User.email == email)
        ).first()
        
        if existing_user:
            return None
        
        user = User(username=username, email=email)
        session.add(user)
        session.commit()
        return user.id
    finally:
        session.close()

# Get user by username
def get_user_by_username(username):
    session = get_session()
    try:
        user = session.query(User).filter(User.username == username).first()
        return user
    finally:
        session.close()

# Database initialization is handled by init_db()