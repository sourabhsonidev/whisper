"""
Database models and configuration for Whisper API
"""
from datetime import datetime
from typing import Optional
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

Base = declarative_base()


class Transcription(Base):
    """Model for storing transcription results"""
    __tablename__ = "transcriptions"

    id = Column(Integer, primary_key=True, index=True)
    audio_filename = Column(String(255), nullable=False, index=True)
    audio_path = Column(String(500), nullable=True)
    transcribed_text = Column(Text, nullable=False)
    language = Column(String(10), nullable=True, index=True)
    model_used = Column(String(50), nullable=False, index=True)
    task = Column(String(20), default="transcribe")  # transcribe or translate
    segments = Column(JSON, nullable=True)  # Store full segments data
    processing_time = Column(Float, nullable=True)  # Time taken in seconds
    audio_duration = Column(Float, nullable=True)  # Duration of audio in seconds
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Additional metadata
    metadata = Column(JSON, nullable=True)  # Store any additional metadata


# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./whisper_transcriptions.db")

# Create engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

