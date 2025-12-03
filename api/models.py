"""
Pydantic models for API request/response schemas
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class TranscriptionRequest(BaseModel):
    """Request model for transcription"""
    model: str = Field(default="turbo", description="Whisper model to use")
    language: Optional[str] = Field(default=None, description="Language code (auto-detect if None)")
    task: str = Field(default="transcribe", description="Task: 'transcribe' or 'translate'")
    word_timestamps: bool = Field(default=False, description="Include word-level timestamps")
    initial_prompt: Optional[str] = Field(default=None, description="Initial prompt for context")
    temperature: float = Field(default=0.0, description="Sampling temperature")


class SegmentResponse(BaseModel):
    """Response model for transcription segments"""
    id: int
    start: float
    end: float
    text: str
    words: Optional[List[Dict[str, Any]]] = None


class TranscriptionResponse(BaseModel):
    """Response model for transcription"""
    id: int
    audio_filename: str
    transcribed_text: str
    language: Optional[str]
    model_used: str
    task: str
    segments: Optional[List[SegmentResponse]]
    processing_time: Optional[float]
    audio_duration: Optional[float]
    created_at: datetime
    
    class Config:
        from_attributes = True


class TranscriptionListResponse(BaseModel):
    """Response model for list of transcriptions"""
    total: int
    page: int
    page_size: int
    transcriptions: List[TranscriptionResponse]


class StatsResponse(BaseModel):
    """Response model for statistics"""
    total_transcriptions: int
    total_audio_duration: float
    average_processing_time: float
    languages: Dict[str, int]
    models_used: Dict[str, int]
    tasks: Dict[str, int]


class LanguageStatsResponse(BaseModel):
    """Response model for language statistics"""
    language: str
    count: int
    total_duration: float
    average_processing_time: float


class DateStatsResponse(BaseModel):
    """Response model for date-based statistics"""
    date: str
    count: int
    total_duration: float

