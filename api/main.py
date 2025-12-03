"""
FastAPI application for Whisper transcription API
"""
import os
import time
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
import torch
import whisper
from whisper import load_model, transcribe

from .database import get_db, init_db, Transcription
from .models import (
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptionListResponse,
    StatsResponse,
    LanguageStatsResponse,
    DateStatsResponse,
    SegmentResponse
)

# Initialize FastAPI app
app = FastAPI(
    title="Whisper Transcription API",
    description="REST API for Whisper speech-to-text transcription with database storage and reporting",
    version="1.0.0"
)

# Configuration
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Global model cache
_model_cache = {}


def get_whisper_model(model_name: str = "turbo"):
    """Get or load Whisper model (with caching)"""
    if model_name not in _model_cache:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model_cache[model_name] = load_model(model_name, device=device)
    return _model_cache[model_name]


@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    init_db()


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Whisper Transcription API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    model: str = Query(default="turbo", description="Whisper model to use"),
    language: Optional[str] = Query(default=None, description="Language code"),
    task: str = Query(default="transcribe", description="Task: transcribe or translate"),
    word_timestamps: bool = Query(default=False, description="Include word timestamps"),
    initial_prompt: Optional[str] = Query(default=None, description="Initial prompt"),
    temperature: float = Query(default=0.0, description="Sampling temperature"),
    db: Session = Depends(get_db)
):
    """
    Transcribe an audio file and save results to database
    """
    start_time = time.time()
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Save uploaded file
    file_extension = Path(file.filename).suffix
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    saved_filename = f"{timestamp}{file_extension}"
    file_path = UPLOAD_DIR / saved_filename
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Load Whisper model
        whisper_model = get_whisper_model(model)
        
        # Transcribe audio
        transcribe_options = {
            "language": language,
            "task": task,
            "word_timestamps": word_timestamps,
            "initial_prompt": initial_prompt,
            "temperature": temperature,
        }
        
        # Remove None values
        transcribe_options = {k: v for k, v in transcribe_options.items() if v is not None}
        
        result = transcribe(whisper_model, str(file_path), **transcribe_options)
        
        processing_time = time.time() - start_time
        
        # Calculate audio duration from segments
        audio_duration = None
        if result.get("segments"):
            last_segment = result["segments"][-1]
            audio_duration = last_segment.get("end", 0)
        
        # Save to database
        db_transcription = Transcription(
            audio_filename=file.filename,
            audio_path=str(file_path),
            transcribed_text=result["text"],
            language=result.get("language"),
            model_used=model,
            task=task,
            segments=result.get("segments"),
            processing_time=processing_time,
            audio_duration=audio_duration,
            metadata={
                "word_timestamps": word_timestamps,
                "temperature": temperature,
                "initial_prompt": initial_prompt
            }
        )
        
        db.add(db_transcription)
        db.commit()
        db.refresh(db_transcription)
        
        # Convert segments to response format
        segments_response = None
        if result.get("segments"):
            segments_response = [
                SegmentResponse(
                    id=seg.get("id", i),
                    start=seg.get("start", 0),
                    end=seg.get("end", 0),
                    text=seg.get("text", ""),
                    words=seg.get("words")
                )
                for i, seg in enumerate(result["segments"])
            ]
        
        return TranscriptionResponse(
            id=db_transcription.id,
            audio_filename=db_transcription.audio_filename,
            transcribed_text=db_transcription.transcribed_text,
            language=db_transcription.language,
            model_used=db_transcription.model_used,
            task=db_transcription.task,
            segments=segments_response,
            processing_time=db_transcription.processing_time,
            audio_duration=db_transcription.audio_duration,
            created_at=db_transcription.created_at
        )
        
    except Exception as e:
        # Clean up file on error
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@app.get("/transcriptions/{transcription_id}", response_model=TranscriptionResponse)
async def get_transcription(
    transcription_id: int,
    db: Session = Depends(get_db)
):
    """Get a specific transcription by ID"""
    transcription = db.query(Transcription).filter(Transcription.id == transcription_id).first()
    
    if not transcription:
        raise HTTPException(status_code=404, detail="Transcription not found")
    
    # Convert segments to response format
    segments_response = None
    if transcription.segments:
        segments_response = [
            SegmentResponse(
                id=seg.get("id", i),
                start=seg.get("start", 0),
                end=seg.get("end", 0),
                text=seg.get("text", ""),
                words=seg.get("words")
            )
            for i, seg in enumerate(transcription.segments)
        ]
    
    return TranscriptionResponse(
        id=transcription.id,
        audio_filename=transcription.audio_filename,
        transcribed_text=transcription.transcribed_text,
        language=transcription.language,
        model_used=transcription.model_used,
        task=transcription.task,
        segments=segments_response,
        processing_time=transcription.processing_time,
        audio_duration=transcription.audio_duration,
        created_at=transcription.created_at
    )


@app.get("/transcriptions", response_model=TranscriptionListResponse)
async def list_transcriptions(
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=10, ge=1, le=100, description="Items per page"),
    language: Optional[str] = Query(default=None, description="Filter by language"),
    model: Optional[str] = Query(default=None, description="Filter by model"),
    db: Session = Depends(get_db)
):
    """List all transcriptions with pagination"""
    query = db.query(Transcription)
    
    # Apply filters
    if language:
        query = query.filter(Transcription.language == language)
    if model:
        query = query.filter(Transcription.model_used == model)
    
    # Get total count
    total = query.count()
    
    # Apply pagination
    offset = (page - 1) * page_size
    transcriptions = query.order_by(desc(Transcription.created_at)).offset(offset).limit(page_size).all()
    
    # Convert to response format
    transcriptions_response = []
    for t in transcriptions:
        segments_response = None
        if t.segments:
            segments_response = [
                SegmentResponse(
                    id=seg.get("id", i),
                    start=seg.get("start", 0),
                    end=seg.get("end", 0),
                    text=seg.get("text", ""),
                    words=seg.get("words")
                )
                for i, seg in enumerate(t.segments)
            ]
        
        transcriptions_response.append(
            TranscriptionResponse(
                id=t.id,
                audio_filename=t.audio_filename,
                transcribed_text=t.transcribed_text,
                language=t.language,
                model_used=t.model_used,
                task=t.task,
                segments=segments_response,
                processing_time=t.processing_time,
                audio_duration=t.audio_duration,
                created_at=t.created_at
            )
        )
    
    return TranscriptionListResponse(
        total=total,
        page=page,
        page_size=page_size,
        transcriptions=transcriptions_response
    )


@app.get("/reports/stats", response_model=StatsResponse)
async def get_statistics(db: Session = Depends(get_db)):
    """Get overall statistics for reporting"""
    total_transcriptions = db.query(Transcription).count()
    
    # Calculate total audio duration
    total_duration_result = db.query(func.sum(Transcription.audio_duration)).scalar()
    total_audio_duration = float(total_duration_result) if total_duration_result else 0.0
    
    # Calculate average processing time
    avg_processing_result = db.query(func.avg(Transcription.processing_time)).scalar()
    average_processing_time = float(avg_processing_result) if avg_processing_result else 0.0
    
    # Language breakdown
    language_counts = db.query(
        Transcription.language,
        func.count(Transcription.id)
    ).group_by(Transcription.language).all()
    languages = {lang or "unknown": count for lang, count in language_counts}
    
    # Model usage breakdown
    model_counts = db.query(
        Transcription.model_used,
        func.count(Transcription.id)
    ).group_by(Transcription.model_used).all()
    models_used = {model: count for model, count in model_counts}
    
    # Task breakdown
    task_counts = db.query(
        Transcription.task,
        func.count(Transcription.id)
    ).group_by(Transcription.task).all()
    tasks = {task: count for task, count in task_counts}
    
    return StatsResponse(
        total_transcriptions=total_transcriptions,
        total_audio_duration=total_audio_duration,
        average_processing_time=average_processing_time,
        languages=languages,
        models_used=models_used,
        tasks=tasks
    )


@app.get("/reports/by-language", response_model=List[LanguageStatsResponse])
async def get_language_statistics(db: Session = Depends(get_db)):
    """Get statistics grouped by language"""
    results = db.query(
        Transcription.language,
        func.count(Transcription.id).label("count"),
        func.sum(Transcription.audio_duration).label("total_duration"),
        func.avg(Transcription.processing_time).label("avg_processing_time")
    ).group_by(Transcription.language).all()
    
    return [
        LanguageStatsResponse(
            language=lang or "unknown",
            count=count,
            total_duration=float(total_duration) if total_duration else 0.0,
            average_processing_time=float(avg_processing_time) if avg_processing_time else 0.0
        )
        for lang, count, total_duration, avg_processing_time in results
    ]


@app.get("/reports/by-date", response_model=List[DateStatsResponse])
async def get_date_statistics(
    days: int = Query(default=30, ge=1, le=365, description="Number of days to include"),
    db: Session = Depends(get_db)
):
    """Get statistics grouped by date"""
    start_date = datetime.utcnow() - timedelta(days=days)
    
    results = db.query(
        func.date(Transcription.created_at).label("date"),
        func.count(Transcription.id).label("count"),
        func.sum(Transcription.audio_duration).label("total_duration")
    ).filter(
        Transcription.created_at >= start_date
    ).group_by(
        func.date(Transcription.created_at)
    ).order_by(desc("date")).all()
    
    return [
        DateStatsResponse(
            date=str(date),
            count=count,
            total_duration=float(total_duration) if total_duration else 0.0
        )
        for date, count, total_duration in results
    ]


@app.delete("/transcriptions/{transcription_id}")
async def delete_transcription(
    transcription_id: int,
    db: Session = Depends(get_db)
):
    """Delete a transcription"""
    transcription = db.query(Transcription).filter(Transcription.id == transcription_id).first()
    
    if not transcription:
        raise HTTPException(status_code=404, detail="Transcription not found")
    
    # Delete associated audio file if exists
    if transcription.audio_path and os.path.exists(transcription.audio_path):
        try:
            os.remove(transcription.audio_path)
        except Exception:
            pass  # Continue even if file deletion fails
    
    db.delete(transcription)
    db.commit()
    
    return {"message": "Transcription deleted successfully", "id": transcription_id}

