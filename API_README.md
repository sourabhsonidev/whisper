# Whisper Transcription API

A REST API wrapper for OpenAI's Whisper speech-to-text model with database storage and reporting capabilities.

## Features

- 🎤 **Audio Transcription**: Upload audio files and get transcriptions
- 💾 **Database Storage**: All transcriptions are saved to a database for later retrieval
- 📊 **Reporting & Analytics**: Built-in endpoints for statistics and reporting
- 🌍 **Multi-language Support**: Automatic language detection or manual specification
- ⚡ **Model Caching**: Models are cached in memory for faster subsequent requests
- 📝 **Word Timestamps**: Optional word-level timestamp extraction
- 🔍 **Filtering & Pagination**: Query transcriptions by language, model, date, etc.

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure ffmpeg is installed (required for audio processing):
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# Windows (using Chocolatey)
choco install ffmpeg
```

## Running the API

### Option 1: Using the run script
```bash
python -m whisper.api.run
```

### Option 2: Using uvicorn directly
```bash
uvicorn whisper.api.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

## API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### 1. Transcribe Audio
**POST** `/transcribe`

Upload an audio file to transcribe.

**Query Parameters:**
- `model` (default: "turbo"): Whisper model to use (tiny, base, small, medium, large, turbo)
- `language` (optional): Language code (e.g., "en", "es", "fr")
- `task` (default: "transcribe"): "transcribe" or "translate"
- `word_timestamps` (default: false): Include word-level timestamps
- `initial_prompt` (optional): Initial prompt for context
- `temperature` (default: 0.0): Sampling temperature

**Example:**
```bash
curl -X POST "http://localhost:8000/transcribe?model=turbo&language=en" \
  -F "file=@audio.mp3"
```

### 2. Get Transcription
**GET** `/transcriptions/{transcription_id}`

Retrieve a specific transcription by ID.

**Example:**
```bash
curl http://localhost:8000/transcriptions/1
```

### 3. List Transcriptions
**GET** `/transcriptions`

List all transcriptions with pagination and filtering.

**Query Parameters:**
- `page` (default: 1): Page number
- `page_size` (default: 10): Items per page (max 100)
- `language` (optional): Filter by language code
- `model` (optional): Filter by model name

**Example:**
```bash
curl "http://localhost:8000/transcriptions?page=1&page_size=20&language=en"
```

### 4. Get Statistics
**GET** `/reports/stats`

Get overall statistics including:
- Total transcriptions
- Total audio duration processed
- Average processing time
- Total tokens
- Total cost
- Language breakdown
- Model usage breakdown
- Task breakdown

**Example:**
```bash
curl http://localhost:8000/reports/stats
```

### 5. Language Statistics
**GET** `/reports/by-language`

Get statistics grouped by language.

**Example:**
```bash
curl http://localhost:8000/reports/by-language
```

### 6. Date Statistics
**GET** `/reports/by-date`

Get statistics grouped by date.

**Query Parameters:**
- `days` (default: 30): Number of days to include (1-365)

**Example:**
```bash
curl "http://localhost:8000/reports/by-date?days=7"
```

### 7. Export to Excel
**GET** `/export/excel`

Export transcriptions to Excel format with count, tokens, and cost information.

**Query Parameters:**
- `language` (optional): Filter by language code
- `model` (optional): Filter by model name
- `start_date` (optional): Start date filter (YYYY-MM-DD)
- `end_date` (optional): End date filter (YYYY-MM-DD)

**Response:**
Returns an Excel file (.xlsx) with two sheets:
- **Transcriptions**: All transcription data including ID, filename, text, language, model, task, token count, cost, duration, processing time, and created date
- **Summary**: Summary statistics including total count, total tokens, total cost, total duration, and average processing time

**Example:**
```bash
# Export all transcriptions
curl -O -J "http://localhost:8000/export/excel"

# Export with filters
curl -O -J "http://localhost:8000/export/excel?language=en&start_date=2024-01-01&end_date=2024-12-31"
```

### 8. Delete Transcription
**DELETE** `/transcriptions/{transcription_id}`

Delete a transcription and its associated audio file.

**Example:**
```bash
curl -X DELETE http://localhost:8000/transcriptions/1
```

## Database

By default, the API uses SQLite database (`whisper_transcriptions.db`). To use PostgreSQL or MySQL, set the `DATABASE_URL` environment variable:

```bash
export DATABASE_URL="postgresql://user:password@localhost/whisper_db"
# or
export DATABASE_URL="mysql://user:password@localhost/whisper_db"
```

## Database Schema

The `transcriptions` table stores:
- `id`: Primary key
- `audio_filename`: Original filename
- `audio_path`: Path to saved audio file
- `transcribed_text`: Full transcription text
- `language`: Detected/specified language
- `model_used`: Whisper model used
- `task`: Task type (transcribe/translate)
- `segments`: JSON array of transcription segments with timestamps
- `processing_time`: Time taken to process (seconds)
- `audio_duration`: Duration of audio (seconds)
- `token_count`: Number of tokens in the transcription
- `cost`: Calculated cost in USD based on tokens and model
- `created_at`: Timestamp
- `updated_at`: Timestamp
- `metadata`: Additional metadata (JSON)

## Example Usage with Python

```python
import requests

# Transcribe an audio file
with open("audio.mp3", "rb") as f:
    response = requests.post(
        "http://localhost:8000/transcribe",
        files={"file": f},
        params={"model": "turbo", "language": "en"}
    )
    result = response.json()
    print(f"Transcription ID: {result['id']}")
    print(f"Text: {result['transcribed_text']}")
    print(f"Token Count: {result['token_count']}")
    print(f"Cost: ${result['cost']:.6f}")

# Get statistics
stats = requests.get("http://localhost:8000/reports/stats").json()
print(f"Total transcriptions: {stats['total_transcriptions']}")
print(f"Total tokens: {stats['total_tokens']}")
print(f"Total cost: ${stats['total_cost']:.6f}")
print(f"Languages: {stats['languages']}")

# Export to Excel
response = requests.get("http://localhost:8000/export/excel?language=en")
with open("transcriptions.xlsx", "wb") as f:
    f.write(response.content)
print("Excel file exported successfully!")
```

## Configuration

### Environment Variables

- `DATABASE_URL`: Database connection string (default: SQLite)
- `UPLOAD_DIR`: Directory for uploaded files (default: `uploads/`)

### Model Selection

Available Whisper models:
- `tiny` / `tiny.en`: Fastest, least accurate (~39M parameters)
- `base` / `base.en`: Good balance (~74M parameters)
- `small` / `small.en`: Better accuracy (~244M parameters)
- `medium` / `medium.en`: High accuracy (~769M parameters)
- `large` / `large-v3`: Best accuracy (~1550M parameters)
- `turbo`: Optimized for speed (~809M parameters, recommended)

## Notes

- Models are downloaded automatically on first use and cached
- Audio files are saved in the `uploads/` directory
- The API supports all audio formats supported by ffmpeg
- For production, consider using a reverse proxy (nginx) and process manager (systemd/supervisor)

## License

Same as Whisper (MIT License)

