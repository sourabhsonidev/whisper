"""
Example client script for Whisper API
"""
import requests
import json

API_BASE_URL = "http://localhost:8000"


def transcribe_audio(file_path: str, model: str = "turbo", language: str = None):
    """Transcribe an audio file"""
    url = f"{API_BASE_URL}/transcribe"
    
    params = {"model": model}
    if language:
        params["language"] = language
    
    with open(file_path, "rb") as f:
        files = {"file": f}
        response = requests.post(url, files=files, params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None


def get_transcription(transcription_id: int):
    """Get a transcription by ID"""
    url = f"{API_BASE_URL}/transcriptions/{transcription_id}"
    response = requests.get(url)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        return None


def list_transcriptions(page: int = 1, page_size: int = 10, language: str = None):
    """List transcriptions"""
    url = f"{API_BASE_URL}/transcriptions"
    params = {"page": page, "page_size": page_size}
    
    if language:
        params["language"] = language
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        return None


def get_statistics():
    """Get overall statistics"""
    url = f"{API_BASE_URL}/reports/stats"
    response = requests.get(url)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        return None


def get_language_statistics():
    """Get language statistics"""
    url = f"{API_BASE_URL}/reports/by-language"
    response = requests.get(url)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        return None


def get_date_statistics(days: int = 30):
    """Get date statistics"""
    url = f"{API_BASE_URL}/reports/by-date"
    params = {"days": days}
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        return None


if __name__ == "__main__":
    # Example usage
    
    # 1. Transcribe an audio file
    print("Transcribing audio file...")
    result = transcribe_audio("path/to/your/audio.mp3", model="turbo", language="en")
    if result:
        print(f"Transcription ID: {result['id']}")
        print(f"Text: {result['transcribed_text']}")
        print(f"Language: {result['language']}")
        print(f"Processing time: {result['processing_time']:.2f}s")
    
    # 2. Get statistics
    print("\nGetting statistics...")
    stats = get_statistics()
    if stats:
        print(f"Total transcriptions: {stats['total_transcriptions']}")
        print(f"Total audio duration: {stats['total_audio_duration']:.2f}s")
        print(f"Average processing time: {stats['average_processing_time']:.2f}s")
        print(f"Languages: {json.dumps(stats['languages'], indent=2)}")
        print(f"Models used: {json.dumps(stats['models_used'], indent=2)}")
    
    # 3. List transcriptions
    print("\nListing transcriptions...")
    transcriptions = list_transcriptions(page=1, page_size=5)
    if transcriptions:
        print(f"Total: {transcriptions['total']}")
        print(f"Page: {transcriptions['page']}")
        for t in transcriptions['transcriptions']:
            print(f"  - ID {t['id']}: {t['audio_filename']} ({t['language']})")
    
    # 4. Get language statistics
    print("\nLanguage statistics...")
    lang_stats = get_language_statistics()
    if lang_stats:
        for stat in lang_stats:
            print(f"  {stat['language']}: {stat['count']} transcriptions, "
                  f"{stat['total_duration']:.2f}s total duration")

