#!/usr/bin/env python3
"""
FastAPI Voice Assistant API
Supports Indian languages: Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, Kannada, Malayalam, Punjabi, Odia

The model is loaded ONCE at startup and kept in memory.
Audio files are received via API - no rebuild needed.
"""

import os
import asyncio
import json
import uuid
import tempfile
import shutil
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

import requests
import edge_tts
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from faster_whisper import WhisperModel

# ============================================================================
# Configuration
# ============================================================================

# Language code mapping for Edge TTS voices
LANGUAGE_VOICES = {
    "hi": "hi-IN-SwaraNeural",      # Hindi
    "ta": "ta-IN-PallaviNeural",    # Tamil
    "te": "te-IN-ShrutiNeural",     # Telugu
    "bn": "bn-IN-TanishaaNeural",   # Bengali
    "mr": "mr-IN-AarohiNeural",     # Marathi
    "gu": "gu-IN-DhwaniNeural",     # Gujarati
    "kn": "kn-IN-SapnaNeural",      # Kannada
    "ml": "ml-IN-SobhanaNeural",    # Malayalam
    "pa": "pa-IN-GurpreetNeural",   # Punjabi
    "or": "or-IN-SubhasiniNeural",  # Odia
    "en": "en-IN-NeerjaNeural",     # English (Indian)
}

LANGUAGE_NAMES = {
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "bn": "Bengali",
    "mr": "Marathi",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "or": "Odia",
    "en": "English",
}

FALLBACK_VOICE = "hi-IN-SwaraNeural"

# Ollama API endpoint
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

# Output directory for generated audio
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/app/output")

# ============================================================================
# Global Model Instance (loaded once, stays in RAM)
# ============================================================================

whisper_model: Optional[WhisperModel] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the Whisper model at startup, keep it in memory."""
    global whisper_model
    
    print("🚀 Starting Voice Assistant API...")
    print("🎤 Loading Whisper large-v3 model (this may take a moment)...")
    
    whisper_model = WhisperModel(
        "large-v3",
        device="cuda",
        compute_type="int8_float16"
    )
    
    print("✅ Whisper model loaded and ready!")
    print(f"🤖 LLM Backend: {OLLAMA_HOST} (model: {OLLAMA_MODEL})")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    yield
    
    # Cleanup on shutdown
    print("👋 Shutting down Voice Assistant API...")
    whisper_model = None


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Multilingual Voice Assistant API",
    description="Process voice queries in Indian languages with Whisper + Ollama",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================================
# Response Models
# ============================================================================

class TranscriptionResponse(BaseModel):
    transcription: str
    language: str
    language_name: str
    confidence: float


class VoiceAssistantResponse(BaseModel):
    request_id: str
    input_audio: str
    detected_language: str
    language_name: str
    confidence: float
    transcription: str
    llm_response: str
    output_audio: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    ollama_host: str
    ollama_model: str


# ============================================================================
# Helper Functions
# ============================================================================

def get_voice_for_language(lang_code: str) -> str:
    """Get the appropriate TTS voice for a language."""
    return LANGUAGE_VOICES.get(lang_code, FALLBACK_VOICE)


def get_language_name(code: str) -> str:
    """Get human-readable language name from code."""
    return LANGUAGE_NAMES.get(code, "Hindi")


def transcribe_audio(audio_path: str) -> tuple[str, str, float]:
    """Transcribe audio using the pre-loaded Whisper model."""
    global whisper_model
    
    if whisper_model is None:
        raise RuntimeError("Whisper model not loaded")
    
    segments, info = whisper_model.transcribe(audio_path)
    full_text = " ".join([seg.text.strip() for seg in segments])
    
    return full_text, info.language, info.language_probability


def query_ollama(prompt: str, language: str, language_name: str) -> str:
    """Send query to Ollama and get response in the same language."""
    
    system_prompt = f"""You are a helpful AI assistant. The user is speaking in {language_name}.
You MUST respond in {language_name} only. Keep your response conversational and natural.
Do not include any English text unless the user specifically asks for it.
Be concise but helpful."""

    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "system": system_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 512
                }
            },
            timeout=120
        )
        response.raise_for_status()
        result = response.json()
        return result.get("response", "Sorry, I couldn't generate a response.")
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail=f"Could not connect to Ollama at {OLLAMA_HOST}. Make sure Ollama is running."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying Ollama: {str(e)}")


async def text_to_speech(text: str, language: str, output_path: str) -> str:
    """Convert text to speech using Edge TTS."""
    voice = get_voice_for_language(language)
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)
    return output_path


async def save_upload_file(upload_file: UploadFile, destination: str) -> None:
    """Save an uploaded file to disk."""
    with open(destination, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)


def cleanup_file(path: str) -> None:
    """Remove a file if it exists."""
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the API and model are ready."""
    return HealthResponse(
        status="healthy" if whisper_model else "model_not_loaded",
        model_loaded=whisper_model is not None,
        ollama_host=OLLAMA_HOST,
        ollama_model=OLLAMA_MODEL,
    )


@app.get("/languages")
async def list_languages():
    """List all supported languages."""
    return {
        "supported_languages": [
            {"code": code, "name": name, "voice": LANGUAGE_VOICES.get(code)}
            for code, name in LANGUAGE_NAMES.items()
        ]
    }


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_endpoint(
    audio: UploadFile = File(..., description="Audio file to transcribe"),
    background_tasks: BackgroundTasks = None,
):
    """
    Transcribe audio to text.
    Supports various audio formats (mp3, m4a, wav, ogg, etc.)
    """
    # Save uploaded file to temp location
    suffix = Path(audio.filename).suffix if audio.filename else ".mp3"
    temp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}{suffix}")
    
    try:
        await save_upload_file(audio, temp_path)
        
        transcription, language, confidence = transcribe_audio(temp_path)
        
        return TranscriptionResponse(
            transcription=transcription,
            language=language,
            language_name=get_language_name(language),
            confidence=confidence,
        )
    finally:
        # Clean up temp file
        if background_tasks:
            background_tasks.add_task(cleanup_file, temp_path)
        else:
            cleanup_file(temp_path)


@app.post("/process", response_model=VoiceAssistantResponse)
async def process_voice_query(
    audio: UploadFile = File(..., description="Audio file to process"),
    background_tasks: BackgroundTasks = None,
):
    """
    Full pipeline: Audio -> Transcription -> LLM -> Voice Response
    
    This is the main endpoint for the voice assistant.
    Upload an audio file and receive:
    - Transcription of the audio
    - LLM response in the same language
    - Generated audio response
    """
    request_id = str(uuid.uuid4())
    
    # Save uploaded file to temp location
    suffix = Path(audio.filename).suffix if audio.filename else ".mp3"
    temp_input = os.path.join(tempfile.gettempdir(), f"{request_id}_input{suffix}")
    output_audio_path = os.path.join(OUTPUT_DIR, f"{request_id}_response.mp3")
    
    try:
        # Save the uploaded audio
        await save_upload_file(audio, temp_input)
        
        # Step 1: Transcribe
        print(f"📝 [{request_id}] Transcribing audio...")
        transcription, language, confidence = transcribe_audio(temp_input)
        language_name = get_language_name(language)
        print(f"✅ [{request_id}] Detected: {language_name} ({confidence:.2%})")
        
        # Step 2: Query LLM
        print(f"🧠 [{request_id}] Querying {OLLAMA_MODEL}...")
        llm_response = query_ollama(transcription, language, language_name)
        print(f"✅ [{request_id}] Got LLM response")
        
        # Step 3: Generate voice response
        print(f"🎵 [{request_id}] Generating voice response...")
        await text_to_speech(llm_response, language, output_audio_path)
        print(f"✅ [{request_id}] Voice response saved")
        
        return VoiceAssistantResponse(
            request_id=request_id,
            input_audio=audio.filename or "uploaded_audio",
            detected_language=language,
            language_name=language_name,
            confidence=confidence,
            transcription=transcription,
            llm_response=llm_response,
            output_audio=f"/audio/{request_id}_response.mp3",
        )
    finally:
        # Clean up temp input file
        if background_tasks:
            background_tasks.add_task(cleanup_file, temp_input)
        else:
            cleanup_file(temp_input)


@app.post("/whatsapp/webhook")
async def whatsapp_webhook(
    audio_url: str = Form(None),
    audio: UploadFile = File(None),
    phone_number: str = Form(None),
    message_id: str = Form(None),
    background_tasks: BackgroundTasks = None,
):
    """
    Webhook endpoint for WhatsApp integration.
    
    Can accept either:
    - audio: Direct file upload
    - audio_url: URL to download the audio from (for WhatsApp Cloud API)
    
    Returns the processed response with audio URL.
    """
    request_id = str(uuid.uuid4())
    temp_input = None
    
    try:
        if audio:
            # Direct file upload
            suffix = Path(audio.filename).suffix if audio.filename else ".mp3"
            temp_input = os.path.join(tempfile.gettempdir(), f"{request_id}_input{suffix}")
            await save_upload_file(audio, temp_input)
        elif audio_url:
            # Download from URL (for WhatsApp Cloud API)
            temp_input = os.path.join(tempfile.gettempdir(), f"{request_id}_input.ogg")
            
            # You may need to add WhatsApp auth headers here
            response = requests.get(audio_url, timeout=30)
            response.raise_for_status()
            
            with open(temp_input, "wb") as f:
                f.write(response.content)
        else:
            raise HTTPException(
                status_code=400,
                detail="Either 'audio' file or 'audio_url' must be provided"
            )
        
        # Process the audio
        output_audio_path = os.path.join(OUTPUT_DIR, f"{request_id}_response.mp3")
        
        # Transcribe
        transcription, language, confidence = transcribe_audio(temp_input)
        language_name = get_language_name(language)
        
        # Query LLM
        llm_response = query_ollama(transcription, language, language_name)
        
        # Generate voice response
        await text_to_speech(llm_response, language, output_audio_path)
        
        return JSONResponse({
            "request_id": request_id,
            "phone_number": phone_number,
            "message_id": message_id,
            "detected_language": language,
            "language_name": language_name,
            "transcription": transcription,
            "response_text": llm_response,
            "response_audio_url": f"/audio/{request_id}_response.mp3",
        })
    finally:
        if temp_input and background_tasks:
            background_tasks.add_task(cleanup_file, temp_input)
        elif temp_input:
            cleanup_file(temp_input)


@app.get("/audio/{filename}")
async def get_audio(filename: str):
    """Download a generated audio response."""
    file_path = os.path.join(OUTPUT_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(
        file_path,
        media_type="audio/mpeg",
        filename=filename,
    )


@app.delete("/audio/{filename}")
async def delete_audio(filename: str):
    """Delete a generated audio response."""
    file_path = os.path.join(OUTPUT_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    os.remove(file_path)
    return {"message": f"Deleted {filename}"}


# ============================================================================
# Run with: uvicorn api:app --host 0.0.0.0 --port 8000
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
