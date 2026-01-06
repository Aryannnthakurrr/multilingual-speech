#!/usr/bin/env python3
"""
Multilingual Voice Assistant
Supports Indian languages: Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, Kannada, Malayalam, Punjabi, Odia
"""

import sys
import os
import asyncio
import json
import requests
import edge_tts
from faster_whisper import WhisperModel
from pathlib import Path

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
    "pa": "pa-IN-GurpreetNeural",   # Punjabi (not available, fallback to Hindi)
    "or": "or-IN-SubhasiniNeural",  # Odia
    "en": "en-IN-NeerjaNeural",     # English (Indian)
}

# Fallback voices if primary not available
FALLBACK_VOICE = "hi-IN-SwaraNeural"

# Ollama API endpoint
# In WSL Docker: use host.docker.internal OR the WSL IP
# Set OLLAMA_HOST env var to override
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")


def get_voice_for_language(lang_code: str) -> str:
    """Get the appropriate TTS voice for a language."""
    return LANGUAGE_VOICES.get(lang_code, FALLBACK_VOICE)


def transcribe_audio(audio_path: str) -> tuple[str, str, float]:
    """
    Transcribe audio using Whisper large-v3.
    Returns: (transcribed_text, detected_language, confidence)
    """
    print(f"🎤 Loading Whisper model...")
    model = WhisperModel(
        "large-v3",
        device="cuda",
        compute_type="int8_float16"
    )
    
    print(f"🔊 Transcribing: {audio_path}")
    segments, info = model.transcribe(audio_path)
    
    # Collect all segments
    full_text = " ".join([seg.text.strip() for seg in segments])
    
    return full_text, info.language, info.language_probability


def query_ollama(prompt: str, language: str, language_name: str) -> str:
    """
    Send query to Ollama and get response in the same language.
    """
    print(f"🤖 Querying {OLLAMA_MODEL} via Ollama...")
    
    # System prompt to ensure response is in the same language
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
        return f"Error: Could not connect to Ollama at {OLLAMA_HOST}. Make sure Ollama is running."
    except Exception as e:
        return f"Error querying Ollama: {str(e)}"


async def text_to_speech(text: str, language: str, output_path: str) -> str:
    """
    Convert text to speech using Edge TTS.
    Returns the path to the generated audio file.
    """
    voice = get_voice_for_language(language)
    print(f"🔈 Generating speech with voice: {voice}")
    
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)
    
    return output_path


def get_language_name(code: str) -> str:
    """Get human-readable language name from code."""
    names = {
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
    return names.get(code, "Hindi")


async def process_voice_query(audio_path: str, output_dir: str = "/app/output") -> dict:
    """
    Full pipeline: Audio -> Transcription -> LLM -> Voice Response
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Transcribe audio
    print("\n" + "="*50)
    print("📝 STEP 1: Transcribing audio...")
    print("="*50)
    
    transcription, language, confidence = transcribe_audio(audio_path)
    language_name = get_language_name(language)
    
    print(f"\n✅ Detected Language: {language_name} ({language})")
    print(f"✅ Confidence: {confidence:.2%}")
    print(f"✅ Transcription: {transcription}\n")
    
    # Step 2: Query LLM
    print("="*50)
    print("🧠 STEP 2: Processing with Qwen...")
    print("="*50)
    
    llm_response = query_ollama(transcription, language, language_name)
    print(f"\n✅ LLM Response: {llm_response}\n")
    
    # Step 3: Generate voice response
    print("="*50)
    print("🎵 STEP 3: Generating voice response...")
    print("="*50)
    
    # Generate unique output filename
    input_name = Path(audio_path).stem
    output_path = os.path.join(output_dir, f"{input_name}_response.mp3")
    
    await text_to_speech(llm_response, language, output_path)
    print(f"\n✅ Voice response saved to: {output_path}\n")
    
    # Summary
    print("="*50)
    print("📊 SUMMARY")
    print("="*50)
    result = {
        "input_audio": audio_path,
        "detected_language": language,
        "language_name": language_name,
        "confidence": confidence,
        "transcription": transcription,
        "llm_response": llm_response,
        "output_audio": output_path
    }
    
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: python voice_assistant.py <audio_file> [output_dir]")
        print("\nSupported Indian Languages:")
        for code, name in [("hi", "Hindi"), ("ta", "Tamil"), ("te", "Telugu"), 
                           ("bn", "Bengali"), ("mr", "Marathi"), ("gu", "Gujarati"),
                           ("kn", "Kannada"), ("ml", "Malayalam"), ("pa", "Punjabi"), ("or", "Odia")]:
            print(f"  - {name} ({code})")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "/app/output"
    
    if not os.path.exists(audio_file):
        print(f"Error: Audio file not found: {audio_file}")
        sys.exit(1)
    
    # Run the async pipeline
    asyncio.run(process_voice_query(audio_file, output_dir))


if __name__ == "__main__":
    main()
