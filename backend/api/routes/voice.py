"""
Voice routes — Azure AI Speech STT + TTS.

POST /api/voice/transcribe
  - Accepts audio bytes (WAV/OGG)
  - Uses Azure SpeechRecognizer (STT)
  - Returns { transcript: str }

POST /api/voice/synthesize
  - Accepts { text: str }
  - Uses Azure SpeechSynthesizer (TTS) with SSML for natural prosody
  - Returns audio/wav stream

Azure Speech SDK endpoint (implicit in SDK, region-based):
  wss://{region}.stt.speech.microsoft.com/...
  https://{region}.tts.speech.microsoft.com/...
"""
import azure.cognitiveservices.speech as speechsdk
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import io
import asyncio

from core.config import settings

router = APIRouter(prefix="/api/voice", tags=["voice"])


def _speech_config() -> speechsdk.SpeechConfig:
    """Build a shared SpeechConfig from settings."""
    config = speechsdk.SpeechConfig(
        subscription=settings.azure_speech_key,
        region=settings.azure_speech_region,
    )
    config.speech_recognition_language = "en-US"
    config.speech_synthesis_voice_name = "en-US-JennyNeural"  # natural neural voice
    return config


@router.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Convert uploaded audio to text using Azure Speech STT.

    Accepts WAV audio (from browser MediaRecorder API).
    Returns { transcript: str, confidence: float }
    """
    audio_bytes = await file.read()

    speech_config = _speech_config()

    import tempfile, os

    # Azure Speech SDK on this version doesn't support webm container.
    # We use a compressed stream with ANY so the SDK attempts auto-detection,
    # but Chrome webm may still fail. The reliable path is OGG_OPUS — the
    # frontend is instructed to send ogg/opus when available.
    content_type = file.content_type or ""
    if "ogg" in content_type or "opus" in content_type:
        audio_format = speechsdk.audio.AudioStreamFormat(
            compressed_stream_format=speechsdk.audio.AudioStreamContainerFormat.OGG_OPUS
        )
        audio_stream = speechsdk.audio.PushAudioInputStream(stream_format=audio_format)
        audio_stream.write(audio_bytes)
        audio_stream.close()
        audio_config = speechsdk.audio.AudioConfig(stream=audio_stream)
        tmp_path = None
    else:
        # WAV or unknown — write to temp file, SDK reads header to detect format
        suffix = ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        audio_config = speechsdk.audio.AudioConfig(filename=tmp_path)
    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        audio_config=audio_config,
    )

    # Run STT synchronously in a thread pool to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    result: speechsdk.SpeechRecognitionResult = await loop.run_in_executor(
        None, recognizer.recognize_once
    )

    # Clean up temp file if one was created
    if tmp_path:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return JSONResponse({
            "transcript": result.text,
            "confidence": 1.0,
        })
    elif result.reason == speechsdk.ResultReason.NoMatch:
        raise HTTPException(status_code=422, detail="Speech not recognized. Please speak clearly.")
    else:
        raise HTTPException(status_code=500, detail=f"Speech recognition failed: {result.reason}")


class TTSRequest(BaseModel):
    text: str
    voice: str = "en-US-JennyNeural"   # caller can override


@router.post("/synthesize")
async def synthesize_speech(request: TTSRequest):
    """
    Convert text to speech using Azure Neural TTS with SSML.

    SSML enables:
      - Natural prosody (pauses, emphasis)
      - Speed control (rate="medium")
      - Pitch variation

    Returns audio/wav binary stream.
    """
    speech_config = _speech_config()
    speech_config.speech_synthesis_voice_name = request.voice

    # Use SSML for natural-sounding speech
    ssml = f"""<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis'
                      xml:lang='en-US'>
        <voice name='{request.voice}'>
            <prosody rate='medium' pitch='default'>
                {_escape_xml(request.text)}
            </prosody>
        </voice>
    </speak>"""

    # Synthesize to in-memory buffer
    audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=False)
    synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config,
        audio_config=None,   # None = return bytes, not play to speaker
    )

    loop = asyncio.get_event_loop()
    result: speechsdk.SpeechSynthesisResult = await loop.run_in_executor(
        None, lambda: synthesizer.speak_ssml_async(ssml).get()
    )

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        audio_data = result.audio_data
        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type="audio/wav",
            headers={"Content-Disposition": "inline; filename=response.wav"},
        )
    else:
        raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {result.reason}")


def _escape_xml(text: str) -> str:
    """Escape special XML characters for SSML safety."""
    return (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )
