import argparse

import numpy as np
import sounddevice as sd

import config
from audio_utils import record_audio

_asr_service = None
_tts_service = None


def _init_riva_asr():
    global _asr_service
    if _asr_service is not None:
        return _asr_service

    import riva.client

    auth = riva.client.Auth(
        ssl_cert=None,
        use_ssl=True,
        uri=config.RIVA_URI,
        metadata_args=[
            ("function-id", config.RIVA_ASR_FUNCTION_ID),
            ("authorization", f"Bearer {config.NVIDIA_API_KEY}"),
        ],
    )
    _asr_service = riva.client.ASRService(auth)
    return _asr_service


def _init_riva_tts():
    global _tts_service
    if _tts_service is not None:
        return _tts_service

    import riva.client

    auth = riva.client.Auth(
        ssl_cert=None,
        use_ssl=True,
        uri=config.RIVA_URI,
        metadata_args=[
            ("function-id", config.RIVA_TTS_FUNCTION_ID),
            ("authorization", f"Bearer {config.NVIDIA_API_KEY}"),
        ],
    )
    _tts_service = riva.client.SpeechSynthesisService(auth)
    return _tts_service


def _listen_riva(max_seconds: int = 8) -> str:
    import riva.client

    asr = _init_riva_asr()
    print(f"Listening (Riva) for up to {max_seconds}s...")
    audio = record_audio(max_seconds, sample_rate=config.RIVA_ASR_SAMPLE_RATE)
    audio_bytes = audio.tobytes()

    rec_config = riva.client.RecognitionConfig(
        encoding=riva.client.AudioEncoding.LINEAR_PCM,
        language_code=config.RIVA_ASR_LANGUAGE,
        max_alternatives=1,
        enable_automatic_punctuation=True,
        sample_rate_hertz=config.RIVA_ASR_SAMPLE_RATE,
        audio_channel_count=1,
    )
    response = asr.offline_recognize(audio_bytes, rec_config)
    if not response.results:
        return ""
    return response.results[0].alternatives[0].transcript.strip()


def _listen_fallback(max_seconds: int = 8) -> str:
    import speech_recognition as sr

    print(f"Listening (fallback) for up to {max_seconds}s...")
    try:
        audio_np = record_audio(max_seconds, sample_rate=16000)
    except Exception as e:
        print(f"Fallback microphone capture failed: {e}")
        return ""
    audio = sr.AudioData(audio_np.tobytes(), sample_rate=16000, sample_width=2)
    recognizer = sr.Recognizer()
    try:
        return recognizer.recognize_google(audio).strip()
    except Exception:
        return ""


def listen(max_seconds: int = 8) -> str:
    if config.USE_RIVA_ASR:
        try:
            return _listen_riva(max_seconds=max_seconds)
        except Exception as e:
            print(f"Riva ASR failed, using fallback: {e}")
    return _listen_fallback(max_seconds=max_seconds)


def _speak_riva(text: str) -> None:
    import riva.client

    tts = _init_riva_tts()
    response = tts.synthesize(
        text=text,
        voice_name=config.RIVA_TTS_VOICE,
        language_code=config.RIVA_TTS_LANGUAGE,
        encoding=riva.client.AudioEncoding.LINEAR_PCM,
        sample_rate_hz=config.RIVA_TTS_SAMPLE_RATE,
    )
    audio = np.frombuffer(response.audio, dtype=np.int16)
    sd.play(audio, samplerate=config.RIVA_TTS_SAMPLE_RATE)
    sd.wait()


def _speak_fallback(text: str) -> None:
    import pyttsx3

    engine = pyttsx3.init()
    engine.setProperty("rate", 180)
    engine.say(text)
    engine.runAndWait()


def speak(text: str) -> None:
    if not text or not text.strip():
        return
    if config.USE_RIVA_TTS:
        try:
            _speak_riva(text)
            return
        except Exception as e:
            print(f"Riva TTS failed, using fallback: {e}")
    _speak_fallback(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speech I/O smoke tests")
    parser.add_argument("--no-riva", action="store_true", help="Force fallback ASR/TTS")
    parser.add_argument("--tts", default="Third Eye speech test", help="Text to speak")
    parser.add_argument("--asr", action="store_true", help="Run speech-to-text test")
    args = parser.parse_args()

    if args.no_riva:
        config.USE_RIVA_ASR = False
        config.USE_RIVA_TTS = False

    print("Running TTS test...")
    speak(args.tts)
    print("TTS test done.")

    if args.asr:
        text = listen()
        print(f"ASR transcript: {text}")
