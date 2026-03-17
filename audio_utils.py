"""
Low-level audio I/O utilities using sounddevice.
Handles microphone recording and speaker playback.
"""
import numpy as np
import sounddevice as sd


def record_audio(duration: float, sample_rate: int = 16000) -> np.ndarray:
    """
    Record audio from the default microphone.

    Args:
        duration: Recording duration in seconds.
        sample_rate: Sample rate in Hz (default 16000 for speech).

    Returns:
        NumPy array of int16 audio samples.
    """
    print(f"Recording for {duration}s...")
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='int16'
    )
    sd.wait()
    return audio.flatten()


def play_audio(audio_data: np.ndarray, sample_rate: int = 44100) -> None:
    """
    Play audio through the default speakers. Blocks until complete.

    Args:
        audio_data: NumPy array of audio samples.
        sample_rate: Sample rate in Hz.
    """
    sd.play(audio_data, samplerate=sample_rate)
    sd.wait()


def get_audio_devices():
    """List available audio input/output devices (for debugging)."""
    return sd.query_devices()
