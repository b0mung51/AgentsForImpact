import numpy as np
import sounddevice as sd


def get_audio_devices():
    """Return available audio input/output devices."""
    return sd.query_devices()


def record_audio(duration: float, sample_rate: int = 16000) -> np.ndarray:
    """Record mono audio from default microphone and return int16 samples."""
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
    )
    sd.wait()
    return audio.flatten()


def play_audio(audio_data: np.ndarray, sample_rate: int = 16000) -> None:
    """Play audio samples through default speaker and block until finished."""
    sd.play(audio_data, samplerate=sample_rate)
    sd.wait()


if __name__ == "__main__":
    print("Listing devices...")
    devices = get_audio_devices()
    print(devices)
    if len(devices) == 0:
        raise RuntimeError(
            "No audio devices found. Check microphone permissions and default input/output settings on macOS."
        )
    print("Recording 3 seconds. Speak now.")
    data = record_audio(3.0, sample_rate=16000)
    print(f"Captured {len(data)} samples. Playing back...")
    play_audio(data, sample_rate=16000)
    print("Done.")
