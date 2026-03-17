import os
from dotenv import load_dotenv

load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

if not NVIDIA_API_KEY:
    raise ValueError(
        "NVIDIA_API_KEY not set. Create a .env file with NVIDIA_API_KEY=nvapi-xxx"
    )

# --- Riva ---
RIVA_URI = "grpc.nvcf.nvidia.com:443"
RIVA_ASR_FUNCTION_ID = "1598d209-5e27-4d3c-8079-4751568b1081"
RIVA_ASR_SAMPLE_RATE = 16000
RIVA_ASR_LANGUAGE = "en-US"

RIVA_TTS_FUNCTION_ID = "0149dedb-2be8-4195-b9a0-e57e0e14f972"
RIVA_TTS_VOICE = "English-US.Female-1"
RIVA_TTS_SAMPLE_RATE = 44100
RIVA_TTS_LANGUAGE = "en-US"

# --- Feature flags ---
USE_RIVA_ASR = True
USE_RIVA_TTS = True

# --- Nemotron (Reasoning Agent) ---
NEMOTRON_BASE_URL = "https://integrate.api.nvidia.com/v1"
NEMOTRON_MODEL = "nvidia/nemotron-3-super-120b-a12b"
NEMOTRON_MAX_TOKENS = 512
NEMOTRON_TEMPERATURE = 0.4

# --- Vision Model (NVIDIA endpoint) ---
VISION_BASE_URL = "https://integrate.api.nvidia.com/v1"
VISION_MODEL = "meta/llama-3.2-90b-vision-instruct"
VISION_MAX_TOKENS = 512
VISION_TEMPERATURE = 0.2

# --- Continuous mode ---
CONTINUOUS_MODE_INTERVAL = 5
