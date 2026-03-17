# Third Eye

A voice-only macOS desktop app for visually impaired users. Uses the webcam as a surrogate eye — speak to an AI agent that captures and interprets the visual scene, describes surroundings, reads text/signs, and gives walking directions, all through voice.

Built for the **NVIDIA AI Agent Hackathon**.

---

## Prerequisites

- macOS (required for camera/mic/speaker support via AVFoundation and PyObjC)
- Python 3.11+
- An NVIDIA API key from [build.nvidia.com](https://build.nvidia.com)

Install system dependencies:

```bash
brew install portaudio
xcode-select --install
```

---

## Setup

**1. Clone and enter the repo:**

```bash
git clone <repo-url>
cd AgentsForImpact
```

**2. Create and activate a virtual environment:**

```bash
python -m venv venv
source venv/bin/activate
```

> **Note:** If you have conda installed, `python` may still point to the conda environment even with the venv active. Always use `venv/bin/python` to run scripts directly (see Running section below), or deactivate conda first with `conda deactivate`.

**3. Install Python dependencies:**

```bash
venv/bin/pip install -r requirements.txt
```

**4. Create your `.env` file:**

```bash
cp .env.example .env
```

Then edit `.env` and add your key:

```
NVIDIA_API_KEY=nvapi-xxx
```

**5. Grant camera and microphone permissions:**

Go to **System Settings > Privacy & Security** and enable Camera and Microphone access for your terminal app.

---

## Running

> Always use `venv/bin/python` instead of `python` to ensure the venv is used.

**Normal mode** (uses Riva ASR/TTS via NVIDIA NIM):

```bash
venv/bin/python main.py
```

**Fallback mode** (uses local SpeechRecognition + pyttsx3, no Riva needed — use this if Riva function IDs aren't available on your account):

```bash
venv/bin/python main.py --no-riva
```

---

## Testing

**Test vision pipeline** (captures a webcam frame and sends it to Llama 3.2 Vision):

```bash
venv/bin/python test_vision.py
```

**Run all tests:**

```bash
venv/bin/python -m pytest tests/
```

**Run a specific test file:**

```bash
venv/bin/python -m pytest tests/test_vision.py
```

---

## Architecture

```
LISTENING -> THINKING -> SPEAKING -> LISTENING
```

| Module | Role |
|---|---|
| `orchestrator.py` | Central coordinator, manages state machine and conversation history |
| `agent.py` | Nemotron 120B reasoning agent via NVIDIA NIM (tool calling) |
| `vision.py` | Webcam capture (OpenCV) + Llama 3.2 90B Vision scene analysis |
| `speech.py` | Riva ASR (streaming) + Riva TTS, with local fallbacks |
| `navigation.py` | Apple MapKit via PyObjC, with Google Maps fallback |
| `config.py` | All constants and API config, loaded from `.env` |
| `prompts.py` | System and vision prompt strings |
