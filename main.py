"""
Third Eye - Entry point.
An AI navigation and surroundings assistant for visually impaired people.

Usage:
    python main.py                  # Normal mode (Riva speech)
    python main.py --no-riva        # Use fallback ASR/TTS (no Riva)
"""
import argparse
import config
from orchestrator import Orchestrator


def main():
    parser = argparse.ArgumentParser(description="Third Eye - AI Vision Assistant")
    parser.add_argument(
        "--no-riva",
        action="store_true",
        help="Use fallback speech engines instead of NVIDIA Riva"
    )
    args = parser.parse_args()

    if args.no_riva:
        config.USE_RIVA_ASR = False
        config.USE_RIVA_TTS = False
        print("Using fallback speech engines (SpeechRecognition + pyttsx3)")

    print("=" * 50)
    print("  THIRD EYE - AI Vision Assistant")
    print("  Speak to navigate. Say 'exit' to quit.")
    print("=" * 50)
    print()

    orchestrator = Orchestrator()
    orchestrator.run()


if __name__ == "__main__":
    main()
