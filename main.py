import argparse

import config
from orchestrator import Orchestrator


def main():
    parser = argparse.ArgumentParser(description="Third Eye - AI Vision Assistant")
    parser.add_argument("--no-riva", action="store_true", help="Use fallback ASR/TTS")
    parser.add_argument(
        "--text",
        help="Run one text query and exit (useful for debugging without mic).",
    )
    args = parser.parse_args()

    if args.no_riva:
        config.USE_RIVA_ASR = False
        config.USE_RIVA_TTS = False
        print("Using fallback speech engines.")

    app = Orchestrator()

    if args.text:
        response = app.run_once_text(args.text)
        print(response)
        return

    app.run()


if __name__ == "__main__":
    main()
