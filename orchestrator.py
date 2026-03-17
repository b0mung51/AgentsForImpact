import threading
import time
from enum import Enum

from agent import call_nemotron_plain, create_initial_history, run_agentic_loop
from config import CONTINUOUS_MODE_INTERVAL
from navigation import get_directions
from prompts import CONTINUOUS_MODE_PROMPT_TEMPLATE
from speech import listen, speak
from vision import capture_and_describe, read_text, release_camera


class AppState(Enum):
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


class Orchestrator:
    def __init__(self):
        self.state = AppState.LISTENING
        self.running = True
        self.conversation_history = create_initial_history()
        self.continuous_mode = False
        self.continuous_interval = CONTINUOUS_MODE_INTERVAL
        self.last_scene_description = None
        self.last_scene_time = None
        self.continuous_thread = None
        self.tool_handlers = {
            "capture_and_describe": capture_and_describe,
            "read_text": read_text,
            "get_directions": get_directions,
            "toggle_continuous_mode": self._handle_toggle_continuous_mode,
        }

    def _handle_toggle_continuous_mode(
        self, enabled: bool, interval_seconds: float | None = None
    ) -> str:
        if interval_seconds and interval_seconds > 0:
            self.continuous_interval = interval_seconds

        if enabled and not self.continuous_mode:
            self.continuous_mode = True
            self._start_continuous_thread()
            return (
                f"Continuous mode enabled. I'll check surroundings every "
                f"{self.continuous_interval} seconds."
            )
        if not enabled and self.continuous_mode:
            self.continuous_mode = False
            return "Continuous mode disabled."
        if enabled:
            return "Continuous mode is already enabled."
        return "Continuous mode is already disabled."

    def _start_continuous_thread(self):
        if self.continuous_thread and self.continuous_thread.is_alive():
            return

        def _loop():
            while self.running and self.continuous_mode:
                try:
                    if self.state == AppState.LISTENING:
                        self._process_continuous_update()
                except Exception as e:
                    print(f"Continuous mode error: {e}")
                time.sleep(self.continuous_interval)

        self.continuous_thread = threading.Thread(target=_loop, daemon=True)
        self.continuous_thread.start()

    def _process_continuous_update(self):
        current_description = capture_and_describe()
        now = time.time()

        if self.last_scene_description:
            seconds_ago = int(now - (self.last_scene_time or now))
            prompt = CONTINUOUS_MODE_PROMPT_TEMPLATE.format(
                seconds_ago=seconds_ago,
                previous_description=self.last_scene_description,
                current_description=current_description,
            )
            temp_history = create_initial_history()
            temp_history.append({"role": "user", "content": prompt})
            decision = call_nemotron_plain(temp_history).strip()
            if decision and decision != "NO_UPDATE":
                self.state = AppState.SPEAKING
                speak(decision)
                self.state = AppState.LISTENING
                self.conversation_history.append(
                    {"role": "assistant", "content": f"[Continuous alert] {decision}"}
                )

        self.last_scene_description = current_description
        self.last_scene_time = now

    def handle_text(self, user_text: str) -> str:
        self.state = AppState.THINKING
        self.conversation_history.append({"role": "user", "content": user_text})
        response = run_agentic_loop(self.conversation_history, self.tool_handlers)
        return response

    def run_once_text(self, user_text: str) -> str:
        response = self.handle_text(user_text)
        self.state = AppState.SPEAKING
        speak(response)
        self.state = AppState.LISTENING
        return response

    def run(self):
        speak("Third Eye is ready. How can I help you?")
        while self.running:
            try:
                self.state = AppState.LISTENING
                user_text = listen().strip()
                if not user_text:
                    continue

                if user_text.lower() in {"exit", "quit", "stop", "goodbye", "bye"}:
                    speak("Goodbye. Stay safe.")
                    self.running = False
                    break

                response = self.handle_text(user_text)

                self.state = AppState.SPEAKING
                speak(response)

            except KeyboardInterrupt:
                self.running = False
                break
            except Exception as e:
                print(f"Orchestrator error: {e}")
                try:
                    speak("I hit an error. Please try again.")
                except Exception:
                    pass
            finally:
                self.state = AppState.LISTENING

        self.continuous_mode = False
        if self.continuous_thread and self.continuous_thread.is_alive():
            self.continuous_thread.join(timeout=1.0)
        release_camera()
