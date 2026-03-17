"""
Central orchestrator: manages the state machine, conversation history,
tool dispatch, continuous mode, and live camera feed with debug overlay.

State Machine:
  LISTENING -> THINKING -> SPEAKING -> LISTENING
"""
import cv2
import datetime
import numpy as np
import os
import threading
import time
import textwrap
from enum import Enum
from agent import run_agentic_loop, create_initial_history
from speech import listen, speak
from vision import capture_and_describe, read_text, release_camera, get_camera
from navigation import get_directions
from config import CONTINUOUS_MODE_INTERVAL
from prompts import CONTINUOUS_MODE_PROMPT_TEMPLATE

MAX_OVERLAY_LINES = 8
WRAP_WIDTH = 70


class AppState(Enum):
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


class Orchestrator:
    def __init__(self):
        self.state = AppState.LISTENING
        self.conversation_history = create_initial_history()
        self.continuous_mode = False
        self.continuous_timer = None
        self.continuous_interval = CONTINUOUS_MODE_INTERVAL
        self.last_scene_description = None
        self.last_scene_time = None
        self.current_nav_step = "No active navigation"
        self.running = True

        # Thread-safe overlay log buffer
        self._overlay_lines = []
        self._overlay_lock = threading.Lock()

        # File-based debug log
        os.makedirs("logs", exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_file = open(f"logs/debug_{ts}.log", "w")

        self.tool_handlers = {
            "capture_and_describe": self._logged_capture_and_describe,
            "read_text": self._logged_read_text,
            "get_directions": self._logged_get_directions,
            "toggle_continuous_mode": self._handle_toggle_continuous,
        }

    def _log(self, tag, msg):
        """Debug log to terminal and overlay buffer."""
        ts = time.strftime("%H:%M:%S")
        full = f"[{ts}] [{tag}] {msg}"
        print(full)
        self._log_file.write(full + "\n")
        self._log_file.flush()

        # Truncate long messages for the overlay
        short = msg if len(msg) <= WRAP_WIDTH else msg[:WRAP_WIDTH] + "..."
        overlay_line = f"[{tag}] {short}"

        with self._overlay_lock:
            self._overlay_lines.append(overlay_line)
            # Keep only the most recent lines
            if len(self._overlay_lines) > MAX_OVERLAY_LINES:
                self._overlay_lines = self._overlay_lines[-MAX_OVERLAY_LINES:]

    def _logged_capture_and_describe(self, **kwargs):
        self._log("TOOL", f"capture_and_describe({kwargs})")
        result = capture_and_describe(**kwargs)
        self._log("VISION", result)
        return result

    def _logged_read_text(self, **kwargs):
        self._log("TOOL", "read_text()")
        result = read_text(**kwargs)
        self._log("VISION", result)
        return result

    def _logged_get_directions(self, **kwargs):
        self._log("TOOL", f"get_directions({kwargs})")
        result = get_directions(**kwargs)
        self._log("NAV", result)
        return result

    def _handle_toggle_continuous(self, enabled: bool, interval_seconds: float = None) -> str:
        if interval_seconds:
            self.continuous_interval = interval_seconds

        if enabled and not self.continuous_mode:
            self.continuous_mode = True
            self._start_continuous_timer()
            return f"Continuous mode enabled. I'll check your surroundings every {self.continuous_interval} seconds."
        elif not enabled and self.continuous_mode:
            self.continuous_mode = False
            self._stop_continuous_timer()
            return "Continuous mode disabled."
        elif enabled:
            return "Continuous mode is already on."
        else:
            return "Continuous mode is already off."

    def _start_continuous_timer(self):
        if self.continuous_timer:
            self.continuous_timer.cancel()

        def _continuous_loop():
            while self.continuous_mode and self.running:
                if self.state == AppState.LISTENING:
                    self._process_continuous_update()
                time.sleep(self.continuous_interval)

        self.continuous_timer = threading.Thread(target=_continuous_loop, daemon=True)
        self.continuous_timer.start()

    def _stop_continuous_timer(self):
        self.continuous_mode = False

    def _process_continuous_update(self):
        try:
            current_description = capture_and_describe()
            self._log("CONTINUOUS", current_description)

            if self.last_scene_description:
                seconds_ago = int(time.time() - self.last_scene_time) if self.last_scene_time else self.continuous_interval

                update_prompt = CONTINUOUS_MODE_PROMPT_TEMPLATE.format(
                    seconds_ago=seconds_ago,
                    previous_description=self.last_scene_description,
                    current_description=current_description,
                    current_nav_step=self.current_nav_step,
                )

                temp_history = self.conversation_history.copy()
                temp_history.append({"role": "user", "content": update_prompt})

                response = run_agentic_loop(temp_history, self.tool_handlers)
                self._log("CONTINUOUS", f"Nemotron: {response}")

                if response.strip() != "NO_UPDATE":
                    self.state = AppState.SPEAKING
                    speak(response)
                    self.state = AppState.LISTENING
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": f"[Continuous mode alert] {response}"
                    })

            self.last_scene_description = current_description
            self.last_scene_time = time.time()

        except Exception as e:
            self._log("ERROR", f"Continuous mode: {e}")

    def _draw_overlay(self, frame):
        """Draw state label and log lines on the camera frame."""
        h, w = frame.shape[:2]

        # --- State label at top-left (green) ---
        state_text = f"State: {self.state.value.upper()}"
        cv2.putText(frame, state_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

        # --- Log overlay at bottom ---
        with self._overlay_lock:
            lines = list(self._overlay_lines)

        if not lines:
            return frame

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        line_height = 20
        padding = 8

        # Calculate overlay height
        overlay_h = len(lines) * line_height + padding * 2

        # Draw semi-transparent dark background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - overlay_h), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Draw each log line
        for i, line in enumerate(lines):
            y = h - overlay_h + padding + (i + 1) * line_height - 4
            cv2.putText(frame, line, (8, y), font, font_scale, (255, 255, 255), thickness)

        return frame

    def _speech_loop(self):
        """Background thread: listen → think → speak loop."""
        self._log("STATE", "Third Eye starting...")
        speak("Third Eye is ready. How can I help you?")

        while self.running:
            try:
                self.state = AppState.LISTENING
                self._log("STATE", "LISTENING — waiting for speech...")
                user_input = listen()

                if not user_input:
                    continue

                self._log("ASR", f"User said: \"{user_input}\"")

                if user_input.lower().strip() in ["exit", "quit", "stop", "goodbye", "bye"]:
                    speak("Goodbye! Stay safe.")
                    self.running = False
                    break

                self.state = AppState.THINKING
                self._log("STATE", "THINKING — sending to Nemotron...")
                self.conversation_history.append({
                    "role": "user",
                    "content": user_input
                })

                response = run_agentic_loop(
                    self.conversation_history,
                    self.tool_handlers
                )

                self._log("NEMOTRON", f"Response: {response}")

                self.state = AppState.SPEAKING
                self._log("STATE", "SPEAKING...")
                speak(response)

            except KeyboardInterrupt:
                self.running = False
            except Exception as e:
                self._log("ERROR", f"Main loop: {e}")
                speak("I encountered an error. Let me try again.")

    def run(self):
        """Main thread runs camera feed (required by macOS), speech loop in background."""
        # Speech loop in background thread
        speech_thread = threading.Thread(target=self._speech_loop, daemon=True)
        speech_thread.start()

        # Camera feed on main thread (macOS requires cv2.imshow on main thread)
        camera = get_camera()
        while self.running:
            ret, frame = camera.read()
            if ret:
                frame = self._draw_overlay(frame)
                cv2.imshow("Third Eye - Live Feed", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                self.running = False
                break

        cv2.destroyAllWindows()
        self._stop_continuous_timer()
        release_camera()
