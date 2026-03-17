import base64

import cv2
from openai import OpenAI

from config import (
    NVIDIA_API_KEY,
    VISION_BASE_URL,
    VISION_MAX_TOKENS,
    VISION_MODEL,
    VISION_TEMPERATURE,
)
from prompts import (
    VISION_DESCRIBE_PROMPT,
    VISION_FOCUSED_PROMPT_TEMPLATE,
    VISION_READ_TEXT_PROMPT,
)

WEBCAM_DEVICE_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
JPEG_QUALITY = 80

_camera = None
_vision_client = OpenAI(base_url=VISION_BASE_URL, api_key=NVIDIA_API_KEY)


def _get_camera() -> cv2.VideoCapture:
    global _camera
    if _camera is None or not _camera.isOpened():
        # AVFoundation backend is the most reliable on macOS.
        _camera = cv2.VideoCapture(WEBCAM_DEVICE_INDEX, cv2.CAP_AVFOUNDATION)
        _camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        _camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        if not _camera.isOpened():
            raise RuntimeError(
                "Could not open webcam. Check System Settings > Privacy & Security > Camera."
            )
    return _camera


def capture_frame() -> bytes:
    camera = _get_camera()
    ok, frame = camera.read()
    if not ok:
        raise RuntimeError("Failed to capture frame from webcam.")

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("Failed to encode frame as JPEG.")
    return jpeg.tobytes()


def save_frame(path: str = "capture_test.jpg") -> str:
    frame_bytes = capture_frame()
    with open(path, "wb") as f:
        f.write(frame_bytes)
    return path


def analyze_frame(jpeg_bytes: bytes, prompt: str) -> str:
    b64_image = base64.b64encode(jpeg_bytes).decode("utf-8")
    response = _vision_client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                    },
                ],
            }
        ],
        max_tokens=VISION_MAX_TOKENS,
        temperature=VISION_TEMPERATURE,
    )
    return (response.choices[0].message.content or "").strip()


def capture_and_describe(focus: str | None = None) -> str:
    jpeg_bytes = capture_frame()
    prompt = (
        VISION_FOCUSED_PROMPT_TEMPLATE.format(focus=focus)
        if focus
        else VISION_DESCRIBE_PROMPT
    )
    return analyze_frame(jpeg_bytes, prompt)


def read_text() -> str:
    jpeg_bytes = capture_frame()
    return analyze_frame(jpeg_bytes, VISION_READ_TEXT_PROMPT)


def release_camera() -> None:
    global _camera
    if _camera is not None:
        _camera.release()
        _camera = None


if __name__ == "__main__":
    try:
        out = save_frame("capture_test.jpg")
        print(f"Saved webcam frame to {out}")
    finally:
        release_camera()
