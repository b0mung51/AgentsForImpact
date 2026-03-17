import argparse
import json
from types import SimpleNamespace

from openai import OpenAI

from config import (
    NEMOTRON_BASE_URL,
    NEMOTRON_MAX_TOKENS,
    NEMOTRON_MODEL,
    NEMOTRON_TEMPERATURE,
    NVIDIA_API_KEY,
)
from navigation import get_directions
from prompts import SYSTEM_PROMPT
from vision import capture_and_describe, read_text

client = OpenAI(
    base_url=NEMOTRON_BASE_URL,
    api_key=NVIDIA_API_KEY,
)

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "capture_and_describe",
            "description": "Capture webcam image and describe surroundings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "focus": {
                        "type": "string",
                        "description": "Optional focus, for example: obstacles, people, traffic.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_text",
            "description": "Capture webcam image and read visible text.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_directions",
            "description": "Get walking directions to a destination.",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {"type": "string"},
                    "origin": {"type": "string"},
                },
                "required": ["destination"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "toggle_continuous_mode",
            "description": "Enable or disable continuous monitoring updates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean"},
                    "interval_seconds": {"type": "number"},
                },
                "required": ["enabled"],
            },
        },
    },
]


def create_initial_history() -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT}]


def call_nemotron_choice(conversation_history: list[dict]):
    response = client.chat.completions.create(
        model=NEMOTRON_MODEL,
        messages=conversation_history,
        tools=TOOL_DEFINITIONS,
        tool_choice="auto",
        temperature=NEMOTRON_TEMPERATURE,
        max_tokens=NEMOTRON_MAX_TOKENS,
    )
    return response.choices[0]


def call_nemotron(conversation_history: list[dict]) -> str:
    choice = call_nemotron_choice(conversation_history)
    return (choice.message.content or "").strip()


def call_nemotron_plain(conversation_history: list[dict]) -> str:
    response = client.chat.completions.create(
        model=NEMOTRON_MODEL,
        messages=conversation_history,
        temperature=NEMOTRON_TEMPERATURE,
        max_tokens=NEMOTRON_MAX_TOKENS,
    )
    return (response.choices[0].message.content or "").strip()


def run_agentic_loop(
    conversation_history: list[dict],
    tool_handlers: dict,
    model_caller=call_nemotron_choice,
    max_iterations: int = 6,
) -> str:
    for _ in range(max_iterations):
        choice = model_caller(conversation_history)
        tool_calls = getattr(choice.message, "tool_calls", None)

        if choice.finish_reason != "tool_calls" or not tool_calls:
            assistant_message = (choice.message.content or "").strip()
            conversation_history.append({"role": "assistant", "content": assistant_message})
            return assistant_message

        conversation_history.append(
            {
                "role": "assistant",
                "content": choice.message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            }
        )

        for tool_call in tool_calls:
            func_name = tool_call.function.name
            try:
                func_args = (
                    json.loads(tool_call.function.arguments)
                    if tool_call.function.arguments
                    else {}
                )
            except json.JSONDecodeError:
                func_args = {}

            if func_name in tool_handlers:
                try:
                    result = tool_handlers[func_name](**func_args)
                except Exception as e:
                    result = f"Error executing {func_name}: {e}"
            else:
                result = f"Unknown tool: {func_name}"

            conversation_history.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result),
                }
            )

    return "I hit a tool-calling safety limit. Please try again."


def _mock_model_caller_factory():
    state = {"count": 0}

    def _caller(_history):
        state["count"] += 1
        if state["count"] == 1:
            return SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="tool_1",
                            function=SimpleNamespace(
                                name="capture_and_describe",
                                arguments='{"focus":"obstacles"}',
                            ),
                        )
                    ],
                ),
            )
        return SimpleNamespace(
            finish_reason="stop",
            message=SimpleNamespace(
                content="I see a clear path ahead with a chair slightly to your right.",
                tool_calls=None,
            ),
        )

    return _caller


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nemotron agent tests")
    parser.add_argument("--live", action="store_true", help="Run live Nemotron call")
    args = parser.parse_args()

    history = create_initial_history()
    history.append({"role": "user", "content": "What's in front of me?"})

    mock_handlers = {
        "capture_and_describe": lambda focus=None: f"[mocked vision] focus={focus}",
        "read_text": lambda: "[mocked read_text] STOP sign ahead",
        "get_directions": lambda destination, origin=None: (
            f"[mocked directions] from {origin or 'current location'} to {destination}"
        ),
        "toggle_continuous_mode": (
            lambda enabled, interval_seconds=None: (
                f"[mocked continuous] enabled={enabled} interval={interval_seconds}"
            )
        ),
    }
    live_handlers = {
        "capture_and_describe": capture_and_describe,
        "read_text": read_text,
        "get_directions": get_directions,
        "toggle_continuous_mode": (
            lambda enabled, interval_seconds=None: (
                f"continuous mode set to {enabled} interval={interval_seconds}"
            )
        ),
    }

    if args.live:
        print(run_agentic_loop(history, live_handlers))
    else:
        mock_caller = _mock_model_caller_factory()
        print(run_agentic_loop(history, mock_handlers, model_caller=mock_caller))
