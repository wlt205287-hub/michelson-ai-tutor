from __future__ import annotations

import base64
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


BASE_DIR = Path(__file__).resolve().parent
DEMO_DIR = BASE_DIR / "agent_demo"
DEFAULT_ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
DEFAULT_ZHIPU_MODEL_NAME = "glm-5v-turbo"


@dataclass(frozen=True)
class VisionImage:
    bytes_data: bytes
    mime: str


def build_vision_user_content(
    images: list[VisionImage],
    user_text: str,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for image in images:
        encoded = base64.b64encode(image.bytes_data).decode("ascii")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{image.mime};base64,{encoded}"},
            }
        )
    content.append({"type": "text", "text": user_text})
    return content


def close_model_stream(response: Any) -> None:
    close = getattr(response, "close", None)
    if callable(close):
        close()


def create_vision_client() -> OpenAI:
    """Create the OpenAI-compatible GLM vision client from environment settings."""
    load_dotenv(DEMO_DIR / ".env")
    load_dotenv(BASE_DIR / ".env")

    api_key = os.getenv("ZHIPU_API_KEY")
    if not api_key:
        raise RuntimeError("ZHIPU_API_KEY is not configured.")

    base_url = os.getenv("ZHIPU_BASE_URL", DEFAULT_ZHIPU_BASE_URL)
    return OpenAI(api_key=api_key, base_url=base_url)


def call_vision_model(
    client: Any,
    system_prompt: str,
    user_text: str,
    images: list[VisionImage],
    temperature: float = 0.3,
) -> str:
    """Call a GLM vision model with image data URLs followed by text."""
    response = client.chat.completions.create(
        model=os.getenv("ZHIPU_MODEL_NAME", DEFAULT_ZHIPU_MODEL_NAME),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_vision_user_content(images, user_text)},
        ],
        temperature=temperature,
    )
    return (response.choices[0].message.content or "").strip()


def iter_vision_model_stream_chunks(
    client: Any,
    system_prompt: str,
    user_text: str,
    images: list[VisionImage],
    temperature: float = 0.3,
) -> Iterator[str]:
    """Yield streaming chunks from a GLM vision model call."""
    response = client.chat.completions.create(
        model=os.getenv("ZHIPU_MODEL_NAME", DEFAULT_ZHIPU_MODEL_NAME),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_vision_user_content(images, user_text)},
        ],
        temperature=temperature,
        stream=True,
    )
    try:
        for chunk in response:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            content = getattr(delta, "content", None)
            if content:
                yield content
    finally:
        close_model_stream(response)
