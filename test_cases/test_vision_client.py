from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any
from unittest import TestCase
from unittest.mock import patch

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from vision_client import (
    VisionImage,
    call_vision_model,
    create_vision_client,
    iter_vision_model_stream_chunks,
)


class VisionClientTests(TestCase):
    def test_call_vision_model_builds_data_url_message(self) -> None:
        client = RecordingVisionClient("视觉回答")

        result = call_vision_model(
            client,
            "system prompt",
            "请看这张图",
            [VisionImage(bytes_data=b"\x89PNG\r\n\x1a\nimage", mime="image/png")],
        )

        self.assertEqual(result, "视觉回答")
        call = client.calls[0]
        self.assertEqual(call["temperature"], 0.3)
        self.assertEqual(call["model"], "glm-5v-turbo")
        self.assertEqual(call["messages"][0], {"role": "system", "content": "system prompt"})
        user_content = call["messages"][1]["content"]
        self.assertEqual(user_content[0]["type"], "image_url")
        self.assertTrue(
            user_content[0]["image_url"]["url"].startswith("data:image/png;base64,")
        )
        self.assertEqual(user_content[1], {"type": "text", "text": "请看这张图"})

    def test_call_vision_model_keeps_multiple_images_before_text(self) -> None:
        client = RecordingVisionClient("ok")
        images = [
            VisionImage(bytes_data=b"one", mime="image/png"),
            VisionImage(bytes_data=b"two", mime="image/jpeg"),
        ]

        call_vision_model(client, "system", "question", images)

        user_content = client.calls[0]["messages"][1]["content"]
        self.assertEqual([item["type"] for item in user_content], ["image_url", "image_url", "text"])
        self.assertTrue(user_content[0]["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertTrue(user_content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
        self.assertEqual(user_content[2]["text"], "question")

    def test_call_vision_model_returns_message_content(self) -> None:
        client = RecordingVisionClient("  模型输出  ")

        result = call_vision_model(client, "system", "question", [])

        self.assertEqual(result, "模型输出")

    def test_iter_vision_model_stream_chunks_uses_streaming_request(self) -> None:
        client = StreamingVisionClient(["第一段", "第二段"])

        chunks = list(
            iter_vision_model_stream_chunks(
                client,
                "system prompt",
                "请看这张图",
                [VisionImage(bytes_data=b"one", mime="image/png")],
                temperature=0.2,
            )
        )

        self.assertEqual(chunks, ["第一段", "第二段"])
        call = client.calls[0]
        self.assertTrue(call["stream"])
        self.assertEqual(call["temperature"], 0.2)
        user_content = call["messages"][1]["content"]
        self.assertEqual([item["type"] for item in user_content], ["image_url", "text"])
        self.assertTrue(user_content[0]["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertEqual(user_content[1], {"type": "text", "text": "请看这张图"})
        self.assertTrue(client.last_response.closed)
    def test_create_vision_client_reads_zhipu_env(self) -> None:
        with patch("vision_client.load_dotenv"):
            with patch("vision_client.OpenAI") as openai_cls:
                with patch.dict(
                    os.environ,
                    {
                        "ZHIPU_API_KEY": "test-key",
                        "ZHIPU_BASE_URL": "https://example.test/v4",
                    },
                    clear=True,
                ):
                    client = create_vision_client()

        openai_cls.assert_called_once_with(
            api_key="test-key",
            base_url="https://example.test/v4",
        )
        self.assertEqual(client, openai_cls.return_value)

    def test_create_vision_client_requires_api_key(self) -> None:
        with patch("vision_client.load_dotenv"):
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(RuntimeError):
                    create_vision_client()


class RecordingVisionClient:
    def __init__(self, content: str) -> None:
        self.calls: list[dict[str, Any]] = []
        self.chat = RecordingVisionChat(self, content)


class RecordingVisionChat:
    def __init__(self, owner: RecordingVisionClient, content: str) -> None:
        self.completions = RecordingVisionCompletions(owner, content)


class RecordingVisionCompletions:
    def __init__(self, owner: RecordingVisionClient, content: str) -> None:
        self.owner = owner
        self.content = content

    def create(self, **kwargs: Any) -> "RecordingVisionResponse":
        self.owner.calls.append(kwargs)
        return RecordingVisionResponse(self.content)


class RecordingVisionResponse:
    def __init__(self, content: str) -> None:
        self.choices = [RecordingVisionChoice(content)]


class RecordingVisionChoice:
    def __init__(self, content: str) -> None:
        self.message = RecordingVisionMessage(content)


class RecordingVisionMessage:
    def __init__(self, content: str) -> None:
        self.content = content

class StreamingVisionClient:
    def __init__(self, chunks: list[str]) -> None:
        self.calls: list[dict[str, Any]] = []
        self.last_response: StreamingVisionResponse | None = None
        self.chat = StreamingVisionChat(self, chunks)


class StreamingVisionChat:
    def __init__(self, owner: StreamingVisionClient, chunks: list[str]) -> None:
        self.completions = StreamingVisionCompletions(owner, chunks)


class StreamingVisionCompletions:
    def __init__(self, owner: StreamingVisionClient, chunks: list[str]) -> None:
        self.owner = owner
        self.chunks = chunks

    def create(self, **kwargs: Any) -> "StreamingVisionResponse":
        self.owner.calls.append(kwargs)
        response = StreamingVisionResponse(self.chunks)
        self.owner.last_response = response
        return response


class StreamingVisionResponse:
    def __init__(self, chunks: list[str]) -> None:
        self.chunks = chunks
        self.closed = False

    def __iter__(self):
        for chunk in self.chunks:
            yield StreamingVisionChunk(chunk)

    def close(self) -> None:
        self.closed = True


class StreamingVisionChunk:
    def __init__(self, content: str) -> None:
        self.choices = [StreamingVisionChoice(content)]


class StreamingVisionChoice:
    def __init__(self, content: str) -> None:
        self.delta = StreamingVisionDelta(content)


class StreamingVisionDelta:
    def __init__(self, content: str) -> None:
        self.content = content
