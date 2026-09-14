from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest import TestCase
from unittest.mock import patch

from fastapi.testclient import TestClient
from pydantic import ValidationError


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import agent_core
import api_server
from quiz_schema import GeneratedQuizEnvelope


def valid_quiz_payload() -> dict[str, Any]:
    return {
        "quiz": {
            "topic": "连续扫描相对表面形貌重建",
            "questions": [
                {
                    "id": "q1",
                    "type": "choice",
                    "question": "参考镜连续移动的主要作用是什么？",
                    "options": ["引入时间载波", "改变激光波长", "提高相机分辨率", "产生白光"],
                    "correct_answer": 0,
                    "explanation": "连续移动参考镜会引入随时间变化的相位。",
                },
                {
                    "id": "q2",
                    "type": "choice",
                    "question": "时间序列 FFT 主要用于提取什么？",
                    "options": ["主频相位", "样品质量", "相机尺寸", "环境温度"],
                    "correct_answer": 0,
                    "explanation": "每个像素的时间序列可用于提取主频及其相位。",
                },
                {
                    "id": "q3",
                    "type": "choice",
                    "question": "反射镜移动 d 时，往返光程变化是多少？",
                    "options": ["d/2", "d", "2d", "4d"],
                    "correct_answer": 2,
                    "explanation": "反射光经历去程和回程，因此光程变化为 2d。",
                },
                {
                    "id": "q4",
                    "type": "short_answer",
                    "question": "相对高度与相对相位之间有什么关系？",
                    "reference_answer": "反射式测量中可按 h = λΦ_rel/(4π) 换算相对高度。",
                    "key_points": ["反射式双程光路", "相对相位", "波长与4π换算"],
                },
                {
                    "id": "q5",
                    "type": "short_answer",
                    "question": "为什么需要对包裹相位进行二维相位展开？",
                    "reference_answer": "消除相位中的 2π 跳变，恢复空间连续的相对相位。",
                    "key_points": ["2π跳变", "空间连续性", "相对相位"],
                },
            ],
        }
    }


def test_resources() -> dict[str, Any]:
    return {
        "knowledge_base": "连续扫描实验知识库",
        "quiz_prompt": (
            "QUIZ_PROMPT_MARKER\n{knowledge_base}\n{focus_instruction}\n"
            "{student_request}\n{validation_feedback}"
        ),
    }


class SequenceModelClient:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.calls = 0
        self.prompts: list[str] = []
        self.system_prompts: list[str] = []
        self.chat = SequenceChat(self)


class SequenceChat:
    def __init__(self, owner: SequenceModelClient) -> None:
        self.completions = SequenceCompletions(owner)


class SequenceCompletions:
    def __init__(self, owner: SequenceModelClient) -> None:
        self.owner = owner

    def create(self, **kwargs: Any) -> "SequenceResponse":
        self.owner.calls += 1
        self.owner.system_prompts.append(kwargs["messages"][0]["content"])
        self.owner.prompts.append(kwargs["messages"][1]["content"])
        output = self.owner.outputs.pop(0)
        return SequenceResponse(output)


class SequenceResponse:
    def __init__(self, content: str) -> None:
        self.choices = [SequenceChoice(content)]


class SequenceChoice:
    def __init__(self, content: str) -> None:
        self.message = SequenceMessage(content)


class SequenceMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class QuizSchemaTests(TestCase):
    def test_valid_quiz_matches_required_distribution(self) -> None:
        quiz = GeneratedQuizEnvelope.model_validate(valid_quiz_payload())

        self.assertEqual(len(quiz.quiz.questions), 5)
        self.assertEqual(
            [question.type for question in quiz.quiz.questions],
            ["choice", "choice", "choice", "short_answer", "short_answer"],
        )

    def test_rejects_invalid_choice_options_and_distribution(self) -> None:
        payload = valid_quiz_payload()
        payload["quiz"]["questions"][0]["options"] = ["重复", "重复", "C", "D"]

        with self.assertRaises(ValidationError):
            GeneratedQuizEnvelope.model_validate(payload)

    def test_rejects_removed_fill_blank_type(self) -> None:
        payload = valid_quiz_payload()
        payload["quiz"]["questions"][2] = {
            "id": "q3",
            "type": "fill_blank",
            "question": "反射镜移动 d 时，往返光程变化为 ____。",
            "correct_answer": "2d",
            "accept_variants": ["2*d"],
            "explanation": "反射光经历去程和回程。",
        }

        with self.assertRaises(ValidationError):
            GeneratedQuizEnvelope.model_validate(payload)


class QuizGenerationTests(TestCase):
    def test_quiz_detection_requires_pre_lab_and_explicit_intent(self) -> None:
        self.assertTrue(agent_core.is_quiz_generation_request("请生成预习题", "pre_lab"))
        self.assertFalse(agent_core.is_quiz_generation_request("请解释实验原理", "pre_lab"))
        self.assertFalse(agent_core.is_quiz_generation_request("请生成预习题", "during_experiment"))

    def test_focus_parser_accepts_only_five_whitelisted_values(self) -> None:
        for focus in agent_core.QUIZ_FOCUS_AREAS:
            with self.subTest(focus=focus):
                self.assertEqual(
                    agent_core.parse_quiz_focus(f"请出题，侧重点：{focus}"),
                    focus,
                )

    def test_focus_parser_falls_back_to_comprehensive_mode(self) -> None:
        self.assertEqual(agent_core.parse_quiz_focus("请生成预习题"), "综合")
        self.assertEqual(agent_core.parse_quiz_focus("请出题，侧重点：实验应用"), "综合")
        self.assertEqual(agent_core.parse_quiz_focus("请围绕数据处理出题"), "综合")

    def test_focus_is_injected_into_prompt_and_retrieval_query(self) -> None:
        student_input = "请生成5道预习题，侧重点：数据处理"

        prompt = agent_core.build_quiz_prompt(
            test_resources()["quiz_prompt"],
            "知识库",
            student_input,
        )
        query = agent_core.build_quiz_rag_query(student_input)

        self.assertIn("本次侧重点为“数据处理”", prompt)
        self.assertIn("至少 3 道题", prompt)
        self.assertIn("侧重点：数据处理", query)
        self.assertIn("Hann窗", query)

    def test_modern_application_focus_uses_application_retrieval_terms(self) -> None:
        query = agent_core.build_quiz_rag_query(
            "请生成预习题，侧重点：现代应用"
        )

        self.assertIn("精密光学表面检测", query)
        self.assertIn("MEMS", query)
        self.assertIn("振动测量", query)

    def test_parser_accepts_json_fence_and_adds_server_fields(self) -> None:
        raw = "```json\n" + json.dumps(valid_quiz_payload(), ensure_ascii=False) + "\n```"

        parsed = json.loads(agent_core.parse_quiz_model_output(raw))

        self.assertTrue(parsed["quiz"]["quiz_id"].startswith("quiz_"))
        self.assertEqual(parsed["quiz"]["stage"], "pre_lab")
        self.assertEqual(len(parsed["quiz"]["questions"]), 5)

    def test_generation_retries_once_after_invalid_json(self) -> None:
        client = SequenceModelClient(
            ["not-json", json.dumps(valid_quiz_payload(), ensure_ascii=False)]
        )

        with patch.object(
            agent_core,
            "build_rag_augmented_knowledge",
            side_effect=lambda base, _query, **_kwargs: base,
        ):
            quiz_json = agent_core.generate_quiz_json(
                "请生成5道预习练习题，侧重点：理论公式",
                client=client,
                resources=test_resources(),
            )

        self.assertEqual(client.calls, 2)
        self.assertIn("上一次输出校验失败", client.prompts[1])
        self.assertIn("本次侧重点为“理论公式”", client.prompts[1])
        self.assertEqual(json.loads(quiz_json)["quiz"]["stage"], "pre_lab")

    def test_generation_injects_reliable_rag_context(self) -> None:
        client = SequenceModelClient(
            [json.dumps(valid_quiz_payload(), ensure_ascii=False)]
        )
        rag_result = {
            "enabled": True,
            "available": True,
            "reliable": True,
            "reason": "ok",
            "results": [
                {
                    "text": "补充资料：统一主频用于保证像素间相位可比。",
                    "source_file": "guidance.md",
                    "section": "FFT",
                    "score": 0.9,
                }
            ],
        }

        with patch.object(agent_core, "retrieve_rag_context", return_value=rag_result):
            agent_core.generate_quiz_json(
                "请生成预习题",
                client=client,
                resources=test_resources(),
            )

        self.assertIn("连续扫描实验知识库", client.prompts[0])
        self.assertIn("统一主频用于保证像素间相位可比", client.prompts[0])

    def test_generation_falls_back_to_base_knowledge_for_unreliable_rag(self) -> None:
        client = SequenceModelClient(
            [json.dumps(valid_quiz_payload(), ensure_ascii=False)]
        )
        rag_result = {
            "enabled": True,
            "available": True,
            "reliable": False,
            "reason": "low_confidence",
            "results": [{"text": "不可靠补充资料"}],
        }

        with patch.object(agent_core, "retrieve_rag_context", return_value=rag_result):
            agent_core.generate_quiz_json(
                "请生成预习题",
                client=client,
                resources=test_resources(),
            )

        self.assertIn("连续扫描实验知识库", client.prompts[0])
        self.assertNotIn("不可靠补充资料", client.prompts[0])

    def test_stream_quiz_buffers_until_validated(self) -> None:
        client = SequenceModelClient(
            [json.dumps(valid_quiz_payload(), ensure_ascii=False)]
        )

        with patch.object(
            agent_core,
            "build_rag_augmented_knowledge",
            side_effect=lambda base, _query, **_kwargs: base,
        ):
            events = list(
                agent_core.stream_tutor_events(
                    "请生成5道预习练习题",
                    stage="pre_lab",
                    save=False,
                    client=client,
                    resources=test_resources(),
                )
            )

        self.assertEqual([event["event"] for event in events], ["meta", "delta", "done"])
        self.assertEqual(events[0]["data"]["route"], "quiz_generate")
        self.assertEqual(events[1]["data"]["text"], events[2]["data"]["answer_markdown"])
        self.assertEqual(json.loads(events[1]["data"]["text"])["quiz"]["stage"], "pre_lab")

    def test_stream_quiz_returns_format_error_without_delta(self) -> None:
        client = SequenceModelClient(["bad-json", "still-bad"])

        with patch.object(
            agent_core,
            "build_rag_augmented_knowledge",
            side_effect=lambda base, _query, **_kwargs: base,
        ):
            events = list(
                agent_core.stream_tutor_events(
                    "请生成预习题",
                    stage="pre_lab",
                    save=False,
                    client=client,
                    resources=test_resources(),
                )
            )

        self.assertEqual([event["event"] for event in events], ["meta", "error"])
        self.assertEqual(events[-1]["data"]["error_code"], "QUIZ_FORMAT_ERROR")

    def test_existing_chat_endpoint_returns_validated_quiz_json(self) -> None:
        model_client = SequenceModelClient(
            [json.dumps(valid_quiz_payload(), ensure_ascii=False)]
        )
        request_body = {
            "session_id": "proj_001",
            "stage": "pre_lab",
            "student_input": "请生成5道预习练习题，侧重点：数据处理，以JSON格式输出。",
            "conversation_history": [],
        }

        with (
            patch.object(agent_core, "create_client", return_value=model_client),
            patch.object(
                agent_core,
                "build_rag_augmented_knowledge",
                side_effect=lambda base, _query, **_kwargs: base,
            ),
            patch.object(agent_core, "save_output"),
        ):
            response = TestClient(api_server.app).post(
                "/api/agent/chat",
                json=request_body,
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["success"])
        self.assertEqual(json.loads(payload["answer_markdown"])["quiz"]["stage"], "pre_lab")

    def test_existing_stream_endpoint_uses_document_request_fields(self) -> None:
        model_client = SequenceModelClient(
            [json.dumps(valid_quiz_payload(), ensure_ascii=False)]
        )

        with (
            patch.object(agent_core, "create_client", return_value=model_client),
            patch.object(
                agent_core,
                "build_rag_augmented_knowledge",
                side_effect=lambda base, _query, **_kwargs: base,
            ),
            patch.object(agent_core, "save_output"),
        ):
            response = TestClient(api_server.app).post(
                "/api/agent/chat/stream",
                json={
                    "session_id": "proj_001",
                    "stage": "pre_lab",
                    "student_input": "请生成5道预习练习题，侧重点：数据处理，以JSON格式输出。",
                    "conversation_history": [],
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("event: meta", response.text)
        self.assertIn('"route": "quiz_generate"', response.text)
        self.assertEqual(response.text.count("event: delta"), 1)
        self.assertIn("event: done", response.text)


class QuizKnowledgeTests(TestCase):
    def test_modern_application_knowledge_is_available_with_boundary(self) -> None:
        base_knowledge = (PROJECT_DIR / "docs" / "04_实验知识库初版.md").read_text(
            encoding="utf-8"
        )
        rag_document = (
            PROJECT_DIR
            / "rag_docs"
            / "guidance"
            / "2026_迈克尔逊干涉与表面形貌测量现代应用.md"
        ).read_text(encoding="utf-8")

        for content in (base_knowledge, rag_document):
            self.assertIn("精密光学表面检测", content)
            self.assertIn("MEMS", content)
            self.assertIn("工业级检验结论", content)
            self.assertIn("不确定度", content)
