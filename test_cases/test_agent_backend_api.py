from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import os
import sys
import warnings
from io import BytesIO
from pathlib import Path
from typing import Any
from unittest import TestCase
from unittest.mock import patch

warnings.filterwarnings(
    "ignore",
    message="Using `httpx` with `starlette.testclient` is deprecated.*",
    category=Warning,
)

from fastapi.testclient import TestClient
from PIL import Image
from pypdf import PdfReader, PdfWriter
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
    NumberObject,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from document_loader import MAX_PDF_SIZE_BYTES, PdfExtractionResult
from vision_client import VisionImage


STREAM_IMAGE_LIMIT_BYTES = 15 * 1024 * 1024
STREAM_PDF_LIMIT_BYTES = 30 * 1024 * 1024
STREAM_VISION_IMAGE_LIMIT = 12


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_api_server_module(module_name: str = "api_server_upload_tests"):
    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))
    return load_module(module_name, PROJECT_DIR / "api_server.py")


def build_text_pdf(text: str) -> bytes:
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {NameObject("/F1"): font}
            )
        }
    )
    stream = DecodedStreamObject()
    escaped_text = (
        text.replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
    )
    stream.set_data(
        f"BT /F1 12 Tf 72 720 Td ({escaped_text}) Tj ET".encode("latin-1")
    )
    page[NameObject("/Contents")] = writer._add_object(stream)

    buffer = BytesIO()
    writer.write(buffer)
    result = buffer.getvalue()
    assert PdfReader(BytesIO(result)).pages, "build_text_pdf produced unreadable PDF"
    return result


def build_pdf_with_images(image_count: int, text: str | None = None) -> bytes:
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    resources = DictionaryObject()
    content_parts: list[str] = []

    if text is not None:
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        resources[NameObject("/Font")] = DictionaryObject({NameObject("/F1"): font})
        escaped_text = (
            text.replace("\\", "\\\\")
            .replace("(", "\\(")
            .replace(")", "\\)")
        )
        content_parts.append(f"BT /F1 12 Tf 72 720 Td ({escaped_text}) Tj ET")

    xobjects = DictionaryObject()
    for index in range(image_count):
        image = DecodedStreamObject()
        image.set_data(bytes([index % 256, 0, 0]))
        image.update(
            {
                NameObject("/Type"): NameObject("/XObject"),
                NameObject("/Subtype"): NameObject("/Image"),
                NameObject("/Width"): NumberObject(1),
                NameObject("/Height"): NumberObject(1),
                NameObject("/ColorSpace"): NameObject("/DeviceRGB"),
                NameObject("/BitsPerComponent"): NumberObject(8),
            }
        )
        name = NameObject(f"/Im{index + 1}")
        xobjects[name] = writer._add_object(image)
        content_parts.append(f"q 10 0 0 10 72 {680 - index} cm {name} Do Q")

    resources[NameObject("/XObject")] = xobjects
    page[NameObject("/Resources")] = resources
    stream = DecodedStreamObject()
    stream.set_data(" ".join(content_parts).encode("latin-1"))
    page[NameObject("/Contents")] = writer._add_object(stream)

    buffer = BytesIO()
    writer.write(buffer)
    result = buffer.getvalue()
    assert PdfReader(BytesIO(result)).pages, "build_pdf_with_images produced unreadable PDF"
    return result


class AgentCoreBackendTests(TestCase):
    def test_agent_core_can_be_imported_without_api_key(self) -> None:
        agent_core = load_module("agent_core", PROJECT_DIR / "agent_core.py")

        self.assertTrue(hasattr(agent_core, "ask_tutor"))
        self.assertTrue(hasattr(agent_core, "check_report"))
        self.assertTrue(hasattr(agent_core, "build_frontend_payload"))

    def test_classify_tutor_route_identifies_guided_tutor_questions(self) -> None:
        agent_core = load_module("agent_core_routes_guided", PROJECT_DIR / "agent_core.py")

        self.assertEqual(agent_core.classify_tutor_route("为什么要用时间 FFT 提取主频相位？"), "guided_tutor")
        self.assertEqual(agent_core.classify_tutor_route("实验时怎么调出清晰条纹？"), "guided_tutor")
        self.assertEqual(agent_core.classify_tutor_route("迈克尔孙测量步骤"), "guided_tutor")
        self.assertEqual(agent_core.classify_tutor_route("迈克尔森干涉仪怎么调条纹"), "guided_tutor")

    def test_classify_tutor_route_uses_stage_as_auxiliary_signal(self) -> None:
        agent_core = load_module("agent_core_routes_stage", PROJECT_DIR / "agent_core.py")

        self.assertEqual(
            agent_core.classify_tutor_route("下一步怎么做", stage="measurement"),
            "guided_tutor",
        )
        self.assertEqual(
            agent_core.classify_tutor_route("今天吃什么", stage="measurement"),
            "out_of_scope",
        )

    def test_normalize_tutor_stage_maps_legacy_values_to_three_stage_model(self) -> None:
        agent_core = load_module("agent_core_stage_normalization", PROJECT_DIR / "agent_core.py")

        expected = {
            "pre_lab": "pre_lab",
            "measurement": "during_experiment",
            "data_processing": "result_evaluation",
            "physics_calculation": "result_evaluation",
            "error_analysis": "result_evaluation",
            "result_evaluation": "result_evaluation",
            "report": "result_evaluation",
            "unexpected": "unknown",
        }

        for raw_stage, normalized_stage in expected.items():
            self.assertEqual(agent_core.normalize_tutor_stage(raw_stage), normalized_stage)

    def test_stage_rule_classifier_requires_one_unambiguous_profile(self) -> None:
        agent_core = load_module("agent_core_stage_rules", PROJECT_DIR / "agent_core.py")

        self.assertEqual(agent_core.infer_tutor_stage_from_rules("课前预习需要准备什么？"), "pre_lab")
        self.assertEqual(agent_core.infer_tutor_stage_from_rules("现场看不到条纹怎么调节？"), "during_experiment")
        self.assertEqual(agent_core.infer_tutor_stage_from_rules("这个计算的单位和误差是否正确？"), "result_evaluation")
        self.assertEqual(agent_core.infer_tutor_stage_from_rules("相机帧率和曝光应该怎么设置？"), "during_experiment")
        self.assertEqual(agent_core.infer_tutor_stage_from_rules("包裹相位如何做二维相位展开？"), "result_evaluation")
        self.assertEqual(agent_core.infer_tutor_stage_from_rules("背景平面扣除后的高度图怎么评价？"), "result_evaluation")
        self.assertEqual(agent_core.infer_tutor_stage_from_rules("我想问一个实验问题"), "unknown")

    def test_unknown_stage_uses_structured_model_fallback(self) -> None:
        agent_core = load_module("agent_core_stage_fallback", PROJECT_DIR / "agent_core.py")
        client = RecordingModelClient('{"stage": "during_experiment"}')

        stage = agent_core.resolve_tutor_stage(
            "unknown",
            "这个现象怎么处理？",
            classifier_template="CLASSIFIER_MARKER\n{recent_context}\n{student_question}",
            classifier_client=client,
        )

        self.assertEqual(stage, "during_experiment")
        self.assertEqual(client.calls, 1)
        self.assertIn("CLASSIFIER_MARKER", client.user_prompt)

    def test_explicit_stage_skips_model_classification_and_injects_strategy(self) -> None:
        agent_core = load_module("agent_core_stage_explicit", PROJECT_DIR / "agent_core.py")
        client = RecordingModelClient("请先确认条纹是否稳定。")

        payload = agent_core.ask_tutor(
            "下一步怎么做？",
            stage="measurement",
            save=False,
            client=client,
            resources=build_test_resources(),
        )

        self.assertEqual(payload["stage"], "measurement")
        self.assertEqual(client.calls, 1)
        self.assertIn("DURING_STAGE_MARKER", client.user_prompt)
        self.assertIn('"stage": "during_experiment"', client.user_prompt)

    def test_unknown_rule_stage_injects_result_strategy(self) -> None:
        agent_core = load_module("agent_core_stage_rule_prompt", PROJECT_DIR / "agent_core.py")
        client = RecordingModelClient("请先核对单位。")

        payload = agent_core.ask_tutor(
            "这组数据的单位和误差如何检查？",
            stage="unknown",
            save=False,
            client=client,
            resources=build_test_resources(),
        )

        self.assertTrue(payload["success"])
        self.assertEqual(client.calls, 1)
        self.assertIn("RESULT_STAGE_MARKER", client.user_prompt)
        self.assertIn('"stage": "result_evaluation"', client.user_prompt)

    def test_teaching_policy_rules_keep_simple_direct_and_complex_guided(self) -> None:
        agent_core = load_module("agent_core_teaching_rules", PROJECT_DIR / "agent_core.py")

        self.assertEqual(
            agent_core.infer_teaching_policy_from_rules(
                "分束器是什么？",
                "pre_lab",
            ),
            "direct",
        )
        self.assertEqual(
            agent_core.infer_teaching_policy_from_rules(
                "系统误差是什么？",
                "result_evaluation",
            ),
            "direct",
        )
        self.assertEqual(
            agent_core.infer_teaching_policy_from_rules(
                "为什么反射镜移动 d 会产生 2d 的光程变化？",
                "pre_lab",
            ),
            "guided",
        )
        self.assertEqual(
            agent_core.infer_teaching_policy_from_rules(
                "不要直接给答案，请一步一步引导我。",
                "pre_lab",
            ),
            "guided",
        )
        self.assertEqual(
            agent_core.infer_teaching_policy_from_rules(
                "现场看不到条纹，下一步怎么做？",
                "during_experiment",
            ),
            "hybrid",
        )
        self.assertEqual(
            agent_core.infer_teaching_policy_from_rules(
                "设备冒烟了，我该怎么办？",
                "during_experiment",
            ),
            "direct",
        )

    def test_teaching_policy_uses_structured_model_only_for_ambiguous_input(self) -> None:
        agent_core = load_module("agent_core_teaching_fallback", PROJECT_DIR / "agent_core.py")
        client = RecordingModelClient(
            '{"policy":"guided","student_status":"new","guidance_level":1}'
        )

        decision = agent_core.resolve_teaching_decision(
            "pre_lab",
            "这个地方该怎么理解？",
            classifier_template=(
                "TEACHING_CLASSIFIER_MARKER\n{stage}\n{recent_context}\n{student_question}"
            ),
            classifier_client=client,
        )

        self.assertEqual(decision["policy"], "guided")
        self.assertEqual(decision["guidance_level"], 1)
        self.assertEqual(client.calls, 1)
        self.assertIn("TEACHING_CLASSIFIER_MARKER", client.user_prompt)

    def test_teaching_policy_invalid_model_output_falls_back_to_direct(self) -> None:
        agent_core = load_module("agent_core_teaching_invalid", PROJECT_DIR / "agent_core.py")

        decision = agent_core.parse_teaching_policy_classifier_output("not-json")

        self.assertEqual(
            decision,
            {"policy": "direct", "student_status": "new", "guidance_level": 0},
        )

    def test_guidance_history_advances_and_releases_answer_at_level_three(self) -> None:
        agent_core = load_module("agent_core_guidance_progress", PROJECT_DIR / "agent_core.py")
        history = [
            {"role": "user", "content": "为什么光程变化是 2d？"},
            {"role": "assistant", "content": "可以先区分去程和回程，各变化多少？"},
            {"role": "user", "content": "我不知道。"},
            {"role": "assistant", "content": "关键是两段路径都会变化 d，你能把它们相加吗？"},
        ]

        decision = agent_core.resolve_teaching_decision(
            "pre_lab",
            "还是不懂。",
            history,
        )

        self.assertEqual(decision["policy"], "guided")
        self.assertEqual(decision["student_status"], "stuck")
        self.assertEqual(decision["guidance_level"], 3)

    def test_direct_answer_request_overrides_guided_history(self) -> None:
        agent_core = load_module("agent_core_guidance_direct", PROJECT_DIR / "agent_core.py")
        history = [
            {"role": "assistant", "content": "你可以先判断去程变化多少？"},
        ]

        decision = agent_core.resolve_teaching_decision(
            "pre_lab",
            "我还是不会，直接告诉我完整推导。",
            history,
        )

        self.assertEqual(decision["policy"], "direct")
        self.assertEqual(decision["student_status"], "asks_direct")
        self.assertEqual(decision["guidance_level"], 0)

    def test_teaching_policy_override_guided_skips_classifier_and_clamps_level(self) -> None:
        agent_core = load_module("agent_core_teaching_override_guided", PROJECT_DIR / "agent_core.py")
        client = RecordingModelClient(
            '{"policy":"direct","student_status":"new","guidance_level":0}'
        )

        decision = agent_core.resolve_teaching_decision(
            "pre_lab",
            "分束器是什么？",
            teaching_policy="guided",
            guidance_level=9,
            classifier_template=(
                "TEACHING_CLASSIFIER_MARKER\n{stage}\n{recent_context}\n{student_question}"
            ),
            classifier_client=client,
        )

        self.assertEqual(decision["policy"], "guided")
        self.assertEqual(decision["guidance_level"], 3)
        self.assertEqual(client.calls, 0)

    def test_teaching_policy_override_invalid_falls_back_to_direct_without_classifier(self) -> None:
        agent_core = load_module("agent_core_teaching_override_invalid", PROJECT_DIR / "agent_core.py")
        client = RecordingModelClient(
            '{"policy":"guided","student_status":"new","guidance_level":3}'
        )

        decision = agent_core.resolve_teaching_decision(
            "pre_lab",
            "为什么 d = Nλ/2？",
            teaching_policy="xxx",
            guidance_level=3,
            classifier_template=(
                "TEACHING_CLASSIFIER_MARKER\n{stage}\n{recent_context}\n{student_question}"
            ),
            classifier_client=client,
        )

        self.assertEqual(decision["policy"], "direct")
        self.assertEqual(decision["guidance_level"], 0)
        self.assertEqual(client.calls, 0)

    def test_guidance_level_override_keeps_auto_policy_and_direct_zero(self) -> None:
        agent_core = load_module("agent_core_teaching_level_override", PROJECT_DIR / "agent_core.py")

        direct_decision = agent_core.resolve_teaching_decision(
            "pre_lab",
            "分束器是什么？",
            guidance_level=3,
        )
        guided_decision = agent_core.resolve_teaching_decision(
            "pre_lab",
            "为什么反射镜移动 d 会产生 2d 的光程变化？",
            guidance_level=9,
        )

        self.assertEqual(direct_decision["policy"], "direct")
        self.assertEqual(direct_decision["guidance_level"], 0)
        self.assertEqual(guided_decision["policy"], "guided")
        self.assertEqual(guided_decision["guidance_level"], 3)

    def test_teaching_policy_override_without_level_uses_history_level(self) -> None:
        agent_core = load_module("agent_core_teaching_override_history_level", PROJECT_DIR / "agent_core.py")
        history = [
            {"role": "assistant", "content": "Can you separate outbound and return paths?"},
            {"role": "user", "content": "Not sure."},
            {"role": "assistant", "content": "What changes on the return path?"},
        ]

        decision = agent_core.resolve_teaching_decision(
            "pre_lab",
            "分束器是什么？",
            history,
            teaching_policy="guided",
        )

        self.assertEqual(decision["policy"], "guided")
        self.assertEqual(decision["guidance_level"], 3)

    def test_safety_question_ignores_guided_teaching_override(self) -> None:
        agent_core = load_module("agent_core_teaching_override_safety", PROJECT_DIR / "agent_core.py")

        decision = agent_core.resolve_teaching_decision(
            "during_experiment",
            "设备冒烟了，我该怎么办？",
            teaching_policy="guided",
            guidance_level=3,
        )

        self.assertEqual(decision["policy"], "direct")
        self.assertEqual(decision["guidance_level"], 0)

    def test_tutor_prompt_injects_stage_and_teaching_strategy(self) -> None:
        agent_core = load_module("agent_core_teaching_prompt", PROJECT_DIR / "agent_core.py")
        client = RecordingModelClient("先区分去程和回程。")

        payload = agent_core.ask_tutor(
            "为什么反射镜移动 d 会产生 2d 的光程变化？",
            stage="pre_lab",
            save=False,
            client=client,
            resources=build_test_resources(),
        )

        self.assertTrue(payload["success"])
        self.assertEqual(client.calls, 1)
        self.assertIn("PRE_LAB_STAGE_MARKER", client.user_prompt)
        self.assertIn("GUIDED_TEACHING_MARKER level=1 status=new", client.user_prompt)
        self.assertIn('"teaching_policy": "guided"', client.user_prompt)

    def test_classify_tutor_route_identifies_experiment_workflow_tasks(self) -> None:
        agent_core = load_module("agent_core_routes_workflow", PROJECT_DIR / "agent_core.py")

        history = [
            {"role": "user", "content": "我已经完成了迈克尔逊干涉仪的课前预习。"},
            {"role": "assistant", "content": "可以继续整理现场测量步骤。"},
        ]

        self.assertEqual(
            agent_core.classify_tutor_route(
                "梳理一下上下文，汇总成实验记录",
                conversation_history=history,
            ),
            "experiment_workflow",
        )
        self.assertEqual(
            agent_core.classify_tutor_route("把这段实验描述改写得更清楚"),
            "experiment_workflow",
        )
        self.assertEqual(
            agent_core.classify_tutor_route(
                "整理一下刚才内容",
                conversation_history=history,
            ),
            "experiment_workflow",
        )

    def test_classify_tutor_route_uses_recent_experiment_history(self) -> None:
        agent_core = load_module("agent_core_routes_history", PROJECT_DIR / "agent_core.py")
        history = [
            {"role": "user", "content": "迈克尔逊干涉仪测量步骤里要先调出清晰条纹。"},
            {"role": "assistant", "content": "可以继续观察反射镜移动后的条纹变化。"},
        ]

        self.assertEqual(
            agent_core.classify_tutor_route("这个怎么弄", conversation_history=history),
            "guided_tutor",
        )
        self.assertEqual(
            agent_core.classify_tutor_route("继续解释", conversation_history=history),
            "guided_tutor",
        )

    def test_classify_tutor_route_identifies_out_of_scope_questions(self) -> None:
        agent_core = load_module("agent_core_routes_scope", PROJECT_DIR / "agent_core.py")

        self.assertEqual(agent_core.classify_tutor_route("推荐一部电影"), "out_of_scope")
        self.assertEqual(agent_core.classify_tutor_route("Python 爬虫怎么写"), "out_of_scope")
        self.assertEqual(agent_core.classify_tutor_route("今天吃什么", stage="measurement"), "out_of_scope")

    def test_classify_tutor_route_identifies_social_messages_before_out_of_scope(self) -> None:
        agent_core = load_module("agent_core_routes_social", PROJECT_DIR / "agent_core.py")

        self.assertEqual(agent_core.classify_tutor_route("你好"), "social")
        self.assertEqual(agent_core.classify_tutor_route("thanks"), "social")
        self.assertEqual(agent_core.classify_tutor_route("拜拜"), "social")
        self.assertEqual(
            agent_core.classify_tutor_route("你好，请问条纹怎么调？"),
            "guided_tutor",
        )

    def test_workflow_prompt_contains_experiment_workflow_constraints(self) -> None:
        prompt = (PROJECT_DIR / "prompts" / "workflow_prompt.txt").read_text(
            encoding="utf-8",
        )

        self.assertIn("{conversation_history}", prompt)
        self.assertIn("{student_question}", prompt)
        self.assertIn("不强制使用固定教学标题", prompt)
        self.assertIn("不得编造实验数据", prompt)
        self.assertIn("不得生成可直接提交的完整实验报告正文", prompt)
        self.assertIn("未提供", prompt)
        self.assertIn("需要学生补充", prompt)

    def test_tutor_prompt_uses_optional_stage_structure_instead_of_fixed_five_sections(self) -> None:
        prompt = (PROJECT_DIR / "prompts" / "tutor_prompt.txt").read_text(encoding="utf-8")

        self.assertIn("{stage_instruction}", prompt)
        self.assertIn("不强制固定标题", prompt)
        self.assertNotIn("### 问题判断", prompt)
        self.assertNotIn("### 原理提示", prompt)
        self.assertNotIn("### 引导思考", prompt)
        self.assertNotIn("### 操作或学习建议", prompt)
        self.assertNotIn("### 注意事项", prompt)

    def test_build_tutor_prompt_trims_history_to_max_turns(self) -> None:
        agent_core = load_module("agent_core_tutor_prompt_history", PROJECT_DIR / "agent_core.py")
        resources = build_test_resources()
        history = [
            {"role": "user", "content": f"TUTOR_HISTORY_MARKER_{index:02d}"}
            for index in range(20)
        ]

        prompt = agent_core.build_tutor_prompt(
            resources["tutor_prompt"],
            resources["knowledge_base"],
            "为什么 d = Nλ/2？",
            {"turn_count": 10},
            history,
        )

        for index in range(12):
            self.assertNotIn(f"TUTOR_HISTORY_MARKER_{index:02d}", prompt)
        for index in range(12, 20):
            self.assertIn(f"TUTOR_HISTORY_MARKER_{index:02d}", prompt)

    def test_build_workflow_prompt_trims_history_to_max_turns(self) -> None:
        agent_core = load_module("agent_core_workflow_prompt_history", PROJECT_DIR / "agent_core.py")
        resources = build_test_resources()
        history = [
            {"role": "user", "content": f"WORKFLOW_HISTORY_MARKER_{index:02d}"}
            for index in range(20)
        ]

        prompt = agent_core.build_workflow_prompt(
            resources["workflow_prompt"],
            "梳理一下上下文，汇总成实验记录",
            {"turn_count": 10},
            history,
        )

        for index in range(12):
            self.assertNotIn(f"WORKFLOW_HISTORY_MARKER_{index:02d}", prompt)
        for index in range(12, 20):
            self.assertIn(f"WORKFLOW_HISTORY_MARKER_{index:02d}", prompt)

    def test_ask_tutor_state_turn_count_reflects_full_history(self) -> None:
        agent_core = load_module("agent_core_turn_count_full_history", PROJECT_DIR / "agent_core.py")
        client = RecordingModelClient("### 问题判断\n这是一个原理问题。")
        history = [
            {
                "role": "user" if index % 2 == 0 else "assistant",
                "content": f"STATE_HISTORY_MARKER_{index:02d}",
            }
            for index in range(20)
        ]

        payload = agent_core.ask_tutor(
            "为什么 d = Nλ/2？",
            stage="pre_lab",
            conversation_history=history,
            save=False,
            client=client,
            resources=build_test_resources(),
        )

        self.assertTrue(payload["success"])
        self.assertEqual(client.calls, 1)
        self.assertIn("TUTOR_PROMPT_MARKER", client.user_prompt)
        self.assertIn('"turn_count": 10', client.user_prompt)
        for index in range(12):
            self.assertNotIn(f"STATE_HISTORY_MARKER_{index:02d}", client.user_prompt)
        for index in range(12, 20):
            self.assertIn(f"STATE_HISTORY_MARKER_{index:02d}", client.user_prompt)

    def test_ask_tutor_guided_route_uses_tutor_prompt(self) -> None:
        agent_core = load_module("agent_core_guided_call", PROJECT_DIR / "agent_core.py")
        client = RecordingModelClient("### 问题判断\n这是一个原理问题。")

        payload = agent_core.ask_tutor(
            "为什么 d = Nλ/2？",
            stage="pre_lab",
            save=False,
            client=client,
            resources=build_test_resources(),
        )

        self.assertEqual(payload["answer_markdown"], "### 问题判断\n这是一个原理问题。")
        self.assertEqual(client.calls, 1)
        self.assertIn("TUTOR_PROMPT_MARKER", client.user_prompt)
        self.assertNotIn("WORKFLOW_PROMPT_MARKER", client.user_prompt)

    def test_ask_tutor_standard_qa_returns_exact_answer_without_model_or_rag(self) -> None:
        agent_core = load_module("agent_core_standard_qa_call", PROJECT_DIR / "agent_core.py")
        qa_items = json.loads(
            (PROJECT_DIR / "assets" / "standard_qa.json").read_text(encoding="utf-8")
        )["items"]
        standard_item = next(
            item for item in qa_items if item["id"] == "beam_splitter_role"
        )

        for stage in ["unknown", "pre_lab", "during_experiment", "result_evaluation"]:
            with self.subTest(stage=stage):
                client = RecordingModelClient("不应该调用模型")
                with patch.object(agent_core, "retrieve_rag_context") as rag_mock:
                    with patch.object(agent_core, "select_teaching_images") as image_mock:
                        payload = agent_core.ask_tutor(
                            "分束镜的作用是什么？",
                            stage=stage,
                            save=False,
                            client=client,
                            resources=build_test_resources(),
                        )

                self.assertEqual(client.calls, 0)
                rag_mock.assert_not_called()
                image_mock.assert_not_called()
                self.assertEqual(payload["answer_markdown"], standard_item["answer_markdown"])
                self.assertEqual(
                    set(payload),
                    {
                        "success",
                        "mode",
                        "stage",
                        "display_type",
                        "answer_markdown",
                        "saved",
                        "warning",
                        "need_student_reply",
                    },
                )
                self.assertTrue(payload["success"])
                self.assertEqual(payload["mode"], "tutor")
                self.assertEqual(payload["stage"], stage)
                self.assertEqual(payload["display_type"], "markdown")
                self.assertFalse(payload["saved"])

    def test_ask_tutor_injects_reliable_rag_context_without_replacing_base_knowledge(self) -> None:
        agent_core = load_module("agent_core_tutor_rag_hit", PROJECT_DIR / "agent_core.py")
        client = RecordingModelClient("### 问题判断\n这是一个原理问题。")
        rag_result = {
            "enabled": True,
            "available": True,
            "reliable": True,
            "reason": "high_confidence_match",
            "query": "为什么 d = Nλ/2？",
            "results": [
                {
                    "text": "扩展资料片段：反射镜移动导致光程差变化为 2d。",
                    "score": 0.92,
                    "source_file": "实验指导.md",
                    "section": "光程差",
                    "task_type": "principle",
                }
            ],
        }

        with patch.object(agent_core, "retrieve_rag_context", return_value=rag_result):
            payload = agent_core.ask_tutor(
                "为什么 d = Nλ/2？",
                stage="pre_lab",
                save=False,
                client=client,
                resources=build_test_resources(),
            )

        self.assertTrue(payload["success"])
        self.assertIn("【基础知识库与固定规则：始终有效】", client.user_prompt)
        self.assertIn("实验知识库", client.user_prompt)
        self.assertIn("【扩展资料检索结果：仅作为补充依据】", client.user_prompt)
        self.assertIn("扩展资料片段：反射镜移动导致光程差变化为 2d。", client.user_prompt)
        self.assertIn("基础规则、核心公式、实验边界和防代写要求优先级最高", client.user_prompt)

    def test_ask_tutor_falls_back_to_base_knowledge_when_rag_is_unreliable(self) -> None:
        agent_core = load_module("agent_core_tutor_rag_fallback", PROJECT_DIR / "agent_core.py")
        client = RecordingModelClient("### 问题判断\n这是一个原理问题。")
        rag_result = {
            "enabled": True,
            "available": True,
            "reliable": False,
            "reason": "low_confidence",
            "query": "为什么 d = Nλ/2？",
            "results": [],
        }

        with patch.object(agent_core, "retrieve_rag_context", return_value=rag_result):
            agent_core.ask_tutor(
                "为什么 d = Nλ/2？",
                stage="pre_lab",
                save=False,
                client=client,
                resources=build_test_resources(),
            )

        self.assertIn("实验知识库", client.user_prompt)
        self.assertNotIn("【扩展资料检索结果：仅作为补充依据】", client.user_prompt)

    def test_ask_tutor_injects_teaching_images_prompt_when_matched(self) -> None:
        agent_core = load_module("agent_core_tutor_images", PROJECT_DIR / "agent_core.py")
        client = RecordingModelClient("### 原理提示\n光路解释。")
        image_path, created = ensure_test_asset_image("michelson_optical_path.jpg")
        try:
            payload = agent_core.ask_tutor(
                "请解释迈克尔逊干涉仪的光路和分光过程。",
                stage="pre_lab",
                save=False,
                client=client,
                resources=build_test_resources(),
            )
        finally:
            if created:
                image_path.unlink(missing_ok=True)

        self.assertTrue(payload["success"])
        self.assertIn("【可用教学图片】", client.user_prompt)
        self.assertIn("迈克耳孙干涉仪实验光路图", client.user_prompt)
        self.assertIn(
            "![迈克耳孙干涉仪实验光路图](/static/ai_agent/images/michelson_optical_path.jpg)",
            client.user_prompt,
        )
        self.assertIn("必须将每条给出的 Markdown 恰好插入一次", client.user_prompt)

    def test_ask_tutor_does_not_inject_images_when_disabled(self) -> None:
        agent_core = load_module("agent_core_tutor_images_disabled", PROJECT_DIR / "agent_core.py")
        client = RecordingModelClient("### 原理提示\n光路解释。")
        image_path, created = ensure_test_asset_image("michelson_optical_path.jpg")
        try:
            with patch.dict(os.environ, {"AI_AGENT_IMAGES_ENABLED": "false"}, clear=False):
                payload = agent_core.ask_tutor(
                    "请解释迈克尔逊干涉仪的光路。",
                    stage="pre_lab",
                    save=False,
                    client=client,
                    resources=build_test_resources(),
                )
        finally:
            if created:
                image_path.unlink(missing_ok=True)

        self.assertTrue(payload["success"])
        self.assertNotIn("【可用教学图片】", client.user_prompt)
        self.assertNotIn("/static/ai_agent/images/", client.user_prompt)

    def test_check_report_injects_at_most_one_teaching_image(self) -> None:
        agent_core = load_module("agent_core_report_images", PROJECT_DIR / "agent_core.py")
        client = RecordingModelClient("### 总体评价\n报告需要修改。")
        created_paths: list[Path] = []
        image_path, created = ensure_test_asset_image("michelson_optical_path.jpg")
        if created:
            created_paths.append(image_path)
        try:
            payload = agent_core.check_report(
                "报告中的 M1 固定镜、M2 动镜、分光和合束光路描述有误。",
                stage="report",
                save=False,
                client=client,
                resources=build_test_resources(),
            )
        finally:
            for image_path in created_paths:
                image_path.unlink(missing_ok=True)

        self.assertTrue(payload["success"])
        self.assertIn("【可用教学图片】", client.user_prompt)
        self.assertEqual(client.user_prompt.count("/static/ai_agent/images/"), 1)
        self.assertIn("不是学生提交的图像或实验事实", client.user_prompt)

    def test_check_report_can_inject_equal_inclination_image(self) -> None:
        agent_core = load_module("agent_core_report_equal_inclination", PROJECT_DIR / "agent_core.py")
        client = RecordingModelClient("### 总体评价\n需要说明形成条件。")
        image_path, created = ensure_test_asset_image("equal_inclination_rings.jpg")
        try:
            payload = agent_core.check_report(
                "报告将 M1 和 M2 近乎平行形成同心圆条纹的等倾干涉条件解释错误。",
                stage="report",
                save=False,
                client=client,
                resources=build_test_resources(),
            )
        finally:
            if created:
                image_path.unlink(missing_ok=True)

        self.assertTrue(payload["success"])
        self.assertIn("等倾干涉原理图", client.user_prompt)
        self.assertIn("/static/ai_agent/images/equal_inclination_rings.jpg", client.user_prompt)
        self.assertEqual(client.user_prompt.count("/static/ai_agent/images/"), 1)

    def test_workflow_route_injects_images_for_current_semantic_match(self) -> None:
        agent_core = load_module("agent_core_workflow_images", PROJECT_DIR / "agent_core.py")
        neutral_client = RecordingModelClient("## 实验记录整理\n- 已完成：预习。")
        operation_client = RecordingModelClient("## 实验记录整理\n- 记录光路调整步骤。")
        image_path, created = ensure_test_asset_image("michelson_optical_path.jpg")
        neutral_history = [
            {"role": "user", "content": "我已经完成迈克尔逊干涉仪实验的预习。"},
            {"role": "assistant", "content": "可以整理已完成事项。"},
        ]
        operation_history = [
            {"role": "user", "content": "我在记录 M1 固定镜和 M2 动镜的光路调整。"},
            {"role": "assistant", "content": "请继续保存真实观察。"},
        ]
        try:
            neutral_payload = agent_core.ask_tutor(
                "梳理一下上下文，汇总成实验记录",
                stage="pre_lab",
                conversation_history=neutral_history,
                save=False,
                client=neutral_client,
                resources=build_test_resources(),
            )
            operation_payload = agent_core.ask_tutor(
                "整理 M1 固定镜、M2 动镜和分光合束的实验记录",
                stage="pre_lab",
                conversation_history=operation_history,
                save=False,
                client=operation_client,
                resources=build_test_resources(),
            )
        finally:
            if created:
                image_path.unlink(missing_ok=True)

        self.assertTrue(neutral_payload["success"])
        self.assertTrue(operation_payload["success"])
        self.assertIn("WORKFLOW_PROMPT_MARKER", neutral_client.user_prompt)
        self.assertIn("WORKFLOW_PROMPT_MARKER", operation_client.user_prompt)
        self.assertNotIn("【可用教学图片】", neutral_client.user_prompt)
        self.assertIn("【可用教学图片】", operation_client.user_prompt)

    def test_build_report_rag_query_is_bounded_and_keeps_report_signals(self) -> None:
        agent_core = load_module("agent_core_report_rag_query", PROJECT_DIR / "agent_core.py")
        long_report = (
            "# 实验报告\n\n"
            "本实验采用电驱连续扫描和时间 FFT 提取主频相位，"
            "背景平面扣除后使用 $h=λΦ_{rel}/(4π)$ 换算相对高度。\n\n"
            "误差分析包括环境振动、有效帧和 ROI。\n"
            + "这是一段很长的正文。" * 200
        )

        query = agent_core.build_report_rag_query(long_report, stage="report")

        self.assertLessEqual(len(query), 600)
        self.assertIn("阶段：report", query)
        self.assertIn("实验报告", query)
        self.assertIn("$h=λΦ_{rel}/(4π)$", query)
        self.assertIn("连续扫描", query)
        self.assertIn("FFT", query)
        self.assertIn("背景平面", query)
        self.assertIn("误差分析", query)

    def test_ask_tutor_workflow_route_uses_workflow_prompt(self) -> None:
        agent_core = load_module("agent_core_workflow_call", PROJECT_DIR / "agent_core.py")
        client = RecordingModelClient("## 实验记录整理\n- 已完成：课前预习。")
        history = [
            {"role": "user", "content": "我已经完成了迈克尔逊干涉仪的课前预习。"},
            {"role": "assistant", "content": "可以继续整理现场测量步骤。"},
        ]

        payload = agent_core.ask_tutor(
            "梳理一下上下文，汇总成实验记录",
            stage="pre_lab",
            conversation_history=history,
            save=False,
            client=client,
            resources=build_test_resources(),
        )

        self.assertEqual(payload["answer_markdown"], "## 实验记录整理\n- 已完成：课前预习。")
        self.assertEqual(client.calls, 1)
        self.assertIn("WORKFLOW_PROMPT_MARKER", client.user_prompt)
        self.assertIn("不得编造实验数据", client.user_prompt)
        self.assertNotIn("TUTOR_PROMPT_MARKER", client.user_prompt)

    def test_ask_tutor_out_of_scope_route_returns_fixed_message_without_model_call(self) -> None:
        agent_core = load_module("agent_core_scope_call", PROJECT_DIR / "agent_core.py")
        client = RecordingModelClient("不应该调用模型")

        payload = agent_core.ask_tutor(
            "推荐一部电影",
            stage="unknown",
            save=False,
            client=client,
            resources=build_test_resources(),
        )

        self.assertEqual(client.calls, 0)
        self.assertIn("迈克尔逊干涉仪实验", payload["answer_markdown"])
        self.assertNotIn("### 问题判断", payload["answer_markdown"])

    def test_ask_tutor_social_route_uses_tutor_prompt(self) -> None:
        agent_core = load_module("agent_core_social_call", PROJECT_DIR / "agent_core.py")
        client = RecordingModelClient("你好，我是迈克尔逊干涉仪实验助教。")

        payload = agent_core.ask_tutor(
            "你好",
            stage="unknown",
            save=False,
            client=client,
            resources=build_test_resources(),
        )

        self.assertEqual(payload["answer_markdown"], "你好，我是迈克尔逊干涉仪实验助教。")
        self.assertEqual(client.calls, 1)
        self.assertIn("TUTOR_PROMPT_MARKER", client.user_prompt)
        self.assertNotIn("WORKFLOW_PROMPT_MARKER", client.user_prompt)

    def test_ask_tutor_logs_runtime_error_without_changing_payload(self) -> None:
        agent_core = load_module("agent_core_tutor_logging", PROJECT_DIR / "agent_core.py")

        def raise_runtime_error(*args: Any, **kwargs: Any) -> str:
            raise RuntimeError("missing config")

        with patch.object(agent_core, "classify_tutor_route", raise_runtime_error):
            with patch.object(agent_core.logger, "exception") as log_exception:
                payload = agent_core.ask_tutor("为什么 d = Nλ/2？", stage="pre_lab")

        self.assertFalse(payload["success"])
        self.assertEqual(payload["error_code"], "CONFIG_ERROR")
        self.assertEqual(payload["message"], "AI 服务尚未配置，请检查环境变量或 .env 配置。")
        self.assertEqual(log_exception.call_count, 1)
        extra = log_exception.call_args.kwargs["extra"]
        self.assertEqual(extra["mode"], "tutor")
        self.assertEqual(extra["stage"], "pre_lab")
        self.assertEqual(extra["route"], "unknown")
        self.assertEqual(extra["error_code"], "CONFIG_ERROR")
        self.assertNotIn("student_input", extra)
        self.assertNotIn("user_prompt", extra)

    def test_check_report_logs_runtime_error_without_changing_payload(self) -> None:
        agent_core = load_module("agent_core_report_logging", PROJECT_DIR / "agent_core.py")

        def raise_runtime_error() -> dict[str, str]:
            raise RuntimeError("missing config")

        with patch.object(agent_core, "load_resources", raise_runtime_error):
            with patch.object(agent_core.logger, "exception") as log_exception:
                payload = agent_core.check_report("报告内容", stage="report")

        self.assertFalse(payload["success"])
        self.assertEqual(payload["error_code"], "CONFIG_ERROR")
        self.assertEqual(payload["message"], "AI 服务尚未配置，请检查环境变量或 .env 配置。")
        self.assertEqual(log_exception.call_count, 1)
        extra = log_exception.call_args.kwargs["extra"]
        self.assertEqual(extra["mode"], "report_check")
        self.assertEqual(extra["stage"], "report")
        self.assertEqual(extra["route"], "unknown")
        self.assertEqual(extra["error_code"], "CONFIG_ERROR")
        self.assertNotIn("report_content", extra)
        self.assertNotIn("knowledge_base", extra)

    def test_iter_model_stream_chunks_closes_stream_response(self) -> None:
        agent_core = load_module("agent_core_stream_close", PROJECT_DIR / "agent_core.py")
        client = StreamingModelClient(["第一段", "第二段"])

        chunks = list(
            agent_core.iter_model_stream_chunks(
                client,
                "system prompt",
                "user prompt",
                0.2,
            )
        )

        self.assertEqual(chunks, ["第一段", "第二段"])
        self.assertTrue(client.last_response.closed)
    def test_stream_tutor_guided_route_returns_meta_delta_done(self) -> None:
        agent_core = load_module("agent_core_guided_stream", PROJECT_DIR / "agent_core.py")
        client = StreamingModelClient(["### 问题判断\n", "这是一个原理问题。"])

        events = list(
            agent_core.stream_tutor_events(
                "为什么 d = Nλ/2？",
                stage="pre_lab",
                save=False,
                client=client,
                resources=build_test_resources(),
            ),
        )

        self.assertEqual(
            [event["event"] for event in events],
            ["meta", "delta", "delta", "done"],
        )
        self.assertEqual(events[0]["data"]["route"], "guided_tutor")
        self.assertEqual(events[0]["data"]["mode"], "tutor")
        self.assertEqual(events[0]["data"]["stage"], "pre_lab")
        self.assertEqual(events[0]["data"]["display_type"], "markdown")
        self.assertEqual(events[1]["data"]["text"], "### 问题判断\n")
        self.assertEqual(
            events[-1]["data"]["answer_markdown"],
            "### 问题判断\n这是一个原理问题。",
        )
        self.assertTrue(events[-1]["data"]["need_student_reply"])
        self.assertFalse(events[-1]["data"]["saved"])
        self.assertIn("TUTOR_PROMPT_MARKER", client.user_prompt)

    def test_stream_tutor_standard_qa_keeps_existing_event_shape(self) -> None:
        agent_core = load_module("agent_core_standard_qa_stream", PROJECT_DIR / "agent_core.py")
        qa_items = json.loads(
            (PROJECT_DIR / "assets" / "standard_qa.json").read_text(encoding="utf-8")
        )["items"]
        standard_item = next(
            item for item in qa_items if item["id"] == "fringe_in_out_principle"
        )
        client = StreamingModelClient(["不应该调用模型"])

        with patch.object(agent_core, "retrieve_rag_context") as rag_mock:
            with patch.object(agent_core, "select_teaching_images") as image_mock:
                events = list(
                    agent_core.stream_tutor_events(
                        "为什么会出现条纹吞吐？",
                        stage="pre_lab",
                        save=False,
                        client=client,
                        resources=build_test_resources(),
                    )
                )

        self.assertEqual(client.calls, 0)
        rag_mock.assert_not_called()
        image_mock.assert_not_called()
        self.assertEqual(
            [event["event"] for event in events],
            ["meta", "delta", "done"],
        )
        self.assertEqual(
            set(events[0]["data"]),
            {"success", "mode", "route", "stage", "display_type"},
        )
        self.assertEqual(events[0]["data"]["route"], "guided_tutor")
        self.assertEqual(set(events[1]["data"]), {"text"})
        self.assertEqual(events[1]["data"]["text"], standard_item["answer_markdown"])
        self.assertEqual(
            set(events[2]["data"]),
            {
                "success",
                "mode",
                "route",
                "stage",
                "display_type",
                "answer_markdown",
                "need_student_reply",
                "saved",
                "warning",
            },
        )
        self.assertEqual(
            events[2]["data"]["answer_markdown"],
            standard_item["answer_markdown"],
        )
        self.assertFalse(events[2]["data"]["saved"])

    def test_stream_tutor_workflow_route_returns_plain_markdown(self) -> None:
        agent_core = load_module("agent_core_workflow_stream", PROJECT_DIR / "agent_core.py")
        client = StreamingModelClient(["## 实验记录整理\n", "- 已完成：课前预习。"])
        history = [
            {"role": "user", "content": "我已经完成了迈克尔逊干涉仪的课前预习。"},
            {"role": "assistant", "content": "可以继续整理现场测量步骤。"},
        ]

        events = list(
            agent_core.stream_tutor_events(
                "梳理一下上下文，汇总成实验记录",
                stage="pre_lab",
                conversation_history=history,
                save=False,
                client=client,
                resources=build_test_resources(),
            ),
        )

        self.assertEqual(events[0]["data"]["route"], "experiment_workflow")
        self.assertEqual(events[-1]["event"], "done")
        self.assertEqual(
            events[-1]["data"]["answer_markdown"],
            "## 实验记录整理\n- 已完成：课前预习。",
        )
        self.assertNotIn("### 问题判断", events[-1]["data"]["answer_markdown"])
        self.assertIn("WORKFLOW_PROMPT_MARKER", client.user_prompt)

    def test_stream_tutor_out_of_scope_returns_fixed_message_without_model_call(self) -> None:
        agent_core = load_module("agent_core_scope_stream", PROJECT_DIR / "agent_core.py")
        client = StreamingModelClient(["不应该调用模型"])

        events = list(
            agent_core.stream_tutor_events(
                "推荐一部电影",
                stage="unknown",
                save=False,
                client=client,
                resources=build_test_resources(),
            ),
        )

        self.assertEqual([event["event"] for event in events], ["meta", "delta", "done"])
        self.assertEqual(client.calls, 0)
        self.assertEqual(events[0]["data"]["route"], "out_of_scope")
        self.assertIn("迈克尔逊干涉仪实验", events[1]["data"]["text"])
        self.assertEqual(events[-1]["data"]["answer_markdown"], events[1]["data"]["text"])
        self.assertFalse(events[-1]["data"]["saved"])

    def test_stream_tutor_social_route_returns_meta_delta_done_from_model(self) -> None:
        agent_core = load_module("agent_core_social_stream", PROJECT_DIR / "agent_core.py")
        client = StreamingModelClient(["你好，我是", "迈克尔逊干涉仪实验助教。"])

        events = list(
            agent_core.stream_tutor_events(
                "你好",
                stage="unknown",
                save=False,
                client=client,
                resources=build_test_resources(),
            ),
        )

        self.assertEqual(
            [event["event"] for event in events],
            ["meta", "delta", "delta", "done"],
        )
        self.assertEqual(events[0]["data"]["route"], "social")
        self.assertEqual(client.calls, 1)
        self.assertEqual(events[-1]["data"]["route"], "social")
        self.assertEqual(
            events[-1]["data"]["answer_markdown"],
            "你好，我是迈克尔逊干涉仪实验助教。",
        )
        self.assertIn("TUTOR_PROMPT_MARKER", client.user_prompt)

    def test_stream_tutor_model_error_returns_error_event_without_done(self) -> None:
        agent_core = load_module("agent_core_error_stream", PROJECT_DIR / "agent_core.py")
        client = StreamingModelClient([], error=Exception("stream failed"))

        with patch.object(agent_core.logger, "exception") as log_exception:
            events = list(
                agent_core.stream_tutor_events(
                    "为什么 d = Nλ/2？",
                    stage="pre_lab",
                    save=False,
                    client=client,
                    resources=build_test_resources(),
                ),
            )

        self.assertEqual([event["event"] for event in events], ["meta", "error"])
        self.assertFalse(events[-1]["data"]["success"])
        self.assertIn("error_code", events[-1]["data"])
        self.assertIn("message", events[-1]["data"])
        self.assertEqual(log_exception.call_count, 1)
        extra = log_exception.call_args.kwargs["extra"]
        self.assertEqual(extra["mode"], "tutor")
        self.assertEqual(extra["stage"], "pre_lab")
        self.assertEqual(extra["route"], "guided_tutor")
        self.assertEqual(extra["error_code"], "AGENT_RUNTIME_ERROR")

    def test_stream_report_success_emits_meta_delta_done(self) -> None:
        agent_core = load_module("agent_core_report_stream_success", PROJECT_DIR / "agent_core.py")
        client = StreamingModelClient(["### 总体评价\n", "报告结构清楚。"])

        events = list(
            agent_core.stream_report_events(
                "本实验观察到20条条纹移动。",
                stage="report",
                save=False,
                client=client,
                resources=build_test_resources(),
            ),
        )

        self.assertEqual(
            [event["event"] for event in events],
            ["meta", "delta", "delta", "done"],
        )
        self.assertEqual(events[0]["data"]["mode"], "report_check")
        self.assertEqual(events[0]["data"]["stage"], "report")
        self.assertEqual(events[0]["data"]["display_type"], "markdown")
        self.assertNotIn("route", events[0]["data"])
        self.assertEqual(events[1]["data"]["text"], "### 总体评价\n")
        self.assertEqual(
            events[-1]["data"]["answer_markdown"],
            "### 总体评价\n报告结构清楚。",
        )
        self.assertEqual(events[-1]["data"]["mode"], "report_check")
        self.assertFalse(events[-1]["data"]["saved"])
        self.assertIsNone(events[-1]["data"]["warning"])
        self.assertNotIn("route", events[-1]["data"])
        self.assertNotIn("need_student_reply", events[-1]["data"])
        self.assertEqual(client.calls, 1)
        self.assertIn("REPORT_PROMPT_MARKER", client.user_prompt)
        self.assertIn("本实验观察到20条条纹移动。", client.user_prompt)

    def test_build_report_prompt_conditionally_injects_quiz_prompt_fragments(self) -> None:
        agent_core = load_module("agent_core_report_quiz_prompt", PROJECT_DIR / "agent_core.py")
        self.assertIn("quiz_context", inspect.signature(agent_core.build_report_prompt).parameters)

        quiz_context = "【附录：课前预习习题作答与标准答案】\n第1题：学生答案 A，标准答案 B。"
        for prompt_name in ("report_checker_prompt.txt", "vision_report_prompt.txt"):
            template = (PROJECT_DIR / "prompts" / prompt_name).read_text(encoding="utf-8")
            with self.subTest(prompt=prompt_name):
                baseline = agent_core.build_report_prompt(template, "知识库", "报告正文")
                blank = agent_core.build_report_prompt(
                    template,
                    "知识库",
                    "报告正文",
                    quiz_context="  \n\t",
                )
                with_quiz = agent_core.build_report_prompt(
                    template,
                    "知识库",
                    "报告正文",
                    quiz_context=f"  {quiz_context}  ",
                )

                self.assertEqual(blank, baseline)
                self.assertNotIn("课前预习作答检查", baseline)
                self.assertNotIn("逐题检查作答", baseline)
                self.assertNotIn("key_points", baseline)
                self.assertNotIn("{quiz_review_", baseline)
                self.assertIn(quiz_context, with_quiz)
                self.assertIn("课前预习作答检查", with_quiz)
                self.assertIn("逐题检查作答", with_quiz)
                self.assertIn("key_points", with_quiz)
                self.assertIn("不计入现有 100 分", with_quiz)
                self.assertNotIn("{quiz_review_", with_quiz)

    def test_check_report_injects_quiz_context_without_saving_it(self) -> None:
        agent_core = load_module("agent_core_report_quiz_check", PROJECT_DIR / "agent_core.py")
        self.assertIn("quiz_context", inspect.signature(agent_core.check_report).parameters)
        client = RecordingModelClient("### 总体评价\n报告结构清楚。")
        quiz_context = "【附录：课前预习习题作答与标准答案】\n第1题：学生答案 A，标准答案 B。"

        with patch.object(agent_core, "save_output") as save_call:
            payload = agent_core.check_report(
                "报告正文",
                stage="report",
                save=True,
                client=client,
                resources=build_test_resources(),
                quiz_context=quiz_context,
            )

        self.assertTrue(payload["success"])
        self.assertIn(quiz_context, client.user_prompt)
        saved_content = save_call.call_args.args[0]
        self.assertNotIn(quiz_context, saved_content)

    def test_stream_report_injects_quiz_context_without_changing_events(self) -> None:
        agent_core = load_module("agent_core_report_quiz_stream", PROJECT_DIR / "agent_core.py")
        self.assertIn("quiz_context", inspect.signature(agent_core.stream_report_events).parameters)
        client = StreamingModelClient(["### 总体评价\n", "报告结构清楚。"])
        quiz_context = "【附录：课前预习习题作答与标准答案】\n第1题：学生答案 A，标准答案 B。"

        events = list(
            agent_core.stream_report_events(
                "报告正文",
                stage="report",
                save=False,
                client=client,
                resources=build_test_resources(),
                quiz_context=quiz_context,
            ),
        )

        self.assertEqual(
            [event["event"] for event in events],
            ["meta", "delta", "delta", "done"],
        )
        self.assertIn(quiz_context, client.user_prompt)

    def test_stream_report_uses_reliable_rag_context_without_changing_event_shape(self) -> None:
        agent_core = load_module("agent_core_report_rag_stream", PROJECT_DIR / "agent_core.py")
        client = StreamingModelClient(["### 总体评价\n", "报告结构清楚。"])
        rag_result = {
            "enabled": True,
            "available": True,
            "reliable": True,
            "reason": "high_confidence_match",
            "query": "report 2d=Nλ",
            "results": [
                {
                    "text": "评分细则：报告需要写出公式来源、代入过程和单位。",
                    "score": 0.87,
                    "source_file": "评分细则.md",
                    "section": "计算过程",
                    "task_type": "report_check",
                }
            ],
        }

        with patch.object(agent_core, "retrieve_rag_context", return_value=rag_result):
            events = list(
                agent_core.stream_report_events(
                    "本实验观察到20条条纹移动，2d=Nλ。",
                    stage="report",
                    save=False,
                    client=client,
                    resources=build_test_resources(),
                ),
            )

        self.assertEqual([event["event"] for event in events], ["meta", "delta", "delta", "done"])
        self.assertNotIn("route", events[0]["data"])
        self.assertIn("【基础知识库与固定规则：始终有效】", client.user_prompt)
        self.assertIn("实验知识库", client.user_prompt)
        self.assertIn("评分细则：报告需要写出公式来源", client.user_prompt)

    def test_stream_report_does_not_send_quiz_context_to_rag_query(self) -> None:
        agent_core = load_module("agent_core_report_quiz_rag_privacy", PROJECT_DIR / "agent_core.py")
        client = StreamingModelClient(["### 总体评价\n", "报告结构清楚。"])
        quiz_context = "QUIZ_SECRET_MARKER\n第1题：学生答案 A，标准答案 B。"
        rag_result = {
            "enabled": True,
            "available": True,
            "reliable": False,
            "reason": "low_confidence",
            "query": "report",
            "results": [],
        }

        with patch.object(agent_core, "retrieve_rag_context", return_value=rag_result) as rag_call:
            list(
                agent_core.stream_report_events(
                    "报告正文包含可检索内容。",
                    stage="report",
                    save=False,
                    client=client,
                    resources=build_test_resources(),
                    quiz_context=quiz_context,
                ),
            )

        query = rag_call.call_args.args[0]
        self.assertIn("报告正文包含可检索内容", query)
        self.assertNotIn("QUIZ_SECRET_MARKER", query)
        self.assertIn("QUIZ_SECRET_MARKER", client.user_prompt)

    def test_stream_tutor_images_keep_event_shape_unchanged(self) -> None:
        agent_core = load_module("agent_core_tutor_image_stream_shape", PROJECT_DIR / "agent_core.py")
        client = StreamingModelClient(["### 原理提示\n", "光路解释。"])
        image_path, created = ensure_test_asset_image("michelson_optical_path.jpg")
        try:
            events = list(
                agent_core.stream_tutor_events(
                    "解释迈克尔逊干涉仪光路。",
                    stage="pre_lab",
                    save=False,
                    client=client,
                    resources=build_test_resources(),
                ),
            )
        finally:
            if created:
                image_path.unlink(missing_ok=True)

        self.assertEqual([event["event"] for event in events], ["meta", "delta", "delta", "done"])
        self.assertEqual(set(events[0]["data"].keys()), {"success", "mode", "route", "stage", "display_type"})
        self.assertIn("【可用教学图片】", client.user_prompt)
        self.assertIn("/static/ai_agent/images/michelson_optical_path.jpg", client.user_prompt)
        self.assertIn("answer_markdown", events[-1]["data"])

    def test_stream_report_images_keep_event_shape_unchanged(self) -> None:
        agent_core = load_module("agent_core_report_image_stream_shape", PROJECT_DIR / "agent_core.py")
        client = StreamingModelClient(["### 总体评价\n", "报告结构清楚。"])
        image_path, created = ensure_test_asset_image("michelson_optical_path.jpg")
        try:
            events = list(
                agent_core.stream_report_events(
                    "报告写到 M1 固定镜、M2 动镜以及分光合束光路。",
                    stage="report",
                    save=False,
                    client=client,
                    resources=build_test_resources(),
                ),
            )
        finally:
            if created:
                image_path.unlink(missing_ok=True)

        self.assertEqual([event["event"] for event in events], ["meta", "delta", "delta", "done"])
        self.assertEqual(set(events[0]["data"].keys()), {"success", "mode", "stage", "display_type"})
        self.assertNotIn("route", events[0]["data"])
        self.assertNotIn("route", events[-1]["data"])
        self.assertIn("【可用教学图片】", client.user_prompt)
        self.assertEqual(client.user_prompt.count("/static/ai_agent/images/"), 1)

    def test_stream_report_empty_input_emits_error(self) -> None:
        agent_core = load_module("agent_core_report_stream_empty", PROJECT_DIR / "agent_core.py")

        events = list(agent_core.stream_report_events("   ", stage="report"))

        self.assertEqual([event["event"] for event in events], ["error"])
        self.assertFalse(events[0]["data"]["success"])
        self.assertEqual(events[0]["data"]["error_code"], "VALIDATION_ERROR")
        self.assertEqual(events[0]["data"]["message"], "报告内容不能为空。")

    def test_stream_report_model_error_logs_and_emits_error(self) -> None:
        agent_core = load_module("agent_core_report_stream_error", PROJECT_DIR / "agent_core.py")
        client = StreamingModelClient([], error=agent_core.OpenAIError("stream failed"))

        with patch.object(agent_core.logger, "exception") as log_exception:
            events = list(
                agent_core.stream_report_events(
                    "本实验观察到20条条纹移动。",
                    stage="report",
                    save=False,
                    client=client,
                    resources=build_test_resources(),
                ),
            )

        self.assertEqual([event["event"] for event in events], ["meta", "error"])
        self.assertFalse(events[-1]["data"]["success"])
        self.assertEqual(events[-1]["data"]["error_code"], "MODEL_API_ERROR")
        self.assertEqual(events[-1]["data"]["message"], "AI 服务暂时不可用，请稍后重试。")
        self.assertEqual(log_exception.call_count, 1)
        extra = log_exception.call_args.kwargs["extra"]
        self.assertEqual(extra["mode"], "report_check")
        self.assertEqual(extra["stage"], "report")
        self.assertEqual(extra.get("route"), "unknown")
        self.assertEqual(extra["error_code"], "MODEL_API_ERROR")
        self.assertNotIn("report_content", extra)
        self.assertNotIn("user_prompt", extra)
        self.assertNotIn("knowledge_base", extra)

    def test_build_frontend_payload_contains_frontend_fields(self) -> None:
        agent_core = load_module("agent_core_payload", PROJECT_DIR / "agent_core.py")

        payload = agent_core.build_frontend_payload(
            "tutor",
            "### 问题判断\n这是一个原理问题。",
            {"stage": "pre_lab", "saved": True, "warning": None},
        )

        self.assertEqual(payload["success"], True)
        self.assertEqual(payload["mode"], "tutor")
        self.assertEqual(payload["stage"], "pre_lab")
        self.assertEqual(payload["display_type"], "markdown")
        self.assertEqual(payload["answer_markdown"], "### 问题判断\n这是一个原理问题。")
        self.assertTrue(payload["need_student_reply"])
        self.assertTrue(payload["saved"])
        self.assertIsNone(payload["warning"])

    def test_api_server_defines_required_routes(self) -> None:
        source = (PROJECT_DIR / "api_server.py").read_text(encoding="utf-8")

        self.assertIn('@app.get("/health")', source)
        self.assertIn('@app.post("/api/agent/chat")', source)
        self.assertIn('@app.post("/api/agent/chat/stream")', source)
        self.assertIn('@app.post("/api/agent/report-check")', source)
        self.assertIn('"Content-Type": "text/event-stream; charset=utf-8"', source)
        self.assertIn('"Cache-Control": "no-cache"', source)
        self.assertIn('"Connection": "keep-alive"', source)
        self.assertIn('"X-Accel-Buffering": "no"', source)
        self.assertIn("AI_AGENT_LOG_LEVEL", source)
        self.assertIn("logging.basicConfig", source)
        self.assertIn("logger.exception", source)

    def test_api_server_exposes_report_check_stream_endpoint(self) -> None:
        source = (PROJECT_DIR / "api_server.py").read_text(encoding="utf-8")

        self.assertIn('@app.post("/api/agent/report-check/stream")', source)
        self.assertIn("stream_report_events", source)
        self.assertIn("StreamingResponse", source)
        self.assertIn("format_sse_event", source)

    def test_api_server_serves_static_teaching_images(self) -> None:
        api_server = load_api_server_module("api_server_static_images")
        client = TestClient(api_server.app)
        image_path, created = ensure_test_asset_image("michelson_optical_path.jpg")
        old_filenames = [
            "michelson_optical_path.png",
            "instrument_structure_labeled.png",
            "compensation_plate.png",
            "equal_inclination_rings.png",
            "fringe_in_out.png",
            "micrometer_reading.png",
        ]
        try:
            response = client.get("/static/ai_agent/images/michelson_optical_path.jpg")
            old_responses = [
                client.get(f"/static/ai_agent/images/{filename}")
                for filename in old_filenames
            ]
        finally:
            if created:
                image_path.unlink(missing_ok=True)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/jpeg")
        with Image.open(BytesIO(response.content)) as served_image:
            self.assertEqual(served_image.format, "JPEG")
            served_image.verify()
        self.assertTrue(all(item.status_code == 404 for item in old_responses))

    def test_api_server_loads_dotenv_at_module_level(self) -> None:
        source = (PROJECT_DIR / "api_server.py").read_text(encoding="utf-8")

        self.assertIn("from dotenv import load_dotenv", source)
        self.assertGreaterEqual(source.count("load_dotenv(BASE_DIR"), 2)

    def test_report_check_upload_success_returns_document_info(self) -> None:
        api_server = load_api_server_module("api_server_upload_success")
        client = TestClient(api_server.app)
        pdf_text = "michelson interferometer report " * 12
        fixed_payload = {
            "success": True,
            "mode": "report_check",
            "stage": "report",
            "display_type": "markdown",
            "answer_markdown": "### 总体评价\n报告结构清楚。",
            "saved": True,
            "warning": None,
        }

        with patch.object(api_server, "check_report", return_value=fixed_payload) as check_call:
            response = client.post(
                "/api/agent/report-check/upload",
                data={"session_id": "student_001", "stage": "report"},
                files={
                    "file": (
                        "report.pdf",
                        build_text_pdf(pdf_text),
                        "application/pdf",
                    )
                },
            )

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload["success"])
        self.assertEqual(payload["document_info"]["page_count"], 1)
        self.assertGreaterEqual(payload["document_info"]["extracted_text_chars"], 200)
        self.assertEqual(payload["document_info"]["embedded_image_count"], 0)
        self.assertFalse(payload["document_info"]["suspected_scanned"])
        self.assertIn("PDF 共", payload["warning"])
        self.assertIn("改用 /api/agent/report-check/vision", payload["warning"])
        check_call.assert_called_once()
        self.assertEqual(check_call.call_args.kwargs["report_content"], pdf_text.strip())
        self.assertEqual(check_call.call_args.kwargs["stage"], "report")
        self.assertTrue(check_call.call_args.kwargs["save"])
        self.assertFalse(check_call.call_args.kwargs["stream"])

    def test_report_check_upload_rejects_non_pdf_mime(self) -> None:
        api_server = load_api_server_module("api_server_upload_non_pdf")
        client = TestClient(api_server.app)

        with patch.object(api_server, "check_report") as check_call:
            response = client.post(
                "/api/agent/report-check/upload",
                data={"session_id": "student_001", "stage": "report"},
                files={"file": ("report.png", b"not a pdf", "image/png")},
            )

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertFalse(payload["success"])
        self.assertEqual(payload["error_code"], "VALIDATION_ERROR")
        check_call.assert_not_called()

    def test_report_check_upload_rejects_oversize(self) -> None:
        api_server = load_api_server_module("api_server_upload_oversize")
        client = TestClient(api_server.app)

        with patch.object(api_server, "check_report") as check_call:
            response = client.post(
                "/api/agent/report-check/upload",
                data={"session_id": "student_001", "stage": "report"},
                files={
                    "file": (
                        "report.pdf",
                        b"x" * (MAX_PDF_SIZE_BYTES + 1),
                        "application/pdf",
                    )
                },
            )

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertFalse(payload["success"])
        self.assertEqual(payload["error_code"], "VALIDATION_ERROR")
        check_call.assert_not_called()

    def test_report_check_upload_detects_scanned_pdf(self) -> None:
        api_server = load_api_server_module("api_server_upload_scanned")
        client = TestClient(api_server.app)

        with patch.object(api_server, "check_report") as check_call:
            response = client.post(
                "/api/agent/report-check/upload",
                data={"session_id": "student_001", "stage": "report"},
                files={
                    "file": (
                        "report.pdf",
                        build_text_pdf("short report"),
                        "application/pdf",
                    )
                },
            )

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertFalse(payload["success"])
        self.assertEqual(payload["error_code"], "SCANNED_NOT_SUPPORTED")
        self.assertIn("/api/agent/report-check/vision", payload["message"])
        self.assertEqual(payload["message"], payload["warning"])
        self.assertTrue(payload["document_info"]["suspected_scanned"])
        check_call.assert_not_called()

    def test_report_check_upload_corrupt_pdf_returns_parse_error(self) -> None:
        api_server = load_api_server_module("api_server_upload_corrupt")
        client = TestClient(api_server.app)

        with patch.object(api_server, "check_report") as check_call:
            with patch("pypdf._reader.logger_warning"):
                with patch.object(api_server.logger, "exception"):
                    response = client.post(
                        "/api/agent/report-check/upload",
                        data={"session_id": "student_001", "stage": "report"},
                        files={
                            "file": (
                                "report.pdf",
                                b"not a pdf",
                                "application/pdf",
                            )
                        },
                    )

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertFalse(payload["success"])
        self.assertEqual(payload["error_code"], "DOCUMENT_PARSE_ERROR")
        check_call.assert_not_called()

    def test_report_check_upload_logs_metadata_without_leaking_text(self) -> None:
        api_server = load_api_server_module("api_server_upload_logging")
        client = TestClient(api_server.app)
        fixed_payload = {
            "success": True,
            "mode": "report_check",
            "stage": "report",
            "display_type": "markdown",
            "answer_markdown": "### 总体评价\n报告结构清楚。",
            "saved": True,
            "warning": None,
        }

        with patch.object(api_server, "check_report", return_value=fixed_payload):
            with patch.object(api_server.logger, "info") as log_info:
                response = client.post(
                    "/api/agent/report-check/upload",
                    data={"session_id": "student_001", "stage": "report"},
                    files={
                        "file": (
                            "report.pdf",
                            build_text_pdf("michelson interferometer report " * 12),
                            "application/pdf",
                        )
                    },
                )

        self.assertEqual(response.status_code, 200)
        self.assertGreaterEqual(log_info.call_count, 1)
        extra = log_info.call_args.kwargs["extra"]
        self.assertEqual(extra["mode"], "report_check")
        self.assertEqual(extra["stage"], "report")
        self.assertEqual(extra["route"], "upload")
        self.assertIn("error_code", extra)
        self.assertIn("page_count", extra)
        self.assertIn("embedded_image_count", extra)
        self.assertIn("duration_ms", extra)
        self.assertNotIn("extracted_text", extra)
        self.assertNotIn("report_content", extra)
        self.assertNotIn("file", extra)
        self.assertNotIn("pdf_bytes", extra)
        self.assertNotIn("session_id", extra)

    def test_existing_report_check_endpoint_accepts_quiz_context_without_response_change(self) -> None:
        api_server = load_api_server_module("api_server_report_check_quiz_context")
        client = TestClient(api_server.app)
        quiz_context = "【附录：课前预习习题作答与标准答案】\n第1题：学生答案 A，标准答案 B。"
        fixed_payload = {
            "success": True,
            "mode": "report_check",
            "stage": "report",
            "display_type": "markdown",
            "answer_markdown": "### 总体评价\n报告结构清楚。",
            "saved": True,
            "warning": None,
        }

        with patch.object(api_server, "check_report", return_value=fixed_payload) as check_call:
            response = client.post(
                "/api/agent/report-check",
                json={
                    "session_id": "student_001",
                    "stage": "report",
                    "report_content": "报告片段",
                    "quiz_context": quiz_context,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), fixed_payload)
        check_call.assert_called_once()
        self.assertEqual(check_call.call_args.kwargs["report_content"], "报告片段")
        self.assertIn("quiz_context", check_call.call_args.kwargs)
        self.assertEqual(check_call.call_args.kwargs["quiz_context"], quiz_context)

    def test_report_check_stream_forwards_quiz_context(self) -> None:
        api_server = load_api_server_module("api_server_report_check_stream_quiz_context")
        client = TestClient(api_server.app)
        quiz_context = "【附录：课前预习习题作答与标准答案】\n第1题：学生答案 A，标准答案 B。"
        events = iter([{"event": "done", "data": {"success": True}}])

        with patch.object(api_server, "stream_report_events", return_value=events) as stream_call:
            response = client.post(
                "/api/agent/report-check/stream",
                json={
                    "session_id": "student_001",
                    "stage": "report",
                    "report_content": "报告片段",
                    "quiz_context": quiz_context,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("event: done", response.text)
        self.assertIn("quiz_context", stream_call.call_args.kwargs)
        self.assertEqual(stream_call.call_args.kwargs["quiz_context"], quiz_context)

    def test_report_check_accepts_max_quiz_context_and_rejects_over_limit(self) -> None:
        api_server = load_api_server_module("api_server_report_check_quiz_context_limit")
        client = TestClient(api_server.app)
        fixed_payload = {
            "success": True,
            "mode": "report_check",
            "stage": "report",
            "display_type": "markdown",
            "answer_markdown": "### 总体评价\n报告结构清楚。",
            "saved": True,
            "warning": None,
        }

        with patch.object(api_server, "check_report", return_value=fixed_payload) as check_call:
            accepted = client.post(
                "/api/agent/report-check",
                json={
                    "session_id": "student_001",
                    "stage": "report",
                    "report_content": "报告片段",
                    "quiz_context": "x" * api_server.MAX_QUIZ_CONTEXT_CHARS,
                },
            )
            rejected = client.post(
                "/api/agent/report-check",
                json={
                    "session_id": "student_001",
                    "stage": "report",
                    "report_content": "报告片段",
                    "quiz_context": "x" * (api_server.MAX_QUIZ_CONTEXT_CHARS + 1),
                },
            )

        self.assertEqual(accepted.status_code, 200)
        self.assertEqual(rejected.status_code, 422)
        check_call.assert_called_once()

    def test_report_check_stream_rejects_over_limit_quiz_context_without_model_call(self) -> None:
        api_server = load_api_server_module("api_server_report_check_stream_quiz_context_limit")
        client = TestClient(api_server.app)

        with patch.object(api_server, "stream_report_events") as stream_call:
            response = client.post(
                "/api/agent/report-check/stream",
                json={
                    "session_id": "student_001",
                    "stage": "report",
                    "report_content": "报告片段",
                    "quiz_context": "x" * (api_server.MAX_QUIZ_CONTEXT_CHARS + 1),
                },
            )

        self.assertEqual(response.status_code, 422)
        stream_call.assert_not_called()

    def test_api_server_formats_sse_events_with_blank_line_separator(self) -> None:
        agent_core = load_module("agent_core_sse_format", PROJECT_DIR / "agent_core.py")

        event_text = agent_core.format_sse_event("delta", {"text": "### 问题判断\n"})

        self.assertTrue(event_text.startswith("event: delta\n"))
        self.assertIn('\ndata: {"text":', event_text)
        self.assertTrue(event_text.endswith("\n\n"))

    def test_api_request_models_have_expected_fields(self) -> None:
        source = (PROJECT_DIR / "api_server.py").read_text(encoding="utf-8")
        tree = ast.parse(source)

        fields_by_model: dict[str, set[str]] = {}
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name in {
                "ChatRequest",
                "ReportCheckRequest",
            }:
                fields_by_model[node.name] = {
                    item.target.id
                    for item in node.body
                    if isinstance(item, ast.AnnAssign)
                    and isinstance(item.target, ast.Name)
                }

        self.assertEqual(
            fields_by_model["ChatRequest"],
            {
                "session_id",
                "stage",
                "student_input",
                "conversation_history",
                "teaching_policy",
                "guidance_level",
            },
        )
        self.assertEqual(
            fields_by_model["ReportCheckRequest"],
            {"session_id", "stage", "report_content", "quiz_context"},
        )

    def test_chat_api_passes_teaching_overrides_to_agent_core(self) -> None:
        api_server = load_api_server_module("api_server_chat_teaching_overrides")
        client = TestClient(api_server.app)
        fixed_payload = {
            "success": True,
            "mode": "tutor",
            "stage": "pre_lab",
            "display_type": "markdown",
            "answer_markdown": "answer",
            "saved": True,
            "warning": None,
        }

        with patch.object(api_server, "ask_tutor", return_value=fixed_payload) as ask_call:
            response = client.post(
                "/api/agent/chat",
                json={
                    "session_id": "student_001",
                    "stage": "pre_lab",
                    "student_input": "分束器是什么？",
                    "conversation_history": [],
                    "teaching_policy": "guided",
                    "guidance_level": 3,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), fixed_payload)
        ask_call.assert_called_once()
        self.assertEqual(ask_call.call_args.kwargs["teaching_policy"], "guided")
        self.assertEqual(ask_call.call_args.kwargs["guidance_level"], 3)

    def test_chat_stream_api_passes_teaching_overrides_to_agent_core(self) -> None:
        api_server = load_api_server_module("api_server_chat_stream_teaching_overrides")
        client = TestClient(api_server.app)
        events = iter(
            [
                {
                    "event": "done",
                    "data": {
                        "success": True,
                        "mode": "tutor",
                        "route": "guided_tutor",
                        "stage": "pre_lab",
                        "display_type": "markdown",
                        "answer_markdown": "answer",
                        "need_student_reply": True,
                        "saved": True,
                        "warning": None,
                    },
                }
            ]
        )

        with patch.object(api_server, "stream_tutor_events", return_value=events) as stream_call:
            response = client.post(
                "/api/agent/chat/stream",
                json={
                    "session_id": "student_001",
                    "stage": "pre_lab",
                    "student_input": "分束器是什么？",
                    "conversation_history": [],
                    "teaching_policy": "hybrid",
                    "guidance_level": 2,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("event: done", response.text)
        stream_call.assert_called_once()
        self.assertEqual(stream_call.call_args.kwargs["teaching_policy"], "hybrid")
        self.assertEqual(stream_call.call_args.kwargs["guidance_level"], 2)

    def test_report_prompt_templates_use_conditional_quiz_placeholders(self) -> None:
        for prompt_name in ("report_checker_prompt.txt", "vision_report_prompt.txt"):
            prompt = (PROJECT_DIR / "prompts" / prompt_name).read_text(encoding="utf-8")
            with self.subTest(prompt=prompt_name):
                self.assertIn("{quiz_review_rules}", prompt)
                self.assertIn("{quiz_review_scoring_note}", prompt)
                self.assertIn("{quiz_review_output_section}", prompt)
                self.assertNotIn("课前预习作答检查", prompt)
                self.assertNotIn("不计入现有 100 分", prompt)

    def test_vision_enabled_picks_up_env_after_module_load(self) -> None:
        api_server = load_api_server_module("vision_enabled_env_test")

        with patch.dict(os.environ, {"ENABLE_VISION": "true"}, clear=False):
            self.assertTrue(api_server.vision_enabled())

        with patch.dict(os.environ, {"ENABLE_VISION": "false"}, clear=False):
            self.assertFalse(api_server.vision_enabled())

    def test_chat_vision_stream_success_emits_doc13_events(self) -> None:
        api_server = load_api_server_module("api_server_chat_vision_stream_success")
        client = TestClient(api_server.app)
        streaming_client = StreamingModelClient(["第一段", "第二段"])

        with patch.dict(os.environ, {"ENABLE_VISION": "true"}, clear=False):
            with patch.object(api_server, "create_vision_client", return_value=streaming_client):
                with patch.object(api_server, "save_output") as save_call:
                    response = client.post(
                        "/api/agent/chat/vision/stream",
                        data={
                            "session_id": "student_001",
                            "stage": "pre_lab",
                            "student_input": "请看图解释条纹。",
                            "conversation_history": "[]",
                        },
                        files={"image": ("stripe.png", b"png-bytes", "image/png")},
                    )

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])
        body = response.text
        self.assertIn("event: meta", body)
        self.assertIn('"mode": "vision_chat"', body)
        self.assertIn('"route": "single_image_vision"', body)
        self.assertIn('"stage": "generation"', body)
        self.assertIn("event: delta", body)
        self.assertIn('data: {"text": "第一段"}', body)
        self.assertIn("event: done", body)
        self.assertIn('"answer_markdown": "第一段第二段"', body)
        self.assertLess(body.index("event: meta"), body.index("event: delta"))
        self.assertLess(body.index("event: delta"), body.index("event: done"))
        save_call.assert_called_once()

    def test_chat_vision_stream_accepts_image_above_non_stream_limit(self) -> None:
        api_server = load_api_server_module("api_server_chat_vision_stream_large_image")
        client = TestClient(api_server.app)
        streaming_client = StreamingModelClient(["大图回答"])
        image_bytes = b"x" * (5 * 1024 * 1024 + 1)

        with patch.dict(os.environ, {"ENABLE_VISION": "true"}, clear=False):
            with patch.object(api_server, "create_vision_client", return_value=streaming_client):
                with patch.object(api_server, "save_output") as save_call:
                    response = client.post(
                        "/api/agent/chat/vision/stream",
                        data={
                            "session_id": "student_001",
                            "stage": "pre_lab",
                            "student_input": "请看大图解释条纹。",
                            "conversation_history": "[]",
                        },
                        files={"image": ("large-stripe.png", image_bytes, "image/png")},
                    )

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])
        self.assertIn("event: done", response.text)
        save_call.assert_called_once()

    def test_chat_vision_stream_rejects_image_above_stream_limit(self) -> None:
        api_server = load_api_server_module("api_server_chat_vision_stream_too_large_image")
        client = TestClient(api_server.app)
        image_bytes = b"x" * (STREAM_IMAGE_LIMIT_BYTES + 1)

        with patch.dict(os.environ, {"ENABLE_VISION": "true"}, clear=False):
            with patch.object(api_server, "create_vision_client") as create_call:
                response = client.post(
                    "/api/agent/chat/vision/stream",
                    data={
                        "session_id": "student_001",
                        "stage": "pre_lab",
                        "student_input": "请看大图解释条纹。",
                        "conversation_history": "[]",
                    },
                    files={"image": ("too-large-stripe.png", image_bytes, "image/png")},
                )

        payload = response.json()
        self.assertFalse(payload["success"])
        self.assertEqual(payload["error_code"], "IMAGE_TOO_LARGE")
        self.assertIn("15MB", payload["message"])
        create_call.assert_not_called()

    def test_chat_vision_stream_model_error_emits_error_without_done_or_save(self) -> None:
        api_server = load_api_server_module("api_server_chat_vision_stream_error")
        client = TestClient(api_server.app)
        streaming_client = StreamingModelClient([], error=RuntimeError("boom"))

        with patch.dict(os.environ, {"ENABLE_VISION": "true"}, clear=False):
            with patch.object(api_server, "create_vision_client", return_value=streaming_client):
                with patch.object(api_server, "save_output") as save_call:
                    response = client.post(
                        "/api/agent/chat/vision/stream",
                        data={
                            "session_id": "student_001",
                            "stage": "pre_lab",
                            "student_input": "请看图解释条纹。",
                            "conversation_history": "[]",
                        },
                        files={"image": ("stripe.png", b"png-bytes", "image/png")},
                    )

        self.assertEqual(response.status_code, 200)
        body = response.text
        self.assertIn("event: error", body)
        self.assertIn('"error_code": "VISION_API_ERROR"', body)
        self.assertNotIn("event: done", body)
        save_call.assert_not_called()
    def test_chat_vision_stream_save_failure_emits_error_without_done(self) -> None:
        api_server = load_api_server_module("api_server_chat_vision_stream_save_error")
        client = TestClient(api_server.app)
        streaming_client = StreamingModelClient(["完整回答"])

        with patch.dict(os.environ, {"ENABLE_VISION": "true"}, clear=False):
            with patch.object(api_server, "create_vision_client", return_value=streaming_client):
                with patch.object(api_server, "save_output", side_effect=RuntimeError("disk full")):
                    response = client.post(
                        "/api/agent/chat/vision/stream",
                        data={
                            "session_id": "student_001",
                            "stage": "pre_lab",
                            "student_input": "请看图解释条纹。",
                            "conversation_history": "[]",
                        },
                        files={"image": ("stripe.png", b"png-bytes", "image/png")},
                    )

        self.assertEqual(response.status_code, 200)
        body = response.text
        self.assertIn("event: error", body)
        self.assertIn('"error_code": "VISION_API_ERROR"', body)
        self.assertNotIn("event: done", body)
    def test_chat_vision_stream_cancel_closes_upstream_without_save(self) -> None:
        api_server = load_api_server_module("api_server_chat_vision_stream_cancel")
        streaming_client = StreamingModelClient(["第一段", "第二段"])

        with patch.object(api_server, "create_vision_client", return_value=streaming_client):
            with patch.object(api_server, "save_output") as save_call:
                events = api_server.iter_chat_vision_stream_events(
                    stage="pre_lab",
                    student_input="请看图解释条纹。",
                    system_prompt="system prompt",
                    user_prompt="user prompt",
                    image=VisionImage(bytes_data=b"png-bytes", mime="image/png"),
                    vision_info={
                        "image_count": 1,
                        "image_format": "image/png",
                        "image_bytes": len(b"png-bytes"),
                    },
                    started_at=0.0,
                )
                self.assertEqual(next(events)["event"], "meta")
                self.assertEqual(next(events)["event"], "delta")
                events.close()

        self.assertTrue(streaming_client.last_response.closed)
        save_call.assert_not_called()
    def test_chat_vision_stream_allows_zero_delta_before_done(self) -> None:
        api_server = load_api_server_module("api_server_chat_vision_stream_zero_delta")
        client = TestClient(api_server.app)
        streaming_client = StreamingModelClient([])

        with patch.dict(os.environ, {"ENABLE_VISION": "true"}, clear=False):
            with patch.object(api_server, "create_vision_client", return_value=streaming_client):
                with patch.object(api_server, "save_output") as save_call:
                    response = client.post(
                        "/api/agent/chat/vision/stream",
                        data={
                            "session_id": "student_001",
                            "stage": "pre_lab",
                            "student_input": "请看图解释条纹。",
                            "conversation_history": "[]",
                        },
                        files={"image": ("stripe.png", b"png-bytes", "image/png")},
                    )

        self.assertEqual(response.status_code, 200)
        body = response.text
        self.assertIn("event: meta", body)
        self.assertNotIn("event: delta", body)
        self.assertIn("event: done", body)
        self.assertIn('"answer_markdown": ""', body)
        save_call.assert_called_once()
    def test_chat_vision_disabled_returns_404(self) -> None:
        with patch.dict(os.environ, {"ENABLE_VISION": "false"}, clear=False):
            api_server = load_api_server_module("api_server_chat_vision_disabled")
            client = TestClient(api_server.app)
            response = client.post(
                "/api/agent/chat/vision",
                data={
                    "session_id": "student_001",
                    "stage": "pre_lab",
                    "student_input": "这张图里的条纹正常吗？",
                },
                files={"image": ("fringe.png", b"png-bytes", "image/png")},
            )

        payload = response.json()
        self.assertEqual(response.status_code, 404)
        self.assertFalse(payload["success"])
        self.assertEqual(payload["error_code"], "VISION_DISABLED")
        self.assertEqual(payload["route"], "vision_tutor")

    def test_chat_vision_success_returns_doc12_vision_info(self) -> None:
        api_server = load_api_server_module("api_server_chat_vision_success")
        client = TestClient(api_server.app)

        with patch.dict(os.environ, {"ENABLE_VISION": "true"}, clear=False):
            with patch.object(api_server, "create_vision_client", return_value=object()):
                with patch.object(api_server, "call_vision_model", return_value="### 原理提示\n看图回答") as call:
                    with patch.object(api_server, "save_output") as save:
                        response = client.post(
                            "/api/agent/chat/vision",
                            data={
                                "session_id": "student_001",
                                "stage": "pre_lab",
                                "student_input": "这张图里的条纹正常吗？",
                                "conversation_history": json.dumps(
                                    [{"role": "user", "content": "上一轮"}],
                                    ensure_ascii=False,
                                ),
                            },
                            files={"image": ("fringe.png", b"png-bytes", "image/png")},
                        )

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload["success"])
        self.assertEqual(payload["route"], "vision_tutor")
        self.assertTrue(payload["saved"])
        self.assertEqual(
            set(payload["vision_info"]),
            {"image_count", "image_format", "image_bytes"},
        )
        self.assertEqual(payload["vision_info"]["image_count"], 1)
        self.assertEqual(payload["vision_info"]["image_format"], "image/png")
        self.assertEqual(payload["vision_info"]["image_bytes"], len(b"png-bytes"))
        self.assertNotIn("image_source", payload["vision_info"])
        call.assert_called_once()
        self.assertIn("当前处于课前预习阶段", call.call_args.args[2])
        self.assertIn('"stage": "pre_lab"', call.call_args.args[2])
        save.assert_called_once()
        self.assertEqual(save.call_args.args[1], "tutor_chat")

    def test_chat_vision_injects_hybrid_teaching_strategy_for_onsite_diagnosis(self) -> None:
        api_server = load_api_server_module("api_server_chat_vision_hybrid")
        client = TestClient(api_server.app)

        with patch.dict(os.environ, {"ENABLE_VISION": "true"}, clear=False):
            with patch.object(api_server, "create_vision_client", return_value=object()):
                with patch.object(api_server, "call_vision_model", return_value="先检查背景光。") as call:
                    with patch.object(api_server, "save_output"):
                        response = client.post(
                            "/api/agent/chat/vision",
                            data={
                                "session_id": "student_001",
                                "stage": "during_experiment",
                                "student_input": "现场看不到条纹，下一步怎么做？",
                                "conversation_history": "[]",
                            },
                            files={"image": ("fringe.png", b"png-bytes", "image/png")},
                        )

        self.assertTrue(response.json()["success"])
        prompt = call.call_args.args[2]
        self.assertIn("本轮采用现场混合策略", prompt)
        self.assertIn('"teaching_policy": "hybrid"', prompt)
        self.assertIn('"guidance_level": 1', prompt)

    def test_chat_vision_stream_injects_guided_strategy_without_changing_events(self) -> None:
        api_server = load_api_server_module("api_server_chat_vision_stream_guided")
        client = TestClient(api_server.app)
        streaming_client = StreamingModelClient(["先区分去程和回程。"])

        with patch.dict(os.environ, {"ENABLE_VISION": "true"}, clear=False):
            with patch.object(api_server, "create_vision_client", return_value=streaming_client):
                with patch.object(api_server, "save_output"):
                    response = client.post(
                        "/api/agent/chat/vision/stream",
                        data={
                            "session_id": "student_001",
                            "stage": "pre_lab",
                            "student_input": "为什么图中的光程变化是 2d？",
                            "conversation_history": "[]",
                        },
                        files={"image": ("fringe.png", b"png-bytes", "image/png")},
                    )

        self.assertIn("event: meta", response.text)
        self.assertIn("event: delta", response.text)
        self.assertIn("event: done", response.text)
        vision_prompt = streaming_client.user_prompt[-1]["text"]
        self.assertIn("本轮采用多轮引导策略", vision_prompt)
        self.assertIn('"teaching_policy": "guided"', vision_prompt)

    def test_chat_vision_stream_accepts_form_teaching_overrides(self) -> None:
        api_server = load_api_server_module("api_server_chat_vision_stream_teaching_overrides")
        client = TestClient(api_server.app)
        streaming_client = StreamingModelClient(["guided image answer"])

        with patch.dict(os.environ, {"ENABLE_VISION": "true"}, clear=False):
            with patch.object(api_server, "create_vision_client", return_value=streaming_client):
                with patch.object(api_server, "save_output"):
                    response = client.post(
                        "/api/agent/chat/vision/stream",
                        data={
                            "session_id": "student_001",
                            "stage": "pre_lab",
                            "student_input": "分束器是什么？",
                            "conversation_history": "[]",
                            "teaching_policy": "guided",
                            "guidance_level": "3",
                        },
                        files={"image": ("fringe.png", b"png-bytes", "image/png")},
                    )

        self.assertIn("event: done", response.text)
        vision_prompt = streaming_client.user_prompt[-1]["text"]
        self.assertIn('"teaching_policy": "guided"', vision_prompt)
        self.assertIn('"guidance_level": 3', vision_prompt)

    def test_chat_vision_rejects_empty_student_input(self) -> None:
        api_server = load_api_server_module("api_server_chat_vision_empty_input")
        client = TestClient(api_server.app)

        with patch.dict(os.environ, {"ENABLE_VISION": "true"}, clear=False):
            with patch.object(api_server, "call_vision_model") as call:
                response = client.post(
                    "/api/agent/chat/vision",
                    data={"session_id": "student_001", "stage": "pre_lab", "student_input": "   "},
                    files={"image": ("fringe.png", b"png-bytes", "image/png")},
                )

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertFalse(payload["success"])
        self.assertEqual(payload["error_code"], "VALIDATION_ERROR")
        self.assertEqual(payload["message"], "问题内容不能为空。")
        call.assert_not_called()

    def test_chat_vision_rejects_unsupported_image_mime(self) -> None:
        api_server = load_api_server_module("api_server_chat_vision_bad_mime")
        client = TestClient(api_server.app)

        with patch.dict(os.environ, {"ENABLE_VISION": "true"}, clear=False):
            with patch.object(api_server, "call_vision_model") as call:
                response = client.post(
                    "/api/agent/chat/vision",
                    data={
                        "session_id": "student_001",
                        "stage": "pre_lab",
                        "student_input": "看图",
                    },
                    files={"image": ("fringe.gif", b"gif-bytes", "image/gif")},
                )

        payload = response.json()
        self.assertFalse(payload["success"])
        self.assertEqual(payload["error_code"], "IMAGE_FORMAT_UNSUPPORTED")
        call.assert_not_called()

    def test_chat_vision_rejects_oversize_image(self) -> None:
        api_server = load_api_server_module("api_server_chat_vision_oversize")
        client = TestClient(api_server.app)

        with patch.dict(os.environ, {"ENABLE_VISION": "true"}, clear=False):
            with patch.object(api_server, "call_vision_model") as call:
                response = client.post(
                    "/api/agent/chat/vision",
                    data={
                        "session_id": "student_001",
                        "stage": "pre_lab",
                        "student_input": "看图",
                    },
                    files={"image": ("fringe.png", b"x" * (5 * 1024 * 1024 + 1), "image/png")},
                )

        payload = response.json()
        self.assertFalse(payload["success"])
        self.assertEqual(payload["error_code"], "IMAGE_TOO_LARGE")
        call.assert_not_called()

    def test_chat_vision_bad_history_uses_empty_history_with_warning(self) -> None:
        api_server = load_api_server_module("api_server_chat_vision_bad_history")
        client = TestClient(api_server.app)

        with patch.dict(os.environ, {"ENABLE_VISION": "true"}, clear=False):
            with patch.object(api_server, "create_vision_client", return_value=object()):
                with patch.object(api_server, "call_vision_model", return_value="回答") as call:
                    response = client.post(
                        "/api/agent/chat/vision",
                        data={
                            "session_id": "student_001",
                            "stage": "pre_lab",
                            "student_input": "看图",
                            "conversation_history": "{bad json",
                        },
                        files={"image": ("fringe.png", b"png-bytes", "image/png")},
                    )

        payload = response.json()
        self.assertTrue(payload["success"])
        self.assertIn("历史对话格式", payload["warning"])
        call.assert_called_once()

    def test_chat_vision_logs_metadata_without_leaking_input(self) -> None:
        api_server = load_api_server_module("api_server_chat_vision_logging")
        client = TestClient(api_server.app)

        with patch.dict(os.environ, {"ENABLE_VISION": "true"}, clear=False):
            with patch.object(api_server, "create_vision_client", return_value=object()):
                with patch.object(api_server, "call_vision_model", return_value="回答"):
                    with patch.object(api_server.logger, "info") as log_info:
                        response = client.post(
                            "/api/agent/chat/vision",
                            data={
                                "session_id": "student_secret",
                                "stage": "pre_lab",
                                "student_input": "不要出现在日志里",
                            },
                            files={"image": ("fringe.png", b"png-bytes", "image/png")},
                        )

        self.assertEqual(response.status_code, 200)
        extra = log_info.call_args.kwargs["extra"]
        self.assertEqual(extra["mode"], "tutor")
        self.assertEqual(extra["route"], "vision_tutor")
        self.assertIn("image_format", extra)
        self.assertEqual(extra["image_bytes_total"], len(b"png-bytes"))
        self.assertNotIn("student_input", extra)
        self.assertNotIn("session_id", extra)
        self.assertNotIn("image", extra)
        self.assertNotIn("base64", extra)

    def test_chat_vision_model_error_returns_vision_api_error(self) -> None:
        api_server = load_api_server_module("api_server_chat_vision_model_error")
        client = TestClient(api_server.app)

        with patch.dict(os.environ, {"ENABLE_VISION": "true"}, clear=False):
            with patch.object(api_server, "create_vision_client", return_value=object()):
                with patch.object(api_server, "call_vision_model", side_effect=RuntimeError("boom")):
                    response = client.post(
                        "/api/agent/chat/vision",
                        data={
                            "session_id": "student_001",
                            "stage": "pre_lab",
                            "student_input": "看图",
                        },
                        files={"image": ("fringe.png", b"png-bytes", "image/png")},
                    )

        payload = response.json()
        self.assertFalse(payload["success"])
        self.assertEqual(payload["error_code"], "VISION_API_ERROR")

    def test_report_check_vision_stream_image_success_emits_doc13_events(self) -> None:
        api_server = load_api_server_module("api_server_report_vision_stream_image_success")
        client = TestClient(api_server.app)
        streaming_client = StreamingModelClient(["### 总体评价\n", "图片报告可读"])
        quiz_context = "【附录：课前预习习题作答与标准答案】\n第1题：学生答案 A，标准答案 B。"

        with patch.dict(os.environ, {"ENABLE_VISION": "true"}, clear=False):
            with patch.object(api_server, "create_vision_client", return_value=streaming_client):
                with patch.object(api_server, "save_output") as save_call:
                    response = client.post(
                        "/api/agent/report-check/vision/stream",
                        data={
                            "session_id": "student_001",
                            "stage": "report",
                            "quiz_context": quiz_context,
                        },
                        files={"file": ("report.png", b"png-bytes", "image/png")},
                    )

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])
        body = response.text
        self.assertIn("event: meta", body)
        self.assertIn('"mode": "report_check_vision"', body)
        self.assertIn('"route": "report_image_vision"', body)
        self.assertIn('"stage": "generation"', body)
        self.assertIn("event: delta", body)
        self.assertIn("event: done", body)
        self.assertIn('"answer_markdown": "### 总体评价\\n图片报告可读"', body)
        prompt_text = next(
            item["text"]
            for item in streaming_client.user_prompt
            if item["type"] == "text"
        )
        self.assertIn(quiz_context, prompt_text)
        save_call.assert_called_once()

    def test_report_check_vision_stream_accepts_image_above_non_stream_limit(self) -> None:
        api_server = load_api_server_module("api_server_report_vision_stream_large_image")
        client = TestClient(api_server.app)
        streaming_client = StreamingModelClient(["### 总体评价\n", "大图报告可读"])
        image_bytes = b"x" * (5 * 1024 * 1024 + 1)

        with patch.dict(os.environ, {"ENABLE_VISION": "true"}, clear=False):
            with patch.object(api_server, "create_vision_client", return_value=streaming_client):
                with patch.object(api_server, "save_output") as save_call:
                    response = client.post(
                        "/api/agent/report-check/vision/stream",
                        data={"session_id": "student_001", "stage": "report"},
                        files={"file": ("large-report.png", image_bytes, "image/png")},
                    )

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])
        self.assertIn("event: done", response.text)
        save_call.assert_called_once()

    def test_report_check_vision_stream_rejects_over_limit_quiz_context_without_model_call(self) -> None:
        api_server = load_api_server_module("api_server_report_vision_stream_quiz_context_limit")
        client = TestClient(api_server.app)

        with patch.dict(os.environ, {"ENABLE_VISION": "true"}, clear=False):
            with patch.object(api_server, "create_vision_client") as vision_client_call:
                response = client.post(
                    "/api/agent/report-check/vision/stream",
                    data={
                        "session_id": "student_001",
                        "stage": "report",
                        "quiz_context": "x" * (32 * 1024 + 1),
                    },
                    files={"file": ("report.png", b"png-bytes", "image/png")},
                )

        self.assertEqual(response.status_code, 422)
        vision_client_call.assert_not_called()

    def test_report_check_vision_stream_pdf_text_fallback_uses_text_stream(self) -> None:
        api_server = load_api_server_module("api_server_report_vision_stream_text_pdf")
        client = TestClient(api_server.app)
        streaming_client = StreamingModelClient(["### 总体评价\n", "文字报告可读"])
        pdf_text = "michelson interferometer report " * 12
        quiz_context = "【附录：课前预习习题作答与标准答案】\n第1题：学生答案 A，标准答案 B。"

        with patch.dict(os.environ, {"ENABLE_VISION": "true"}, clear=False):
            with patch.object(api_server, "create_client", return_value=streaming_client):
                with patch.object(api_server, "create_vision_client") as vision_client_call:
                    with patch.object(api_server, "save_output") as save_call:
                        response = client.post(
                            "/api/agent/report-check/vision/stream",
                            data={
                                "session_id": "student_001",
                                "stage": "report",
                                "quiz_context": quiz_context,
                            },
                            files={"file": ("report.pdf", build_text_pdf(pdf_text), "application/pdf")},
                        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])
        body = response.text
        self.assertIn('"mode": "report_check_vision"', body)
        self.assertIn('"route": "pdf_text_fallback"', body)
        self.assertIn('"document_info"', body)
        self.assertNotIn('"vision_info"', body)
        self.assertIn("event: done", body)
        self.assertIn('"answer_markdown": "### 总体评价\\n文字报告可读"', body)
        self.assertIn(quiz_context, streaming_client.user_prompt)
        vision_client_call.assert_not_called()
        save_call.assert_called_once()

    def test_report_check_vision_stream_accepts_pdf_above_non_stream_limit(self) -> None:
        api_server = load_api_server_module("api_server_report_vision_stream_large_pdf")
        client = TestClient(api_server.app)
        streaming_client = StreamingModelClient(["### 总体评价\n", "大 PDF 可读"])
        extraction = PdfExtractionResult(
            text="michelson interferometer report " * 12,
            page_count=3,
            embedded_image_count=0,
            suspected_scanned=False,
        )
        pdf_bytes = b"%PDF-1.4\n" + b"x" * (MAX_PDF_SIZE_BYTES + 1)

        with patch.dict(os.environ, {"ENABLE_VISION": "true"}, clear=False):
            with patch.object(api_server, "extract_pdf_text", return_value=extraction):
                with patch.object(api_server, "create_client", return_value=streaming_client):
                    with patch.object(api_server, "save_output") as save_call:
                        response = client.post(
                            "/api/agent/report-check/vision/stream",
                            data={"session_id": "student_001", "stage": "report"},
                            files={"file": ("large-report.pdf", pdf_bytes, "application/pdf")},
                        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])
        self.assertIn('"route": "pdf_text_fallback"', response.text)
        self.assertIn("event: done", response.text)
        save_call.assert_called_once()

    def test_report_check_vision_stream_rejects_pdf_above_stream_limit(self) -> None:
        api_server = load_api_server_module("api_server_report_vision_stream_too_large_pdf")
        client = TestClient(api_server.app)
        pdf_bytes = b"%PDF-1.4\n" + b"x" * (STREAM_PDF_LIMIT_BYTES + 1)

        with patch.dict(os.environ, {"ENABLE_VISION": "true"}, clear=False):
            with patch.object(api_server, "extract_pdf_text") as extract_call:
                response = client.post(
                    "/api/agent/report-check/vision/stream",
                    data={"session_id": "student_001", "stage": "report"},
                    files={"file": ("too-large-report.pdf", pdf_bytes, "application/pdf")},
                )

        payload = response.json()
        self.assertFalse(payload["success"])
        self.assertEqual(payload["error_code"], "VALIDATION_ERROR")
        self.assertIn("30MB", payload["message"])
        extract_call.assert_not_called()

    def test_report_check_vision_stream_embedded_pdf_uses_pdf_image_route(self) -> None:
        api_server = load_api_server_module("api_server_report_vision_stream_embedded_pdf")
        client = TestClient(api_server.app)
        streaming_client = StreamingModelClient(["### 总体评价\n", "带图报告可读"])
        pdf_text = "michelson interferometer report " * 12
        quiz_context = "【附录：课前预习习题作答与标准答案】\n第1题：学生答案 A，标准答案 B。"

        with patch.dict(os.environ, {"ENABLE_VISION": "true"}, clear=False):
            with patch.object(api_server, "create_vision_client", return_value=streaming_client):
                with patch.object(api_server, "save_output") as save_call:
                    response = client.post(
                        "/api/agent/report-check/vision/stream",
                        data={
                            "session_id": "student_001",
                            "stage": "report",
                            "quiz_context": quiz_context,
                        },
                        files={"file": ("report.pdf", build_pdf_with_images(2, pdf_text), "application/pdf")},
                    )

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])
        body = response.text
        self.assertIn('"mode": "report_check_vision"', body)
        self.assertIn('"route": "pdf_image_vision"', body)
        self.assertIn('"document_info"', body)
        self.assertIn('"vision_info"', body)
        self.assertIn('"image_source": "pdf_embedded"', body)
        self.assertIn("event: done", body)
        self.assertIn('"answer_markdown": "### 总体评价\\n带图报告可读"', body)
        prompt_text = next(
            item["text"]
            for item in streaming_client.user_prompt
            if item["type"] == "text"
        )
        self.assertIn(quiz_context, prompt_text)
        save_call.assert_called_once()

    def test_report_check_vision_stream_scanned_pdf_uses_pdf_image_route(self) -> None:
        api_server = load_api_server_module("api_server_report_vision_stream_scanned_pdf")
        client = TestClient(api_server.app)
        streaming_client = StreamingModelClient(["### 总体评价\n", "扫描报告可读"])
        extraction = PdfExtractionResult(
            text="short",
            page_count=4,
            embedded_image_count=0,
            suspected_scanned=True,
        )
        rendered_images = [VisionImage(bytes_data=b"page-1", mime="image/png")]
        quiz_context = "【附录：课前预习习题作答与标准答案】\n第1题：学生答案 A，标准答案 B。"

        with patch.dict(os.environ, {"ENABLE_VISION": "true"}, clear=False):
            with patch.object(api_server, "extract_pdf_text", return_value=extraction):
                with patch.object(api_server, "rasterize_scanned_pdf", return_value=rendered_images):
                    with patch.object(api_server, "create_vision_client", return_value=streaming_client):
                        with patch.object(api_server, "save_output") as save_call:
                            response = client.post(
                                "/api/agent/report-check/vision/stream",
                                data={
                                    "session_id": "student_001",
                                    "stage": "report",
                                    "quiz_context": quiz_context,
                                },
                                files={"file": ("scan.pdf", b"%PDF-1.4", "application/pdf")},
                            )

        self.assertEqual(response.status_code, 200)
        body = response.text
        self.assertIn('"route": "pdf_image_vision"', body)
        self.assertIn('"image_source": "pdf_rasterized"', body)
        self.assertIn('"document_info"', body)
        self.assertIn("event: done", body)
        prompt_text = next(
            item["text"]
            for item in streaming_client.user_prompt
            if item["type"] == "text"
        )
        self.assertIn(quiz_context, prompt_text)
        save_call.assert_called_once()

    def test_report_check_vision_stream_embedded_pdf_processes_twelve_images(self) -> None:
        api_server = load_api_server_module("api_server_report_vision_stream_embedded_pdf_12")
        client = TestClient(api_server.app)
        streaming_client = StreamingModelClient(["### 总体评价\n", "十二图报告可读"])
        pdf_text = "michelson interferometer report " * 12

        with patch.dict(os.environ, {"ENABLE_VISION": "true"}, clear=False):
            with patch.object(api_server, "create_vision_client", return_value=streaming_client):
                with patch.object(api_server, "save_output"):
                    response = client.post(
                        "/api/agent/report-check/vision/stream",
                        data={"session_id": "student_001", "stage": "report"},
                        files={
                            "file": (
                                "report.pdf",
                                build_pdf_with_images(STREAM_VISION_IMAGE_LIMIT, pdf_text),
                                "application/pdf",
                            )
                        },
                    )

        self.assertEqual(response.status_code, 200)
        body = response.text
        self.assertIn('"route": "pdf_image_vision"', body)
        self.assertIn('"image_source": "pdf_embedded"', body)
        self.assertIn('"image_count": 12', body)
        self.assertIn('"truncated": false', body)

    def test_report_check_vision_stream_scanned_pdf_uses_twelve_page_limit(self) -> None:
        api_server = load_api_server_module("api_server_report_vision_stream_scanned_pdf_12")
        client = TestClient(api_server.app)
        streaming_client = StreamingModelClient(["### 总体评价\n", "扫描报告可读"])
        extraction = PdfExtractionResult(
            text="short",
            page_count=13,
            embedded_image_count=0,
            suspected_scanned=True,
        )
        rendered_images = [
            VisionImage(bytes_data=f"page-{index}".encode("ascii"), mime="image/png")
            for index in range(STREAM_VISION_IMAGE_LIMIT)
        ]

        with patch.dict(os.environ, {"ENABLE_VISION": "true"}, clear=False):
            with patch.object(api_server, "extract_pdf_text", return_value=extraction):
                with patch.object(api_server, "rasterize_scanned_pdf", return_value=rendered_images) as rasterize_call:
                    with patch.object(api_server, "create_vision_client", return_value=streaming_client):
                        with patch.object(api_server, "save_output"):
                            response = client.post(
                                "/api/agent/report-check/vision/stream",
                                data={"session_id": "student_001", "stage": "report"},
                                files={"file": ("scan.pdf", b"%PDF-1.4", "application/pdf")},
                            )

        self.assertEqual(response.status_code, 200)
        body = response.text
        self.assertIn('"image_source": "pdf_rasterized"', body)
        self.assertIn('"image_count": 12', body)
        self.assertIn('"truncated": true', body)
        rasterize_call.assert_called_once_with(
            b"%PDF-1.4",
            dpi=150,
            limit=STREAM_VISION_IMAGE_LIMIT,
        )

    def test_report_check_vision_stream_disabled_returns_json_for_text_pdf(self) -> None:
        api_server = load_api_server_module("api_server_report_vision_stream_disabled")
        client = TestClient(api_server.app)

        with patch.dict(os.environ, {"ENABLE_VISION": "false"}, clear=False):
            response = client.post(
                "/api/agent/report-check/vision/stream",
                data={"session_id": "student_001", "stage": "report"},
                files={"file": ("report.pdf", build_text_pdf("michelson report " * 20), "application/pdf")},
            )

        self.assertEqual(response.status_code, 404)
        self.assertNotIn("text/event-stream", response.headers.get("content-type", ""))
        payload = response.json()
        self.assertFalse(payload["success"])
        self.assertEqual(payload["error_code"], "VISION_DISABLED")
    def test_report_check_vision_disabled_returns_404(self) -> None:
        with patch.dict(os.environ, {"ENABLE_VISION": "false"}, clear=False):
            api_server = load_api_server_module("api_server_report_vision_disabled")
            client = TestClient(api_server.app)
            response = client.post(
                "/api/agent/report-check/vision",
                data={"session_id": "student_001", "stage": "report"},
                files={"file": ("report.png", b"png-bytes", "image/png")},
            )

        payload = response.json()
        self.assertEqual(response.status_code, 404)
        self.assertFalse(payload["success"])
        self.assertEqual(payload["error_code"], "VISION_DISABLED")
        self.assertEqual(payload["route"], "vision_report")

    def test_report_check_vision_image_success_uses_path_c_vision_info(self) -> None:
        api_server = load_api_server_module("api_server_report_vision_image_success")
        client = TestClient(api_server.app)

        with patch.dict(os.environ, {"ENABLE_VISION": "true"}, clear=False):
            with patch.object(api_server, "create_vision_client", return_value=object()):
                with patch.object(api_server, "call_vision_model", return_value="### 总体评价\n图片报告可读") as call:
                    response = client.post(
                        "/api/agent/report-check/vision",
                        data={"session_id": "student_001", "stage": "report"},
                        files={"file": ("report.png", b"png-bytes", "image/png")},
                    )

        payload = response.json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["route"], "vision_report")
        self.assertEqual(
            set(payload["vision_info"]),
            {"image_count", "image_source", "image_bytes_total", "truncated"},
        )
        self.assertEqual(payload["vision_info"]["image_source"], "upload")
        self.assertFalse(payload["vision_info"]["truncated"])
        self.assertEqual(payload["vision_info"]["image_bytes_total"], len(b"png-bytes"))
        call.assert_called_once()

    def test_report_check_vision_model_error_returns_vision_api_error(self) -> None:
        api_server = load_api_server_module("api_server_report_vision_model_error")
        client = TestClient(api_server.app)

        with patch.dict(os.environ, {"ENABLE_VISION": "true"}, clear=False):
            with patch.object(api_server, "create_vision_client", return_value=object()):
                with patch.object(api_server, "call_vision_model", side_effect=RuntimeError("boom")):
                    response = client.post(
                        "/api/agent/report-check/vision",
                        data={"session_id": "student_001", "stage": "report"},
                        files={"file": ("report.png", b"png-bytes", "image/png")},
                    )

        payload = response.json()
        self.assertFalse(payload["success"])
        self.assertEqual(payload["error_code"], "VISION_API_ERROR")

    def test_report_check_vision_pdf_text_only_reuses_check_report(self) -> None:
        api_server = load_api_server_module("api_server_report_vision_text_pdf")
        client = TestClient(api_server.app)
        pdf_text = "michelson interferometer report " * 12
        fixed_payload = {
            "success": True,
            "mode": "report_check",
            "stage": "report",
            "display_type": "markdown",
            "answer_markdown": "### 总体评价\n文字报告可读",
            "saved": True,
            "warning": None,
        }

        with patch.dict(os.environ, {"ENABLE_VISION": "true"}, clear=False):
            with patch.object(api_server, "call_vision_model") as vision_call:
                with patch.object(api_server, "check_report", return_value=fixed_payload) as check_call:
                    response = client.post(
                        "/api/agent/report-check/vision",
                        data={"session_id": "student_001", "stage": "report"},
                        files={"file": ("report.pdf", build_text_pdf(pdf_text), "application/pdf")},
                    )

        payload = response.json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["route"], "vision_report")
        self.assertIn("document_info", payload)
        self.assertNotIn("vision_info", payload)
        self.assertIn("未触发视觉模型", payload["warning"])
        vision_call.assert_not_called()
        check_call.assert_called_once()
        self.assertEqual(check_call.call_args.kwargs["report_content"], pdf_text.strip())

    def test_report_check_vision_embedded_truncated_flag(self) -> None:
        api_server = load_api_server_module("api_server_report_vision_embedded_truncated")
        client = TestClient(api_server.app)
        pdf_text = "michelson interferometer report " * 12

        with patch.dict(os.environ, {"ENABLE_VISION": "true"}, clear=False):
            with patch.object(api_server, "create_vision_client", return_value=object()):
                with patch.object(api_server, "call_vision_model", return_value="### 总体评价\n带图报告") as call:
                    response = client.post(
                        "/api/agent/report-check/vision",
                        data={"session_id": "student_001", "stage": "report"},
                        files={
                            "file": (
                                "report.pdf",
                                build_pdf_with_images(10, text=pdf_text),
                                "application/pdf",
                            )
                        },
                    )

        payload = response.json()
        self.assertTrue(payload["success"])
        self.assertEqual(
            set(payload["vision_info"]),
            {"image_count", "image_source", "image_bytes_total", "truncated"},
        )
        self.assertEqual(payload["vision_info"]["image_source"], "pdf_embedded")
        self.assertEqual(payload["vision_info"]["image_count"], 8)
        self.assertTrue(payload["vision_info"]["truncated"])
        call.assert_called_once()

    def test_report_check_vision_scanned_pdf_uses_rasterized_images(self) -> None:
        api_server = load_api_server_module("api_server_report_vision_scanned_pdf")
        client = TestClient(api_server.app)
        rasterized = [
            VisionImage(bytes_data=f"page-{index}".encode("ascii"), mime="image/png")
            for index in range(8)
        ]

        with patch.dict(os.environ, {"ENABLE_VISION": "true"}, clear=False):
            with patch.object(
                api_server,
                "extract_pdf_text",
                return_value=PdfExtractionResult(
                    text="",
                    page_count=10,
                    embedded_image_count=0,
                    suspected_scanned=True,
                ),
            ):
                with patch.object(api_server, "rasterize_scanned_pdf", return_value=rasterized):
                    with patch.object(api_server, "create_vision_client", return_value=object()):
                        with patch.object(api_server, "call_vision_model", return_value="### 总体评价\n扫描报告") as call:
                            response = client.post(
                                "/api/agent/report-check/vision",
                                data={"session_id": "student_001", "stage": "report"},
                                files={"file": ("report.pdf", b"%PDF-1.4", "application/pdf")},
                            )

        payload = response.json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["document_info"]["page_count"], 10)
        self.assertEqual(payload["vision_info"]["image_source"], "pdf_rasterized")
        self.assertEqual(payload["vision_info"]["image_count"], 8)
        self.assertTrue(payload["vision_info"]["truncated"])
        call.assert_called_once()

    def test_report_check_vision_corrupt_pdf_returns_parse_error(self) -> None:
        api_server = load_api_server_module("api_server_report_vision_corrupt_pdf")
        client = TestClient(api_server.app)

        with patch.dict(os.environ, {"ENABLE_VISION": "true"}, clear=False):
            with patch.object(api_server, "call_vision_model") as call:
                with patch("pypdf._reader.logger_warning"):
                    response = client.post(
                        "/api/agent/report-check/vision",
                        data={"session_id": "student_001", "stage": "report"},
                        files={"file": ("report.pdf", b"not a pdf", "application/pdf")},
                    )

        payload = response.json()
        self.assertFalse(payload["success"])
        self.assertEqual(payload["error_code"], "DOCUMENT_PARSE_ERROR")
        call.assert_not_called()

    def test_report_check_vision_too_many_pages_returns_validation_error(self) -> None:
        api_server = load_api_server_module("api_server_report_vision_too_many_pages")
        client = TestClient(api_server.app)

        with patch.dict(os.environ, {"ENABLE_VISION": "true"}, clear=False):
            with patch.object(api_server, "extract_pdf_text", side_effect=api_server.PdfTooManyPagesError):
                response = client.post(
                    "/api/agent/report-check/vision",
                    data={"session_id": "student_001", "stage": "report"},
                    files={"file": ("report.pdf", b"%PDF-1.4", "application/pdf")},
                )

        payload = response.json()
        self.assertFalse(payload["success"])
        self.assertEqual(payload["error_code"], "VALIDATION_ERROR")

    def test_report_check_vision_logs_metadata_without_image_format_or_leaks(self) -> None:
        api_server = load_api_server_module("api_server_report_vision_logging")
        client = TestClient(api_server.app)

        with patch.dict(os.environ, {"ENABLE_VISION": "true"}, clear=False):
            with patch.object(api_server, "create_vision_client", return_value=object()):
                with patch.object(api_server, "call_vision_model", return_value="### 总体评价\n图片报告可读"):
                    with patch.object(api_server.logger, "info") as log_info:
                        response = client.post(
                            "/api/agent/report-check/vision",
                            data={"session_id": "student_secret", "stage": "report"},
                            files={"file": ("secret-name.png", b"png-bytes", "image/png")},
                        )

        self.assertEqual(response.status_code, 200)
        extra = log_info.call_args.kwargs["extra"]
        self.assertEqual(extra["mode"], "report_check")
        self.assertEqual(extra["route"], "vision_report")
        self.assertEqual(extra["image_count"], 1)
        self.assertEqual(extra["image_bytes_total"], len(b"png-bytes"))
        self.assertNotIn("image_format", extra)
        self.assertNotIn("student_input", extra)
        self.assertNotIn("session_id", extra)
        self.assertNotIn("file", extra)
        self.assertNotIn("pdf_bytes", extra)
        self.assertNotIn("report_content", extra)


def build_test_resources() -> dict[str, Any]:
    return {
        "knowledge_base": "实验知识库",
        "tutor_prompt": (
            "TUTOR_PROMPT_MARKER\n"
            "### 问题判断\n"
            "### 原理提示\n"
            "### 引导思考\n"
            "### 操作或学习建议\n"
            "### 注意事项\n"
            "{knowledge_base}\n"
            "{stage_instruction}\n"
            "{teaching_instruction}\n"
            "{conversation_state}\n"
            "{conversation_history}\n"
            "{student_question}"
        ),
        "quiz_prompt": (
            "QUIZ_PROMPT_MARKER\n"
            "{knowledge_base}\n"
            "{focus_instruction}\n"
            "{student_request}\n"
            "{validation_feedback}"
        ),
        "workflow_prompt": (
            "WORKFLOW_PROMPT_MARKER\n"
            "不得编造实验数据\n"
            "{stage_instruction}\n"
            "{conversation_state}\n"
            "{conversation_history}\n"
            "{student_question}"
        ),
        "report_prompt": "REPORT_PROMPT_MARKER\n{knowledge_base}\n{report_content}",
        "anti_cheating_prompt": "anti",
        "stage_classifier_prompt": "CLASSIFIER_MARKER\n{recent_context}\n{student_question}",
        "teaching_policy_classifier_prompt": (
            "TEACHING_CLASSIFIER_MARKER\n{stage}\n{recent_context}\n{student_question}"
        ),
        "stage_prompts": {
            "pre_lab": "PRE_LAB_STAGE_MARKER",
            "during_experiment": "DURING_STAGE_MARKER",
            "result_evaluation": "RESULT_STAGE_MARKER",
            "unknown": "UNKNOWN_STAGE_MARKER",
        },
        "teaching_prompts": {
            "direct": "DIRECT_TEACHING_MARKER status={student_status}",
            "guided": (
                "GUIDED_TEACHING_MARKER level={guidance_level} status={student_status}"
            ),
            "hybrid": (
                "HYBRID_TEACHING_MARKER level={guidance_level} status={student_status}"
            ),
        },
    }


def ensure_test_asset_image(filename: str) -> tuple[Path, bool]:
    image_path = PROJECT_DIR / "assets" / "images" / filename
    if image_path.exists():
        return image_path, False
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (1, 1), color="white").save(image_path, format="JPEG")
    return image_path, True


class RecordingModelClient:
    def __init__(self, content: str) -> None:
        self.calls = 0
        self.user_prompt = ""
        self.system_prompt = ""
        self.chat = RecordingChat(self, content)


class RecordingChat:
    def __init__(self, owner: RecordingModelClient, content: str) -> None:
        self.completions = RecordingCompletions(owner, content)


class RecordingCompletions:
    def __init__(self, owner: RecordingModelClient, content: str) -> None:
        self.owner = owner
        self.content = content

    def create(self, **kwargs: Any) -> "RecordingResponse":
        self.owner.calls += 1
        messages = kwargs["messages"]
        self.owner.system_prompt = messages[0]["content"]
        self.owner.user_prompt = messages[1]["content"]
        return RecordingResponse(self.content)


class RecordingResponse:
    def __init__(self, content: str) -> None:
        self.choices = [RecordingChoice(content)]


class RecordingChoice:
    def __init__(self, content: str) -> None:
        self.message = RecordingMessage(content)


class RecordingMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class StreamingModelClient:
    def __init__(self, chunks: list[str], error: Exception | None = None) -> None:
        self.calls = 0
        self.user_prompt = ""
        self.system_prompt = ""
        self.last_response: StreamingResponse | None = None
        self.chat = StreamingChat(self, chunks, error)


class StreamingChat:
    def __init__(
        self,
        owner: StreamingModelClient,
        chunks: list[str],
        error: Exception | None,
    ) -> None:
        self.completions = StreamingCompletions(owner, chunks, error)


class StreamingCompletions:
    def __init__(
        self,
        owner: StreamingModelClient,
        chunks: list[str],
        error: Exception | None,
    ) -> None:
        self.owner = owner
        self.chunks = chunks
        self.error = error

    def create(self, **kwargs: Any) -> "StreamingResponse":
        self.owner.calls += 1
        messages = kwargs["messages"]
        self.owner.system_prompt = messages[0]["content"]
        self.owner.user_prompt = messages[1]["content"]
        if self.error is not None:
            raise self.error
        response = StreamingResponse(self.chunks)
        self.owner.last_response = response
        return response


class StreamingResponse:
    def __init__(self, chunks: list[str]) -> None:
        self.chunks = chunks
        self.closed = False

    def __iter__(self):
        for chunk in self.chunks:
            yield StreamingChunk(chunk)

    def close(self) -> None:
        self.closed = True


class StreamingChunk:
    def __init__(self, content: str) -> None:
        self.choices = [StreamingChunkChoice(content)]


class StreamingChunkChoice:
    def __init__(self, content: str) -> None:
        self.delta = StreamingDelta(content)


class StreamingDelta:
    def __init__(self, content: str) -> None:
        self.content = content
