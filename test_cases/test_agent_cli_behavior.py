from __future__ import annotations

import importlib.util
from io import StringIO
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch


def load_agent_module():
    agent_path = Path(__file__).resolve().parents[1] / "agent_demo" / "agent.py"
    spec = importlib.util.spec_from_file_location("agent", agent_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载 agent.py")
    agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent)
    return agent


class AgentCliBehaviorTests(TestCase):
    def test_choose_mode_rejects_removed_mode_3(self) -> None:
        agent = load_agent_module()

        with patch("builtins.input", return_value="3"):
            with self.assertRaises(ValueError):
                agent.choose_mode()

    def test_agent_uses_single_deepseek_client_entrypoint(self) -> None:
        agent = load_agent_module()

        self.assertTrue(hasattr(agent, "create_client"))
        self.assertFalse(hasattr(agent, "create_multiturn_client"))

    def test_mode_1_dialogue_function_is_named_for_tutor_module(self) -> None:
        agent = load_agent_module()

        self.assertTrue(hasattr(agent, "run_tutor_dialogue"))
        self.assertFalse(hasattr(agent, "run_multiturn_dialogue"))

    def test_independent_multiturn_prompt_is_not_used(self) -> None:
        agent = load_agent_module()

        self.assertFalse(hasattr(agent, "MULTITURN_TUTOR_PROMPT_PATH"))
        self.assertFalse(hasattr(agent, "build_multiturn_prompt"))

    def test_tutor_prompt_supports_contextual_guidance(self) -> None:
        prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "tutor_prompt.txt"
        prompt = prompt_path.read_text(encoding="utf-8")

        self.assertIn("{conversation_history}", prompt)
        self.assertIn("{stage_instruction}", prompt)
        self.assertIn("{teaching_instruction}", prompt)
        self.assertIn("不强制固定标题", prompt)
        self.assertIn("不向学生展示“问题判断”", prompt)
        self.assertIn("不直接代写完整实验报告", prompt)
        self.assertIn("公式使用 Markdown 兼容的 LaTeX", prompt)
        self.assertNotIn("`$d$`", prompt)
        self.assertNotIn("`$\\Delta L$`", prompt)
        for heading in [
            "### 问题判断",
            "### 原理提示",
            "### 引导思考",
            "### 操作或学习建议",
            "### 注意事项",
        ]:
            self.assertNotIn(heading, prompt)

    def test_tutor_prompt_separates_stage_content_from_guided_strategy(self) -> None:
        prompts_dir = Path(__file__).resolve().parents[1] / "prompts"
        stage_prompt = (prompts_dir / "stage_pre_lab_prompt.txt").read_text(encoding="utf-8")
        guided_prompt = (prompts_dir / "teaching_guided_prompt.txt").read_text(encoding="utf-8")

        self.assertIn("公式推导或原理理解", stage_prompt)
        self.assertIn("由本轮教学策略决定", stage_prompt)
        self.assertNotIn("第一轮先给关键物理方向", stage_prompt)
        self.assertIn("$\\omega_0=4\\pi v/\\lambda$", stage_prompt)
        self.assertIn("$f_0=2v/\\lambda$", stage_prompt)
        self.assertIn("$h=\\lambda\\Phi_{rel}/(4\\pi)$", stage_prompt)
        self.assertIn("四步相移", stage_prompt)

        self.assertIn("级别 1", guided_prompt)
        self.assertIn("级别 2", guided_prompt)
        self.assertIn("级别 3", guided_prompt)
        self.assertIn("必须给出完整解释", guided_prompt)
        self.assertIn("每轮最多提出一个核心问题", guided_prompt)

    def test_report_prompt_uses_markdown_sections(self) -> None:
        prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "report_checker_prompt.txt"
        prompt = prompt_path.read_text(encoding="utf-8")

        self.assertIn("{knowledge_base}", prompt)
        self.assertIn("{report_content}", prompt)
        self.assertIn("公式使用 `$...$` 或独立的 `$$...$$`", prompt)
        self.assertIn("连续扫描时间载波", prompt)
        self.assertIn("原理与方法：20 分", prompt)
        self.assertIn("最终成绩由教师确认", prompt)
        for heading in [
            "### 总体评价",
            "### 发现的问题",
            "### 问题原因",
            "### 修改建议",
            "### 初步评分",
            "### 是否需要教师复查",
        ]:
            self.assertIn(heading, prompt)

    def test_tutor_prompt_locks_surface_topography_method_and_scope(self) -> None:
        prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "tutor_prompt.txt"
        prompt = prompt_path.read_text(encoding="utf-8")

        self.assertIn("电驱参考镜连续单向扫描", prompt)
        self.assertIn("多帧时间序列 FFT", prompt)
        self.assertIn("不得把四步相移", prompt)
        self.assertIn("相对高度或相对表面形貌", prompt)
        self.assertIn("不提供 `process_phase.py`", prompt)

    def test_streaming_model_call_prints_chunks_and_returns_full_text(self) -> None:
        agent = load_agent_module()
        client = FakeStreamingClient(["第一段", "第二段"])

        with patch("sys.stdout", new_callable=StringIO) as stdout:
            result = agent.call_model_stream(
                client,
                "system",
                "user",
                temperature=0.3,
                model="deepseek-v4-pro",
            )

        self.assertEqual(result, "第一段第二段")
        self.assertEqual(stdout.getvalue(), "第一段第二段")
        self.assertTrue(client.stream_enabled)

    def test_normalize_model_output_preserves_latex_markdown(self) -> None:
        agent = load_agent_module()
        markdown = "### 原理提示\n\\[\n\\Delta L = 2d\n\\]"

        self.assertEqual(agent.normalize_model_output(markdown), markdown)

    def test_frontend_payload_shape_for_tutor_markdown(self) -> None:
        agent = load_agent_module()

        payload = agent.build_frontend_payload(
            "tutor",
            "### 问题判断\n原理理解",
            {"saved": True, "warning": None},
        )

        self.assertEqual(payload["mode"], "tutor")
        self.assertEqual(payload["answer_markdown"], "### 问题判断\n原理理解")
        self.assertEqual(payload["display_type"], "markdown")
        self.assertTrue(payload["need_student_reply"])
        self.assertTrue(payload["saved"])
        self.assertIsNone(payload["warning"])

    def test_default_model_name_is_supported_deepseek_v4_pro(self) -> None:
        agent = load_agent_module()
        client = FakeStreamingClient(["默认模型"])

        with patch.dict("os.environ", {}, clear=True):
            with patch("sys.stdout", new_callable=StringIO):
                agent.call_model_stream(
                    client,
                    "system",
                    "user",
                    temperature=0.3,
                )

        self.assertEqual(client.model_name, "deepseek-v4-pro")

    def test_report_check_uses_agent_core_check_report(self) -> None:
        agent = load_agent_module()
        client = object()
        resources = {
            "knowledge_base": "知识库",
            "tutor_prompt": "tutor",
            "report_prompt": "report",
            "anti_cheating_prompt": "anti",
        }
        payload = {
            "success": True,
            "mode": "report_check",
            "stage": "report",
            "display_type": "markdown",
            "answer_markdown": "### 总体评价\n内容",
            "saved": True,
            "warning": None,
        }

        with patch.object(agent, "load_resources", return_value=resources):
            with patch.object(agent, "choose_mode", return_value="2"):
                with patch.object(agent, "create_client", return_value=client):
                    with patch.object(agent, "read_multiline_input", return_value="报告片段"):
                        with patch.object(agent, "check_report", return_value=payload) as report_call:
                            with patch("sys.stdout", new_callable=StringIO):
                                agent.main()

        report_call.assert_called_once()
        self.assertEqual(report_call.call_args.kwargs["report_content"], "报告片段")
        self.assertEqual(report_call.call_args.kwargs["stage"], "report")
        self.assertTrue(report_call.call_args.kwargs["save"])
        self.assertTrue(report_call.call_args.kwargs["stream"])
        self.assertEqual(report_call.call_args.kwargs["client"], client)
        self.assertEqual(report_call.call_args.kwargs["resources"], resources)

    def test_report_check_saved_message_hides_full_path(self) -> None:
        agent = load_agent_module()
        client = object()
        resources = {
            "knowledge_base": "知识库",
            "tutor_prompt": "tutor",
            "report_prompt": "report",
            "anti_cheating_prompt": "anti",
        }

        with patch.object(agent, "load_resources", return_value=resources):
            with patch.object(agent, "choose_mode", return_value="2"):
                with patch.object(agent, "create_client", return_value=client):
                    with patch.object(agent, "read_multiline_input", return_value="报告片段"):
                        with patch.object(
                            agent,
                            "check_report",
                            return_value={
                                "success": True,
                                "mode": "report_check",
                                "stage": "report",
                                "display_type": "markdown",
                                "answer_markdown": "### 总体评价\n内容",
                                "saved": True,
                                "warning": None,
                            },
                        ):
                            with patch("sys.stdout", new_callable=StringIO) as stdout:
                                agent.main()

        output = stdout.getvalue()
        self.assertIn("本轮输出已保存到 outputs 目录", output)
        self.assertNotIn("D:/secret/output.md", output)


class FakeStreamingClient:
    def __init__(self, chunks: list[str]) -> None:
        self.stream_enabled = False
        self.model_name = None
        self.chat = FakeChat(self, chunks)


class FakeChat:
    def __init__(self, owner: FakeStreamingClient, chunks: list[str]) -> None:
        self.completions = FakeCompletions(owner, chunks)


class FakeCompletions:
    def __init__(self, owner: FakeStreamingClient, chunks: list[str]) -> None:
        self.owner = owner
        self.chunks = chunks

    def create(self, **kwargs):
        self.owner.stream_enabled = kwargs.get("stream") is True
        self.owner.model_name = kwargs.get("model")
        return [FakeChunk(content) for content in self.chunks]


class FakeChunk:
    def __init__(self, content: str) -> None:
        self.choices = [FakeChoice(content)]


class FakeChoice:
    def __init__(self, content: str) -> None:
        self.delta = FakeDelta(content)


class FakeDelta:
    def __init__(self, content: str) -> None:
        self.content = content
