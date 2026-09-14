from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path
from unittest import TestCase


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

STANDARD_QA_PATH = PROJECT_DIR / "assets" / "standard_qa.json"
APPROVED_STANDARD_QA_SHA256 = (
    "0050a4cad07c27e931e77bcb1e7400925dff6c9907025da06fcd563366460e29"
)


def write_standard_qa(path: Path, items: list[dict[str, object]]) -> None:
    path.write_text(
        json.dumps({"items": items}, ensure_ascii=False),
        encoding="utf-8",
    )


class StandardQATests(TestCase):
    def test_repository_json_is_the_approved_unmodified_file(self) -> None:
        self.assertTrue(STANDARD_QA_PATH.is_file())
        self.assertEqual(
            hashlib.sha256(STANDARD_QA_PATH.read_bytes()).hexdigest(),
            APPROVED_STANDARD_QA_SHA256,
        )

        parsed = json.loads(STANDARD_QA_PATH.read_text(encoding="utf-8"))
        self.assertEqual(len(parsed["items"]), 5)
        self.assertEqual(
            sum(len(item["aliases"]) for item in parsed["items"]),
            101,
        )

    def test_every_repository_alias_returns_its_exact_answer(self) -> None:
        import standard_qa

        parsed = json.loads(STANDARD_QA_PATH.read_text(encoding="utf-8"))
        for item in parsed["items"]:
            for alias in item["aliases"]:
                with self.subTest(item_id=item["id"], alias=alias):
                    match = standard_qa.find_standard_answer(alias)
                    self.assertIsNotNone(match)
                    self.assertEqual(match.item_id, item["id"])
                    self.assertEqual(match.answer_markdown, item["answer_markdown"])

    def test_matching_normalizes_width_case_whitespace_and_terminal_punctuation(self) -> None:
        import standard_qa

        with tempfile.TemporaryDirectory() as tmp:
            qa_path = Path(tmp) / "standard_qa.json"
            write_standard_qa(
                qa_path,
                [
                    {
                        "id": "beam_splitter_role",
                        "aliases": ["ABC 分束镜的作用是什么"],
                        "answer_markdown": "固定答案",
                    }
                ],
            )

            match = standard_qa.find_standard_answer(
                "ａｂｃ　分束镜 的作用是什么？！",
                qa_path=qa_path,
            )

        self.assertIsNotNone(match)
        self.assertEqual(match.item_id, "beam_splitter_role")

    def test_unregistered_near_question_does_not_match(self) -> None:
        import standard_qa

        self.assertIsNone(
            standard_qa.find_standard_answer("为什么要始终沿同一方向移动动镜")
        )

    def test_answer_markdown_is_returned_without_stripping_or_rewriting(self) -> None:
        import standard_qa

        expected_answer = " \n# 固定回答\n\n$\\Delta d$\n "
        with tempfile.TemporaryDirectory() as tmp:
            qa_path = Path(tmp) / "standard_qa.json"
            write_standard_qa(
                qa_path,
                [
                    {
                        "id": "exact_markdown",
                        "aliases": ["固定回答测试"],
                        "answer_markdown": expected_answer,
                    }
                ],
            )

            match = standard_qa.find_standard_answer(
                "固定回答测试",
                qa_path=qa_path,
            )

        self.assertIsNotNone(match)
        self.assertEqual(match.answer_markdown, expected_answer)

    def test_missing_or_invalid_file_falls_back_to_no_match(self) -> None:
        import standard_qa

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = root / "missing.json"
            broken = root / "broken.json"
            broken.write_text("{bad json", encoding="utf-8")

            self.assertIsNone(
                standard_qa.find_standard_answer("任意问题", qa_path=missing)
            )
            self.assertIsNone(
                standard_qa.find_standard_answer("任意问题", qa_path=broken)
            )

    def test_invalid_entries_are_skipped_without_disabling_valid_entries(self) -> None:
        import standard_qa

        with tempfile.TemporaryDirectory() as tmp:
            qa_path = Path(tmp) / "standard_qa.json"
            write_standard_qa(
                qa_path,
                [
                    {
                        "id": "Invalid-ID",
                        "aliases": ["非法条目"],
                        "answer_markdown": "不应命中",
                    },
                    {
                        "id": "valid_item",
                        "aliases": ["有效问题"],
                        "answer_markdown": "有效答案",
                    },
                ],
            )

            self.assertIsNone(
                standard_qa.find_standard_answer("非法条目", qa_path=qa_path)
            )
            valid_match = standard_qa.find_standard_answer(
                "有效问题",
                qa_path=qa_path,
            )

        self.assertIsNotNone(valid_match)
        self.assertEqual(valid_match.answer_markdown, "有效答案")

    def test_cross_answer_alias_collision_fails_closed(self) -> None:
        import standard_qa

        with tempfile.TemporaryDirectory() as tmp:
            qa_path = Path(tmp) / "standard_qa.json"
            write_standard_qa(
                qa_path,
                [
                    {
                        "id": "first_answer",
                        "aliases": ["重复问题"],
                        "answer_markdown": "答案一",
                    },
                    {
                        "id": "second_answer",
                        "aliases": ["重 复 问 题？"],
                        "answer_markdown": "答案二",
                    },
                ],
            )

            match = standard_qa.find_standard_answer(
                "重复问题",
                qa_path=qa_path,
            )

        self.assertIsNone(match)
