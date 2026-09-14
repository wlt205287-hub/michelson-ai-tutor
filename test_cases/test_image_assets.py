from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n")


def write_manifest(path: Path, items: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"images": items}, ensure_ascii=False), encoding="utf-8")


class ImageAssetsTests(TestCase):
    def setUp(self) -> None:
        self._env_patch = patch.dict(os.environ, {}, clear=False)
        self._env_patch.start()
        os.environ.pop("AI_AGENT_IMAGES_ENABLED", None)
        os.environ.pop("AI_AGENT_PUBLIC_BASE_URL", None)

    def tearDown(self) -> None:
        self._env_patch.stop()

    def test_disabled_images_return_empty_prompt_block(self) -> None:
        import image_assets

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images_dir = root / "images"
            manifest = root / "image_manifest.json"
            write_image(images_dir / "michelson_optical_path.png")
            write_manifest(
                manifest,
                [
                    {
                        "id": "optical_path",
                        "filename": "michelson_optical_path.png",
                        "title": "迈克耳孙干涉仪光路原理图",
                        "alt": "迈克耳孙干涉仪光路原理图",
                        "concepts": ["光路"],
                        "description": "用于解释光路。",
                        "task_types": ["tutor"],
                        "keywords": ["光路"],
                    }
                ],
            )

            with patch.dict(os.environ, {"AI_AGENT_IMAGES_ENABLED": "false"}, clear=False):
                candidates = image_assets.select_teaching_images(
                    "解释一下光路",
                    task_type="tutor",
                    manifest_path=manifest,
                    images_dir=images_dir,
                )

        self.assertEqual(candidates, [])
        self.assertEqual(image_assets.build_teaching_images_prompt(candidates), "")

    def test_public_base_url_empty_and_non_empty(self) -> None:
        import image_assets

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images_dir = root / "images"
            manifest = root / "image_manifest.json"
            write_image(images_dir / "micrometer_reading.png")
            write_manifest(
                manifest,
                [
                    {
                        "id": "reading",
                        "filename": "micrometer_reading.png",
                        "title": "粗调 / 微调读数示意图",
                        "alt": "粗调 / 微调读数示意图",
                        "concepts": ["读数"],
                        "description": "用于解释读数。",
                        "task_types": ["tutor"],
                        "keywords": ["读数"],
                    }
                ],
            )

            relative = image_assets.select_teaching_images(
                "微调鼓轮怎么读数？",
                task_type="tutor",
                manifest_path=manifest,
                images_dir=images_dir,
            )
            with patch.dict(os.environ, {"AI_AGENT_PUBLIC_BASE_URL": "http://127.0.0.1:8000/"}, clear=False):
                absolute = image_assets.select_teaching_images(
                    "微调鼓轮怎么读数？",
                    task_type="tutor",
                    manifest_path=manifest,
                    images_dir=images_dir,
                )

        self.assertEqual(
            relative[0]["markdown"],
            "![粗调 / 微调读数示意图](/static/ai_agent/images/micrometer_reading.png)",
        )
        self.assertEqual(
            absolute[0]["markdown"],
            "![粗调 / 微调读数示意图](http://127.0.0.1:8000/static/ai_agent/images/micrometer_reading.png)",
        )

    def test_manifest_security_skips_invalid_paths_and_external_urls(self) -> None:
        import image_assets

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images_dir = root / "images"
            manifest = root / "image_manifest.json"
            write_image(images_dir / "valid.png")
            write_image(images_dir / "notes.txt")
            write_manifest(
                manifest,
                [
                    {
                        "id": "valid",
                        "filename": "valid.png",
                        "title": "有效图片",
                        "alt": "有效图片",
                        "concepts": ["光路"],
                        "description": "短说明",
                        "task_types": ["tutor"],
                        "keywords": ["光路"],
                    },
                    {"id": "parent", "filename": "../secret.png", "keywords": ["光路"]},
                    {"id": "external", "filename": "https://example.com/x.png", "keywords": ["光路"]},
                    {"id": "absolute", "filename": str(images_dir / "valid.png"), "keywords": ["光路"]},
                    {"id": "bad_ext", "filename": "notes.txt", "keywords": ["光路"]},
                    {"id": "missing", "filename": "missing.png", "keywords": ["光路"]},
                ],
            )

            candidates = image_assets.select_teaching_images(
                "讲解光路",
                task_type="tutor",
                manifest_path=manifest,
                images_dir=images_dir,
            )

        self.assertEqual([item["id"] for item in candidates], ["valid"])

    def test_manifest_missing_or_invalid_falls_back_to_empty(self) -> None:
        import image_assets

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images_dir = root / "images"
            missing = root / "missing.json"
            broken = root / "broken.json"
            broken.write_text("{bad json", encoding="utf-8")

            self.assertEqual(
                image_assets.select_teaching_images(
                    "讲解光路",
                    task_type="tutor",
                    manifest_path=missing,
                    images_dir=images_dir,
                ),
                [],
            )
            self.assertEqual(
                image_assets.select_teaching_images(
                    "讲解光路",
                    task_type="tutor",
                    manifest_path=broken,
                    images_dir=images_dir,
                ),
                [],
            )

    def test_keyword_matching_and_default_limits(self) -> None:
        import image_assets

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images_dir = root / "images"
            manifest = root / "image_manifest.json"
            for filename in [
                "michelson_optical_path.png",
                "instrument_structure_labeled.png",
                "compensation_plate.png",
                "fringe_in_out.png",
            ]:
                write_image(images_dir / filename)
            write_manifest(
                manifest,
                [
                    {
                        "id": "optical_path",
                        "filename": "michelson_optical_path.png",
                        "title": "光路原理图",
                        "alt": "光路原理图",
                        "concepts": ["光路"],
                        "description": "解释光路。",
                        "task_types": ["tutor"],
                        "keywords": ["光路", "分光"],
                    },
                    {
                        "id": "structure",
                        "filename": "instrument_structure_labeled.png",
                        "title": "仪器结构标注图",
                        "alt": "仪器结构标注图",
                        "concepts": ["结构"],
                        "description": "解释部件。",
                        "task_types": ["tutor"],
                        "keywords": ["结构", "部件"],
                    },
                    {
                        "id": "compensation",
                        "filename": "compensation_plate.png",
                        "title": "补偿板作用示意图",
                        "alt": "补偿板作用示意图",
                        "concepts": ["补偿板"],
                        "description": "解释补偿板。",
                        "task_types": ["tutor"],
                        "keywords": ["补偿板"],
                    },
                    {
                        "id": "fringe",
                        "filename": "fringe_in_out.png",
                        "title": "条纹吞吐示意图",
                        "alt": "条纹吞吐示意图",
                        "concepts": ["条纹吞吐"],
                        "description": "解释波长测量。",
                        "task_types": ["tutor", "report_check"],
                        "keywords": ["条纹", "波长"],
                    },
                ],
            )

            tutor = image_assets.select_teaching_images(
                "综合讲解光路、仪器结构、补偿板和条纹波长测量",
                task_type="tutor",
                manifest_path=manifest,
                images_dir=images_dir,
            )
            report = image_assets.select_teaching_images(
                "报告里条纹吞吐和波长公式写错了",
                task_type="report_check",
                manifest_path=manifest,
                images_dir=images_dir,
            )
            unrelated = image_assets.select_teaching_images(
                "今天吃什么？",
                task_type="tutor",
                manifest_path=manifest,
                images_dir=images_dir,
            )

        self.assertEqual(len(tutor), 3)
        self.assertEqual(len(report), 1)
        self.assertEqual(unrelated, [])

    def test_default_images_are_mode_agnostic_and_semantically_selected(self) -> None:
        import image_assets

        cases = {
            "michelson_optical_path": "请解释 M1 固定镜、M2 动镜的分光、反射和合束光路。",
            "compensation_plate": "补偿板如何补偿两束光经过玻璃板次数不同造成的光程差？",
            "equal_inclination_rings": "M1 虚像和 M2 构成等效空气膜时为什么出现等倾圆环？",
            "fringe_in_out": "M2 动镜远离或靠近时，条纹为什么会冒出或缩进？",
        }

        for task_type in ["tutor", "report_check", "experiment_workflow"]:
            for expected_id, query in cases.items():
                with self.subTest(task_type=task_type, image=expected_id):
                    candidates = image_assets.select_teaching_images(
                        query,
                        task_type=task_type,
                    )
                    self.assertTrue(candidates)
                    self.assertEqual(candidates[0]["id"], expected_id)
                    self.assertTrue(candidates[0]["filename"].endswith((".jpg", ".gif")))

    def test_default_manifest_registers_exactly_four_replacement_images(self) -> None:
        manifest = json.loads(
            (PROJECT_DIR / "assets" / "image_manifest.json").read_text(encoding="utf-8")
        )
        images_by_id = {item["id"]: item for item in manifest["images"]}

        self.assertEqual(
            [(item["id"], item["filename"]) for item in manifest["images"]],
            [
                ("michelson_optical_path", "michelson_optical_path.jpg"),
                ("compensation_plate", "compensation_plate.jpg"),
                ("equal_inclination_rings", "equal_inclination_rings.jpg"),
                ("fringe_in_out", "fringe_in_out.gif"),
            ],
        )
        self.assertIn(
            "不用于判断固定镜或动镜",
            images_by_id["equal_inclination_rings"]["description"],
        )
        self.assertIn(
            "底部 Nλ/2 是反射式高度差换算",
            images_by_id["fringe_in_out"]["description"],
        )
        self.assertIn(
            "求波长时用 λ=2Δd/N",
            images_by_id["fringe_in_out"]["description"],
        )

    def test_fringe_image_is_selected_but_legacy_micrometer_image_is_not(self) -> None:
        import image_assets

        wavelength_measurement = image_assets.select_teaching_images(
            "如何根据条纹吞吐和条纹数计算激光波长？",
            task_type="tutor",
        )
        micrometer = image_assets.select_teaching_images(
            "微调鼓轮如何读数，为什么有空程误差？",
            task_type="tutor",
        )
        unrelated_history_cannot_override = image_assets.select_teaching_images(
            "电驱连续扫描为什么要保持单方向？",
            task_type="tutor",
            context_text="前面正在根据条纹数 N 和波长公式计算结果。",
        )

        self.assertTrue(wavelength_measurement)
        self.assertEqual(wavelength_measurement[0]["id"], "fringe_in_out")
        self.assertEqual(micrometer, [])
        self.assertEqual(unrelated_history_cannot_override, [])

    def test_build_prompt_block_is_short_and_contains_rules(self) -> None:
        import image_assets

        candidates = [
            {
                "id": "optical_path",
                "title": "光路原理图",
                "concepts": ["光路", "分光"],
                "description": "解释光路。",
                "markdown": "![光路原理图](/static/ai_agent/images/michelson_optical_path.png)",
            }
        ]

        block = image_assets.build_teaching_images_prompt(candidates)

        self.assertIn("【可用教学图片】", block)
        self.assertIn("光路原理图", block)
        self.assertIn("![光路原理图](/static/ai_agent/images/michelson_optical_path.png)", block)
        self.assertIn("后端提供的通用教学示意", block)
        self.assertIn("必须将每条给出的 Markdown 恰好插入一次", block)
        self.assertLess(len(block), 500)
