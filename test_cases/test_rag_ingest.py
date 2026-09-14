from __future__ import annotations

import json
import sys
import tempfile
from io import BytesIO
from pathlib import Path
from unittest import TestCase

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def build_text_pdf(text: str) -> bytes:
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        },
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})},
    )
    stream = DecodedStreamObject()
    escaped_text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream.set_data(f"BT /F1 12 Tf 72 720 Td ({escaped_text}) Tj ET".encode("latin-1"))
    page[NameObject("/Contents")] = writer._add_object(stream)
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


class FakeCollection:
    def __init__(self) -> None:
        self.add_calls: list[dict[str, object]] = []

    def add(self, **kwargs) -> None:
        self.add_calls.append(kwargs)


class RagIngestTests(TestCase):
    def make_config(self, docs_dir: Path, index_dir: Path):
        from rag_config import RagConfig

        return RagConfig(
            enabled=True,
            docs_dir=docs_dir,
            index_dir=index_dir,
            embedding_model="fake-embedding",
            chunk_size=80,
            chunk_overlap=10,
            top_k=3,
            min_score=0.4,
        )

    def test_markdown_parsing_keeps_frontmatter_and_heading_metadata(self) -> None:
        from rag_ingest import build_chunks_for_file

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs_dir = root / "rag_docs"
            index_dir = root / "rag_index"
            source = docs_dir / "faq" / "条纹问题FAQ.md"
            source.parent.mkdir(parents=True)
            source.write_text(
                "---\n"
                "task_type: operation\n"
                "authority_level: high\n"
                "source_type: internal_experiment_plan\n"
                "source_file: 实验方案.md\n"
                "measurement_method: continuous_scan_temporal_fft\n"
                "measurement_target: relative_surface_topography\n"
                "---\n"
                "# 常见问题\n\n"
                "## 空程误差\n\n"
                "空程误差会影响读数，需要从同一方向逼近测量位置。\n",
                encoding="utf-8",
            )
            config = self.make_config(docs_dir, index_dir)

            chunks = build_chunks_for_file(source, config)

        self.assertEqual(len(chunks), 1)
        self.assertIn("空程误差", chunks[0].text)
        self.assertEqual(chunks[0].metadata["source_file"], "条纹问题FAQ.md")
        self.assertEqual(chunks[0].metadata["source_path"], "rag_docs/faq/条纹问题FAQ.md")
        self.assertEqual(chunks[0].metadata["section"], "常见问题 > 空程误差")
        self.assertEqual(chunks[0].metadata["task_type"], "operation")
        self.assertEqual(chunks[0].metadata["authority_level"], "high")
        self.assertEqual(chunks[0].metadata["source_type"], "internal_experiment_plan")
        self.assertEqual(chunks[0].metadata["original_source_file"], "实验方案.md")
        self.assertEqual(chunks[0].metadata["measurement_method"], "continuous_scan_temporal_fft")
        self.assertEqual(chunks[0].metadata["measurement_target"], "relative_surface_topography")

    def test_repository_rag_docs_only_expose_surface_topography_method(self) -> None:
        from rag_ingest import iter_source_files

        docs_dir = PROJECT_DIR / "rag_docs"
        source_names = {path.name for path in iter_source_files(docs_dir)}
        combined = "\n".join(
            path.read_text(encoding="utf-8")
            for path in iter_source_files(docs_dir)
            if path.suffix.lower() in {".md", ".txt"}
        )

        self.assertIn("2026_电驱连续扫描表面形貌实验指导.md", source_names)
        self.assertIn("2026_时间FFT相位提取与形貌重建.md", source_names)
        self.assertIn("2026_表面形貌实验报告检查量规.md", source_names)
        self.assertNotIn("2026_迈克耳孙干涉仪实验指导_RAG整理版.md", source_names)
        self.assertIn("continuous_scan_temporal_fft", combined)
        self.assertIn("relative_surface_topography", combined)
        self.assertNotIn("钠黄光双线", combined)
        self.assertNotIn("白光干涉测透明介质", combined)

    def test_txt_and_pdf_sources_are_loaded(self) -> None:
        from rag_ingest import build_chunks_for_file

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs_dir = root / "rag_docs"
            index_dir = root / "rag_index"
            txt = docs_dir / "grading" / "评分细则.txt"
            pdf = docs_dir / "guidance" / "实验指导.pdf"
            txt.parent.mkdir(parents=True)
            pdf.parent.mkdir(parents=True)
            txt.write_text("报告必须包含原始数据、公式代入过程和误差分析。", encoding="utf-8")
            pdf.write_bytes(build_text_pdf("Michelson interferometer alignment steps"))
            config = self.make_config(docs_dir, index_dir)

            txt_chunks = build_chunks_for_file(txt, config)
            pdf_chunks = build_chunks_for_file(pdf, config)

        self.assertEqual(txt_chunks[0].metadata["task_type"], "report_check")
        self.assertIn("原始数据", txt_chunks[0].text)
        self.assertEqual(pdf_chunks[0].metadata["task_type"], "principle")
        self.assertIn("Michelson", pdf_chunks[0].text)
        self.assertEqual(pdf_chunks[0].metadata["page"], 1)

    def test_build_index_writes_collection_and_manifest(self) -> None:
        from rag_ingest import build_index

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs_dir = root / "rag_docs"
            index_dir = root / "rag_index"
            source = docs_dir / "faq" / "条纹问题FAQ.md"
            source.parent.mkdir(parents=True)
            source.write_text("# FAQ\n\n## 条纹模糊\n\n检查光束重合、反射镜角度和环境振动。", encoding="utf-8")
            config = self.make_config(docs_dir, index_dir)
            collection = FakeCollection()

            summary = build_index(
                config=config,
                collection=collection,
                embed_texts=lambda texts: [[float(len(text)), 0.0] for text in texts],
            )

            manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["indexed_chunks"], 1)
        self.assertEqual(len(collection.add_calls), 1)
        self.assertEqual(collection.add_calls[0]["documents"][0], "检查光束重合、反射镜角度和环境振动。")
        self.assertEqual(manifest["embedding_model"], "fake-embedding")
        self.assertEqual(manifest["chunk_size"], 80)
        self.assertEqual(manifest["chunk_overlap"], 10)
        self.assertEqual(manifest["indexed_chunks"], 1)

    def test_manifest_detects_document_changes(self) -> None:
        from rag_ingest import build_index, manifest_status

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs_dir = root / "rag_docs"
            index_dir = root / "rag_index"
            source = docs_dir / "faq" / "条纹问题FAQ.md"
            source.parent.mkdir(parents=True)
            source.write_text("# FAQ\n\n条纹问题。", encoding="utf-8")
            config = self.make_config(docs_dir, index_dir)
            build_index(
                config=config,
                collection=FakeCollection(),
                embed_texts=lambda texts: [[1.0] for _ in texts],
            )

            self.assertTrue(manifest_status(config)["current"])
            source.write_text("# FAQ\n\n条纹问题已更新。", encoding="utf-8")

            status = manifest_status(config)

        self.assertFalse(status["current"])
        self.assertEqual(status["reason"], "docs_changed")
