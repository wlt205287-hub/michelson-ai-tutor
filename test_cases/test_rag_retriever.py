from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest import TestCase


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


class FakeCollection:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.last_where = None

    def query(self, **kwargs) -> dict:
        self.last_where = kwargs.get("where")
        return self.payload


class BrokenCollection:
    def query(self, **kwargs) -> dict:
        raise RuntimeError("boom")


class RagRetrieverTests(TestCase):
    def make_config(self, enabled: bool = True, min_score: float = 0.45):
        from rag_config import RagConfig

        root = Path(tempfile.mkdtemp())
        docs_dir = root / "rag_docs"
        index_dir = root / "rag_index"
        index_dir.mkdir(parents=True)
        return RagConfig(
            enabled=enabled,
            docs_dir=docs_dir,
            index_dir=index_dir,
            embedding_model="fake-embedding",
            chunk_size=600,
            chunk_overlap=80,
            top_k=3,
            min_score=min_score,
        )

    def test_disabled_rag_returns_fallback_state(self) -> None:
        from rag_retriever import retrieve_rag_context

        result = retrieve_rag_context("条纹模糊怎么办？", task_type="tutor", config=self.make_config(enabled=False))

        self.assertFalse(result.enabled)
        self.assertFalse(result.reliable)
        self.assertEqual(result.reason, "disabled")
        self.assertEqual(result.results, [])

    def test_reliable_result_requires_score_text_and_allowed_task(self) -> None:
        from rag_retriever import retrieve_rag_context

        collection = FakeCollection(
            {
                "ids": [["a", "b"]],
                "documents": [["检查光束重合和反射镜角度。", "报告评分应检查公式和单位。"]],
                "metadatas": [[
                    {"source_file": "条纹问题FAQ.md", "section": "条纹模糊", "task_type": "operation", "source_dir": "faq"},
                    {"source_file": "评分细则.md", "section": "评分", "task_type": "report_check", "source_dir": "grading"},
                ]],
                "distances": [[0.1, 0.2]],
            }
        )

        result = retrieve_rag_context(
            "条纹模糊怎么办？",
            task_type="tutor",
            config=self.make_config(),
            collection=collection,
            embed_query=lambda text: [0.1, 0.2],
        )

        self.assertTrue(result.enabled)
        self.assertTrue(result.available)
        self.assertTrue(result.reliable)
        self.assertEqual(result.reason, "high_confidence_match")
        self.assertEqual(len(result.results), 1)
        self.assertEqual(result.results[0]["source_file"], "条纹问题FAQ.md")
        self.assertEqual(collection.last_where["source_dir"]["$in"], ["guidance", "faq"])

    def test_low_score_or_exception_returns_unreliable_without_raising(self) -> None:
        from rag_retriever import retrieve_rag_context

        low_score = retrieve_rag_context(
            "条纹模糊怎么办？",
            task_type="tutor",
            config=self.make_config(min_score=0.95),
            collection=FakeCollection(
                {
                    "ids": [["a"]],
                    "documents": [["检查光束重合。"]],
                    "metadatas": [[{"source_file": "FAQ.md", "section": "条纹", "task_type": "operation", "source_dir": "faq"}]],
                    "distances": [[0.2]],
                }
            ),
            embed_query=lambda text: [0.1],
        )
        broken = retrieve_rag_context(
            "条纹模糊怎么办？",
            task_type="tutor",
            config=self.make_config(),
            collection=BrokenCollection(),
            embed_query=lambda text: [0.1],
        )

        self.assertFalse(low_score.reliable)
        self.assertEqual(low_score.reason, "low_confidence")
        self.assertFalse(broken.reliable)
        self.assertEqual(broken.reason, "retrieval_error")

    def test_duplicate_results_are_removed(self) -> None:
        from rag_retriever import retrieve_rag_context

        payload = {
            "ids": [["a", "b"]],
            "documents": [["检查光束重合。", "检查光束重合。"]],
            "metadatas": [[
                {"source_file": "FAQ.md", "section": "条纹", "task_type": "operation", "source_dir": "faq"},
                {"source_file": "FAQ.md", "section": "条纹", "task_type": "operation", "source_dir": "faq"},
            ]],
            "distances": [[0.1, 0.11]],
        }

        result = retrieve_rag_context(
            "条纹模糊怎么办？",
            task_type="tutor",
            config=self.make_config(),
            collection=FakeCollection(payload),
            embed_query=lambda text: [0.1],
        )

        self.assertTrue(result.reliable)
        self.assertEqual(len(result.results), 1)
