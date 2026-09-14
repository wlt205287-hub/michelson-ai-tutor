from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


class RagConfigTests(TestCase):
    def test_default_config_is_disabled_and_uses_project_paths(self) -> None:
        from rag_config import get_rag_config

        with patch.dict(os.environ, {}, clear=True):
            config = get_rag_config()

        self.assertFalse(config.enabled)
        self.assertEqual(config.docs_dir, PROJECT_DIR / "rag_docs")
        self.assertEqual(config.index_dir, PROJECT_DIR / "rag_index")
        self.assertEqual(config.embedding_model, "BAAI/bge-small-zh-v1.5")
        self.assertEqual(config.chunk_size, 600)
        self.assertEqual(config.chunk_overlap, 80)
        self.assertEqual(config.top_k, 5)
        self.assertEqual(config.min_score, 0.45)

    def test_environment_overrides_are_parsed(self) -> None:
        from rag_config import get_rag_config

        env = {
            "AI_AGENT_RAG_ENABLED": "true",
            "AI_AGENT_RAG_EMBEDDING_MODEL": "local-model",
            "AI_AGENT_RAG_CHUNK_SIZE": "500",
            "AI_AGENT_RAG_CHUNK_OVERLAP": "70",
            "AI_AGENT_RAG_TOP_K": "3",
            "AI_AGENT_RAG_MIN_SCORE": "0.66",
        }
        with patch.dict(os.environ, env, clear=True):
            config = get_rag_config()

        self.assertTrue(config.enabled)
        self.assertEqual(config.embedding_model, "local-model")
        self.assertEqual(config.chunk_size, 500)
        self.assertEqual(config.chunk_overlap, 70)
        self.assertEqual(config.top_k, 3)
        self.assertEqual(config.min_score, 0.66)

    def test_task_source_mapping_keeps_rag_supplemental(self) -> None:
        from rag_config import get_rag_config

        config = get_rag_config()

        self.assertIn("guidance", config.allowed_sources_for("tutor"))
        self.assertIn("faq", config.allowed_sources_for("tutor"))
        self.assertIn("grading", config.allowed_sources_for("report_check"))
        self.assertIn("examples", config.allowed_sources_for("report_check"))
        self.assertIn("grading", config.allowed_sources_for("vision"))
