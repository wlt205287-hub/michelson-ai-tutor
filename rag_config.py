from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DOCS_DIR = BASE_DIR / "rag_docs"
DEFAULT_INDEX_DIR = BASE_DIR / "rag_index"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_int(value: str | None, default: int, minimum: int) -> int:
    try:
        parsed = int(value) if value is not None else default
    except ValueError:
        parsed = default
    return max(parsed, minimum)


def _parse_float(value: str | None, default: float) -> float:
    try:
        return float(value) if value is not None else default
    except ValueError:
        return default


@dataclass(frozen=True)
class RagConfig:
    enabled: bool = False
    docs_dir: Path = DEFAULT_DOCS_DIR
    index_dir: Path = DEFAULT_INDEX_DIR
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    chunk_size: int = 600
    chunk_overlap: int = 80
    top_k: int = 5
    min_score: float = 0.45
    min_text_chars: int = 4
    collection_name: str = "ai_agent_rag"
    task_source_map: dict[str, list[str]] = field(
        default_factory=lambda: {
            "tutor": ["guidance", "faq"],
            "report_check": ["grading", "examples", "faq"],
            "vision": ["guidance", "grading", "faq"],
        },
    )

    def allowed_sources_for(self, task_type: str) -> list[str]:
        return list(self.task_source_map.get(task_type, []))


def get_rag_config() -> RagConfig:
    load_dotenv(BASE_DIR / "agent_demo" / ".env")
    load_dotenv(BASE_DIR / ".env")

    docs_dir = Path(os.getenv("AI_AGENT_RAG_DOCS_DIR", str(DEFAULT_DOCS_DIR)))
    index_dir = Path(os.getenv("AI_AGENT_RAG_INDEX_DIR", str(DEFAULT_INDEX_DIR)))
    chunk_size = _parse_int(os.getenv("AI_AGENT_RAG_CHUNK_SIZE"), 600, 100)
    chunk_overlap = _parse_int(os.getenv("AI_AGENT_RAG_CHUNK_OVERLAP"), 80, 0)
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size // 5)

    return RagConfig(
        enabled=_parse_bool(os.getenv("AI_AGENT_RAG_ENABLED"), False),
        docs_dir=docs_dir,
        index_dir=index_dir,
        embedding_model=os.getenv("AI_AGENT_RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=_parse_int(os.getenv("AI_AGENT_RAG_TOP_K"), 5, 1),
        min_score=_parse_float(os.getenv("AI_AGENT_RAG_MIN_SCORE"), 0.45),
        min_text_chars=_parse_int(os.getenv("AI_AGENT_RAG_MIN_TEXT_CHARS"), 4, 1),
        collection_name=os.getenv("AI_AGENT_RAG_COLLECTION", "ai_agent_rag"),
    )
