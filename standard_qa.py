from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any


logger = logging.getLogger("ai_agent.standard_qa")
logger.addHandler(logging.NullHandler())

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_STANDARD_QA_PATH = BASE_DIR / "assets" / "standard_qa.json"

STANDARD_QA_ID_PATTERN = re.compile(r"[a-z0-9_]+")
TERMINAL_PUNCTUATION = "?!.。！？"


@dataclass(frozen=True, slots=True)
class StandardQAMatch:
    item_id: str
    answer_markdown: str


def _normalize_question(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = re.sub(r"\s+", "", normalized)
    return normalized.rstrip(TERMINAL_PUNCTUATION)


def _as_valid_aliases(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [
        alias
        for alias in value
        if isinstance(alias, str) and _normalize_question(alias)
    ]


def _load_alias_index(qa_path: Path) -> dict[str, StandardQAMatch]:
    try:
        parsed = json.loads(qa_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(
            "standard qa unavailable",
            extra={"reason": type(exc).__name__},
        )
        return {}

    if not isinstance(parsed, dict) or not isinstance(parsed.get("items"), list):
        logger.warning("standard qa has invalid root structure")
        return {}

    alias_index: dict[str, StandardQAMatch] = {}
    ambiguous_aliases: set[str] = set()
    seen_ids: set[str] = set()

    for item in parsed["items"]:
        if not isinstance(item, dict):
            continue

        item_id = item.get("id")
        answer_markdown = item.get("answer_markdown")
        aliases = _as_valid_aliases(item.get("aliases"))
        if (
            not isinstance(item_id, str)
            or STANDARD_QA_ID_PATTERN.fullmatch(item_id) is None
            or item_id in seen_ids
            or not isinstance(answer_markdown, str)
            or not answer_markdown.strip()
            or not aliases
        ):
            continue

        seen_ids.add(item_id)
        match = StandardQAMatch(
            item_id=item_id,
            answer_markdown=answer_markdown,
        )
        for alias in aliases:
            normalized_alias = _normalize_question(alias)
            if normalized_alias in ambiguous_aliases:
                continue

            existing = alias_index.get(normalized_alias)
            if existing is None:
                alias_index[normalized_alias] = match
                continue
            if existing.item_id != item_id:
                alias_index.pop(normalized_alias, None)
                ambiguous_aliases.add(normalized_alias)

    if ambiguous_aliases:
        logger.warning(
            "standard qa contains ambiguous aliases",
            extra={"count": len(ambiguous_aliases)},
        )
    return alias_index


def find_standard_answer(
    question: str,
    *,
    qa_path: str | Path | None = None,
) -> StandardQAMatch | None:
    """Return an exact approved answer after conservative normalization."""
    try:
        normalized_question = _normalize_question(question or "")
        if not normalized_question:
            return None
        path = Path(qa_path or DEFAULT_STANDARD_QA_PATH)
        return _load_alias_index(path).get(normalized_question)
    except Exception as exc:
        logger.warning(
            "standard qa lookup skipped",
            extra={"reason": type(exc).__name__},
        )
        return None
