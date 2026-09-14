from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any


logger = logging.getLogger("ai_agent.images")
logger.addHandler(logging.NullHandler())

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
DEFAULT_IMAGES_DIR = ASSETS_DIR / "images"
DEFAULT_MANIFEST_PATH = ASSETS_DIR / "image_manifest.json"

STATIC_IMAGES_PREFIX = "/static/ai_agent/images"
ALLOWED_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif"})
TUTOR_MODE = "tutor"
REPORT_CHECK_MODE = "report_check"
WORKFLOW_MODE = "experiment_workflow"

COMPREHENSIVE_KEYWORDS = frozenset(
    {
        "综合",
        "整体",
        "系统",
        "完整",
        "全面",
        "从光路到",
        "从原理到",
        "串起来",
    }
)


def images_enabled() -> bool:
    value = os.getenv("AI_AGENT_IMAGES_ENABLED")
    if value is None:
        return True
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_str_list(value: Any) -> list[str]:
    return [item.strip() for item in _as_list(value) if isinstance(item, str) and item.strip()]


def _normalize_match_text(value: str) -> str:
    normalized = value.strip().lower().translate(
        str.maketrans(
            {
                "（": "(",
                "）": ")",
                "＝": "=",
                "－": "-",
                "／": "/",
            }
        )
    )
    normalized = normalized.replace("lambda", "λ")
    return re.sub(r"\s+", "", normalized)


def _normalize_match_terms(value: Any) -> list[str]:
    return [
        normalized
        for item in _as_str_list(value)
        if (normalized := _normalize_match_text(item))
    ]


def _normalize_match_rules(item: dict[str, Any]) -> list[dict[str, Any]]:
    normalized_rules: list[dict[str, Any]] = []
    for raw_rule in _as_list(item.get("match_rules")):
        if not isinstance(raw_rule, dict):
            continue

        any_of = _normalize_match_terms(raw_rule.get("any_of"))
        all_of = [
            terms
            for group in _as_list(raw_rule.get("all_of"))
            if (terms := _normalize_match_terms(group))
        ]
        if not any_of and not all_of:
            continue

        raw_score = raw_rule.get("score", 100)
        score = raw_score if isinstance(raw_score, int) and not isinstance(raw_score, bool) else 100
        normalized_rules.append(
            {
                "any_of": any_of,
                "all_of": all_of,
                "score": max(1, min(score, 1000)),
            }
        )

    if normalized_rules:
        return normalized_rules

    legacy_keywords = _normalize_match_terms(item.get("keywords"))
    if not legacy_keywords:
        return []
    return [{"any_of": legacy_keywords, "all_of": [], "score": 50}]


def _is_safe_filename(filename: Any) -> bool:
    if not isinstance(filename, str):
        return False
    cleaned = filename.strip()
    if not cleaned:
        return False
    if "://" in cleaned or cleaned.startswith("//"):
        return False
    if "/" in cleaned or "\\" in cleaned:
        return False
    path = Path(cleaned)
    if path.is_absolute() or ".." in path.parts:
        return False
    return path.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS


def _is_inside_directory(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def _build_image_url(filename: str) -> str:
    static_path = f"{STATIC_IMAGES_PREFIX}/{filename}"
    public_base_url = os.getenv("AI_AGENT_PUBLIC_BASE_URL", "").strip()
    if not public_base_url:
        return static_path
    return f"{public_base_url.rstrip('/')}{static_path}"


def _build_markdown(alt: str, filename: str) -> str:
    return f"![{alt}]({_build_image_url(filename)})"


def _load_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    try:
        if not manifest_path.exists():
            return []
        parsed = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(parsed, dict):
            items = parsed.get("images", [])
        else:
            items = parsed
        return [item for item in _as_list(items) if isinstance(item, dict)]
    except Exception as exc:
        logger.warning(
            "image manifest unavailable",
            extra={"reason": type(exc).__name__},
        )
        return []


def _normalize_manifest_item(item: dict[str, Any], images_dir: Path) -> dict[str, Any] | None:
    filename = item.get("filename")
    if not _is_safe_filename(filename):
        return None

    filename = str(filename).strip()
    image_path = images_dir / filename
    if not _is_inside_directory(image_path, images_dir):
        return None
    if not image_path.is_file():
        return None

    title = str(item.get("title") or item.get("id") or filename).strip()
    if not title:
        title = filename
    alt = str(item.get("alt") or title).strip() or title
    description = str(item.get("description") or "").strip()
    if len(description) > 80:
        description = description[:77].rstrip() + "..."

    concepts = _as_str_list(item.get("concepts"))
    match_rules = _normalize_match_rules(item)

    return {
        "id": str(item.get("id") or Path(filename).stem),
        "filename": filename,
        "title": title,
        "alt": alt,
        "concepts": concepts,
        "description": description,
        "match_rules": match_rules,
        "markdown": _build_markdown(alt, filename),
    }


def _default_limit(task_type: str, query_text: str) -> int:
    if task_type == REPORT_CHECK_MODE:
        return 1
    if task_type == TUTOR_MODE and any(keyword in query_text for keyword in COMPREHENSIVE_KEYWORDS):
        return 3
    return 2


def _score_image(item: dict[str, Any], normalized_text: str) -> tuple[int, int]:
    best_score = 0
    best_specificity = 0
    if not normalized_text:
        return best_score, best_specificity

    for rule in item.get("match_rules", []):
        any_of = rule.get("any_of", [])
        all_of = rule.get("all_of", [])
        matched_any = [term for term in any_of if term in normalized_text]
        matched_groups = [
            group
            for group in all_of
            if any(term in normalized_text for term in group)
        ]
        if any_of and not matched_any:
            continue
        if all_of and len(matched_groups) != len(all_of):
            continue
        if not any_of and not all_of:
            continue

        score = int(rule.get("score", 0))
        specificity = len(matched_any) + len(matched_groups)
        if (score, specificity) > (best_score, best_specificity):
            best_score, best_specificity = score, specificity

    return best_score, best_specificity


def select_teaching_images(
    query: str,
    *,
    task_type: str,
    manifest_path: Path | None = None,
    images_dir: Path | None = None,
    context_text: str = "",
    max_images: int | None = None,
) -> list[dict[str, Any]]:
    if not images_enabled():
        return []

    query_text = _normalize_match_text(query or "")
    context_match_text = _normalize_match_text(context_text or "")

    manifest = Path(manifest_path or DEFAULT_MANIFEST_PATH)
    image_root = Path(images_dir or DEFAULT_IMAGES_DIR)

    try:
        items = [
            normalized
            for raw_item in _load_manifest(manifest)
            if (normalized := _normalize_manifest_item(raw_item, image_root)) is not None
        ]
        scored: list[tuple[int, int, int, int, int, dict[str, Any]]] = []
        for index, item in enumerate(items):
            score, specificity = _score_image(item, query_text)
            if score <= 0:
                continue
            context_score, context_specificity = _score_image(item, context_match_text)
            scored.append(
                (
                    score,
                    specificity,
                    context_score,
                    context_specificity,
                    -index,
                    item,
                )
            )

        scored.sort(reverse=True)
        limit = max_images if max_images is not None else _default_limit(task_type, query_text)
        return [item for *_ranking, item in scored[: max(0, limit)]]
    except Exception as exc:
        logger.warning(
            "image selection failed",
            extra={"task_type": task_type, "reason": type(exc).__name__},
        )
        return []


def build_teaching_images_prompt(images: list[dict[str, Any]]) -> str:
    if not images:
        return ""

    lines = ["【可用教学图片】"]
    for image in images:
        concepts = "、".join(image.get("concepts", [])) or "相关概念"
        description = str(image.get("description", "")).strip()
        lines.append(
            f"- 标题：{image.get('title', '')}；适用概念：{concepts}；说明：{description}；Markdown：{image.get('markdown', '')}"
        )
    lines.extend(
        [
            "",
            "【插图规则】以上均为后端提供的通用教学示意，不是学生提交的图像或实验事实；必须将每条给出的 Markdown 恰好插入一次，且放在对应解释段落附近；只能使用上方图片，不得编造路径，不要统一放在结尾。",
        ]
    )
    return "\n".join(lines)
