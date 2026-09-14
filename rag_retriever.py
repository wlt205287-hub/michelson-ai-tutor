from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from rag_config import RagConfig, get_rag_config
from rag_ingest import manifest_status


logger = logging.getLogger("ai_agent.rag")
logger.addHandler(logging.NullHandler())


@dataclass(frozen=True)
class RagRetrievalResult:
    enabled: bool
    available: bool
    reliable: bool
    reason: str
    query: str
    results: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "available": self.available,
            "reliable": self.reliable,
            "reason": self.reason,
            "query": self.query,
            "results": self.results,
        }


def _result(
    *,
    enabled: bool,
    available: bool,
    reliable: bool,
    reason: str,
    query: str,
    results: list[dict[str, Any]] | None = None,
) -> RagRetrievalResult:
    return RagRetrievalResult(
        enabled=enabled,
        available=available,
        reliable=reliable,
        reason=reason,
        query=query,
        results=results or [],
    )


def _default_embed_query(model_name: str) -> Callable[[str], list[float]]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    def embed(text: str) -> list[float]:
        vector = model.encode([text], normalize_embeddings=True)[0]
        return list(map(float, vector))

    return embed


def _load_chroma_collection(config: RagConfig):
    import chromadb

    client = chromadb.PersistentClient(path=str(config.index_dir))
    return client.get_collection(config.collection_name)


def _score_from_distance(distance: float | int | None) -> float:
    if distance is None:
        return 0.0
    return 1.0 - float(distance)


def _flatten_query_response(response: dict[str, Any]) -> list[tuple[str, dict[str, Any], float]]:
    documents = (response.get("documents") or [[]])[0] or []
    metadatas = (response.get("metadatas") or [[]])[0] or []
    distances = (response.get("distances") or [[]])[0] or []
    flattened: list[tuple[str, dict[str, Any], float]] = []
    for index, text in enumerate(documents):
        metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
        distance = distances[index] if index < len(distances) else None
        flattened.append((text or "", dict(metadata), _score_from_distance(distance)))
    return flattened


def retrieve_rag_context(
    query: str,
    *,
    task_type: str,
    config: RagConfig | None = None,
    collection=None,
    embed_query: Callable[[str], list[float]] | None = None,
) -> RagRetrievalResult:
    config = config or get_rag_config()
    cleaned_query = (query or "").strip()
    if not config.enabled:
        logger.debug("rag retrieval skipped", extra={"enabled": False, "reason": "disabled"})
        return _result(enabled=False, available=False, reliable=False, reason="disabled", query=cleaned_query)
    if not cleaned_query:
        return _result(enabled=True, available=False, reliable=False, reason="empty_query", query=cleaned_query)

    allowed_sources = config.allowed_sources_for(task_type)
    if not allowed_sources:
        return _result(enabled=True, available=False, reliable=False, reason="task_not_allowed", query=cleaned_query)

    try:
        if collection is None:
            status = manifest_status(config)
            if not status.get("current"):
                reason = str(status.get("reason", "index_unavailable"))
                logger.info("rag index unavailable", extra={"reason": reason, "task_type": task_type})
                return _result(enabled=True, available=False, reliable=False, reason=reason, query=cleaned_query)
            collection = _load_chroma_collection(config)
        if embed_query is None:
            embed_query = _default_embed_query(config.embedding_model)

        response = collection.query(
            query_embeddings=[embed_query(cleaned_query)],
            n_results=config.top_k,
            where={"source_dir": {"$in": allowed_sources}},
        )
        seen: set[str] = set()
        results: list[dict[str, Any]] = []
        for text, metadata, score in _flatten_query_response(response):
            normalized = " ".join(text.split())
            if len(normalized) < config.min_text_chars or normalized in seen:
                continue
            if metadata.get("source_dir") not in allowed_sources:
                continue
            seen.add(normalized)
            results.append(
                {
                    "text": text.strip(),
                    "score": score,
                    "source_file": metadata.get("source_file", ""),
                    "source_path": metadata.get("source_path", ""),
                    "section": metadata.get("section", ""),
                    "task_type": metadata.get("task_type", ""),
                    "title": metadata.get("title", ""),
                    "source_type": metadata.get("source_type", ""),
                    "original_source_file": metadata.get("original_source_file", ""),
                    "measurement_method": metadata.get("measurement_method", ""),
                    "measurement_target": metadata.get("measurement_target", ""),
                    "version": metadata.get("version", ""),
                    "chunk_index": metadata.get("chunk_index", 0),
                }
            )

        if not results:
            reason = "no_results"
            reliable = False
        elif results[0]["score"] < config.min_score:
            reason = "low_confidence"
            reliable = False
        else:
            reason = "high_confidence_match"
            reliable = True
        logger.info(
            "rag retrieval completed",
            extra={
                "enabled": True,
                "available": True,
                "task_type": task_type,
                "reliable": reliable,
                "reason": reason,
                "result_count": len(results),
            },
        )
        return _result(
            enabled=True,
            available=True,
            reliable=reliable,
            reason=reason,
            query=cleaned_query,
            results=results,
        )
    except Exception as exc:
        logger.warning(
            "rag retrieval failed",
            extra={"task_type": task_type, "reason": "retrieval_error", "error": type(exc).__name__},
        )
        return _result(
            enabled=True,
            available=False,
            reliable=False,
            reason="retrieval_error",
            query=cleaned_query,
        )
