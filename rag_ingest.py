from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Callable, Iterable, Sequence

from pypdf import PdfReader

from rag_config import RagConfig, get_rag_config


logger = logging.getLogger("ai_agent.rag")
logger.addHandler(logging.NullHandler())

SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}
MANIFEST_NAME = "manifest.json"
SOURCE_DIR_TASK_TYPES = {
    "guidance": "principle",
    "grading": "report_check",
    "faq": "faq",
    "examples": "report_check",
}


@dataclass(frozen=True)
class SourceDocument:
    text: str
    metadata: dict[str, object]


@dataclass(frozen=True)
class RagChunk:
    text: str
    metadata: dict[str, object]


def _normalize_text(text: str) -> str:
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    compact: list[str] = []
    blank = False
    for line in lines:
        if not line.strip():
            if not blank:
                compact.append("")
            blank = True
            continue
        compact.append(line)
        blank = False
    return "\n".join(compact).strip()


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    end_index = None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end_index = index
            break
    if end_index is None:
        return {}, text
    metadata: dict[str, str] = {}
    for line in lines[1:end_index]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip('"').strip("'")
    return metadata, "\n".join(lines[end_index + 1 :]).strip()


def _relative_source_path(path: Path, docs_dir: Path) -> str:
    try:
        return path.relative_to(docs_dir.parent).as_posix()
    except ValueError:
        return path.name


def _base_metadata(path: Path, docs_dir: Path, frontmatter: dict[str, str] | None = None) -> dict[str, object]:
    frontmatter = frontmatter or {}
    try:
        relative_to_docs = path.relative_to(docs_dir)
        source_dir = relative_to_docs.parts[0] if len(relative_to_docs.parts) > 1 else ""
    except ValueError:
        source_dir = ""
    return {
        "source_file": path.name,
        "source_path": _relative_source_path(path, docs_dir),
        "source_dir": source_dir,
        "task_type": frontmatter.get("task_type") or SOURCE_DIR_TASK_TYPES.get(source_dir, "faq"),
        "authority_level": frontmatter.get("authority_level") or "normal",
        "title": frontmatter.get("title") or path.stem,
        "source_type": frontmatter.get("source_type") or "curated_document",
        "original_source_file": frontmatter.get("source_file") or path.name,
        "measurement_method": frontmatter.get("measurement_method") or "",
        "measurement_target": frontmatter.get("measurement_target") or "",
        "version": frontmatter.get("version") or "",
    }


def _split_markdown_sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    heading_stack: list[str] = []
    current_lines: list[str] = []
    current_section = ""

    def flush() -> None:
        nonlocal current_lines, current_section
        content = _normalize_text("\n".join(current_lines))
        if content:
            sections.append((current_section, content))
        current_lines = []

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            marker, _, title = stripped.partition(" ")
            if set(marker) == {"#"} and title.strip():
                flush()
                level = len(marker)
                heading_stack[:] = heading_stack[: level - 1]
                heading_stack.append(title.strip())
                current_section = " > ".join(heading_stack)
                continue
        current_lines.append(line)
    flush()
    if sections:
        return sections
    normalized = _normalize_text(text)
    return [("", normalized)] if normalized else []


def _load_pdf_documents(path: Path, docs_dir: Path) -> list[SourceDocument]:
    documents: list[SourceDocument] = []
    reader = PdfReader(BytesIO(path.read_bytes()))
    base = _base_metadata(path, docs_dir)
    for page_index, page in enumerate(reader.pages, start=1):
        text = _normalize_text(page.extract_text() or "")
        if not text:
            continue
        metadata = dict(base)
        metadata["section"] = f"page {page_index}"
        metadata["page"] = page_index
        documents.append(SourceDocument(text=text, metadata=metadata))
    return documents


def load_source_documents(path: Path, docs_dir: Path) -> list[SourceDocument]:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        logger.info("rag_ingest skipped unsupported file", extra={"path": str(path)})
        return []
    try:
        if suffix == ".pdf":
            return _load_pdf_documents(path, docs_dir)

        raw_text = path.read_text(encoding="utf-8")
        frontmatter, body = _parse_frontmatter(raw_text) if suffix == ".md" else ({}, raw_text)
        base = _base_metadata(path, docs_dir, frontmatter)
        if suffix == ".md":
            documents: list[SourceDocument] = []
            for section, section_text in _split_markdown_sections(body):
                metadata = dict(base)
                metadata["section"] = frontmatter.get("section") or section or path.stem
                documents.append(SourceDocument(text=section_text, metadata=metadata))
            return documents
        metadata = dict(base)
        metadata["section"] = path.stem
        return [SourceDocument(text=_normalize_text(body), metadata=metadata)]
    except Exception as exc:
        logger.warning("rag_ingest skipped unreadable file", extra={"path": str(path), "error": type(exc).__name__})
        return []


def _split_long_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [chunk for chunk in chunks if chunk]


def chunk_document(document: SourceDocument, config: RagConfig) -> list[RagChunk]:
    paragraphs = [part.strip() for part in document.text.split("\n\n") if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs or [document.text.strip()]:
        if len(paragraph) > config.chunk_size:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(_split_long_text(paragraph, config.chunk_size, config.chunk_overlap))
            continue
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= config.chunk_size:
            current = candidate
        else:
            chunks.append(current.strip())
            overlap_text = current[-config.chunk_overlap :].strip() if config.chunk_overlap else ""
            current = f"{overlap_text}\n\n{paragraph}".strip() if overlap_text else paragraph
    if current:
        chunks.append(current.strip())

    result: list[RagChunk] = []
    for chunk in chunks:
        normalized = _normalize_text(chunk)
        if len(normalized) < config.min_text_chars:
            continue
        result.append(RagChunk(text=normalized, metadata=dict(document.metadata)))
    return result


def build_chunks_for_file(path: Path, config: RagConfig) -> list[RagChunk]:
    chunks: list[RagChunk] = []
    for document in load_source_documents(path, config.docs_dir):
        for chunk in chunk_document(document, config):
            metadata = dict(chunk.metadata)
            metadata["chunk_index"] = len(chunks)
            chunks.append(RagChunk(text=chunk.text, metadata=metadata))
    return chunks


def iter_source_files(docs_dir: Path) -> Iterable[Path]:
    if not docs_dir.exists():
        return []
    return (
        path
        for path in sorted(docs_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def compute_docs_hash(docs_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in iter_source_files(docs_dir):
        try:
            relative = path.relative_to(docs_dir).as_posix()
        except ValueError:
            relative = path.name
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _manifest_path(config: RagConfig) -> Path:
    return config.index_dir / MANIFEST_NAME


def _build_manifest(config: RagConfig, docs_hash: str, indexed_chunks: int) -> dict[str, object]:
    return {
        "embedding_model": config.embedding_model,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "docs_hash": docs_hash,
        "indexed_chunks": indexed_chunks,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def manifest_status(config: RagConfig) -> dict[str, object]:
    path = _manifest_path(config)
    if not path.exists():
        return {"current": False, "reason": "manifest_missing"}
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"current": False, "reason": "manifest_unreadable"}
    if manifest.get("embedding_model") != config.embedding_model:
        return {"current": False, "reason": "embedding_model_changed"}
    if manifest.get("chunk_size") != config.chunk_size:
        return {"current": False, "reason": "chunk_size_changed"}
    if manifest.get("chunk_overlap") != config.chunk_overlap:
        return {"current": False, "reason": "chunk_overlap_changed"}
    if manifest.get("docs_hash") != compute_docs_hash(config.docs_dir):
        return {"current": False, "reason": "docs_changed"}
    return {"current": True, "reason": "current", "manifest": manifest}


def _default_embed_texts(model_name: str) -> Callable[[Sequence[str]], list[list[float]]]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    def embed(texts: Sequence[str]) -> list[list[float]]:
        vectors = model.encode(list(texts), normalize_embeddings=True)
        return [list(map(float, vector)) for vector in vectors]

    return embed


def _create_chroma_collection(config: RagConfig):
    import chromadb

    client = chromadb.PersistentClient(path=str(config.index_dir))
    try:
        client.delete_collection(config.collection_name)
    except Exception:
        pass
    return client.create_collection(
        name=config.collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def build_index(
    *,
    config: RagConfig | None = None,
    collection=None,
    embed_texts: Callable[[Sequence[str]], list[list[float]]] | None = None,
) -> dict[str, object]:
    config = config or get_rag_config()
    config.index_dir.mkdir(parents=True, exist_ok=True)

    chunks: list[RagChunk] = []
    for path in iter_source_files(config.docs_dir):
        chunks.extend(build_chunks_for_file(path, config))

    if collection is None:
        collection = _create_chroma_collection(config)
    if embed_texts is None:
        embed_texts = _default_embed_texts(config.embedding_model)

    if chunks:
        documents = [chunk.text for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        ids = [f"chunk-{index:06d}" for index in range(len(chunks))]
        embeddings = embed_texts(documents)
        collection.add(documents=documents, metadatas=metadatas, ids=ids, embeddings=embeddings)

    docs_hash = compute_docs_hash(config.docs_dir)
    manifest = _build_manifest(config, docs_hash, len(chunks))
    _manifest_path(config).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("rag index built", extra={"indexed_chunks": len(chunks)})
    return {"indexed_chunks": len(chunks), "manifest": manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or inspect the local RAG index.")
    parser.add_argument("command", choices=["build", "status"])
    args = parser.parse_args()
    config = get_rag_config()
    if args.command == "status":
        print(json.dumps(manifest_status(config), ensure_ascii=False, indent=2))
        return
    print(json.dumps(build_index(config=config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
