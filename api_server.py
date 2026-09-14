from __future__ import annotations

import asyncio
import logging
import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent_core import (
    GUIDED_TUTOR_ROUTE,
    REPORT_CHECK_MODE,
    TUTOR_MODE,
    _trim_prompt_history,
    ask_tutor,
    build_frontend_payload,
    build_default_teaching_decision,
    build_rag_augmented_knowledge,
    build_report_prompt,
    build_report_rag_query,
    build_system_prompt,
    build_tutor_prompt,
    check_report,
    create_client,
    format_sse_event,
    get_stage_instruction,
    get_teaching_instruction,
    iter_model_stream_chunks,
    normalize_model_output,
    format_saved_content,
    load_resources,
    load_text_file,
    prepare_tutor_conversation_state,
    resolve_tutor_stage_and_route,
    resolve_teaching_decision,
    save_output,
    stream_report_events,
    stream_tutor_events,
)
from document_loader import (
    MAX_PDF_SIZE_BYTES,
    MAX_VISION_IMAGES_PER_REQUEST,
    PdfParseError,
    PdfTooManyPagesError,
    extract_embedded_images,
    extract_pdf_text,
    rasterize_scanned_pdf,
)
from image_assets import DEFAULT_IMAGES_DIR, STATIC_IMAGES_PREFIX
from vision_client import (
    VisionImage,
    call_vision_model,
    create_vision_client,
    iter_vision_model_stream_chunks,
)


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / "agent_demo" / ".env")
load_dotenv(BASE_DIR / ".env")


logger = logging.getLogger("ai_agent")


def configure_logging() -> None:
    level_name = os.getenv("AI_AGENT_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    if not isinstance(level, int):
        level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


configure_logging()


def format_log_message(event: str, context: dict[str, Any]) -> str:
    keys = (
        "mode",
        "stage",
        "route",
        "teaching_policy",
        "guidance_level",
        "error_code",
    )
    details = " ".join(f"{key}={context[key]}" for key in keys if key in context)
    return f"{event} {details}" if details else event


def build_upload_log_context(
    stage: str,
    error_code: str,
    started_at: float,
    page_count: int = 0,
    embedded_image_count: int = 0,
) -> dict[str, Any]:
    return {
        "mode": "report_check",
        "stage": stage,
        "route": "upload",
        "error_code": error_code,
        "page_count": page_count,
        "embedded_image_count": embedded_image_count,
        "duration_ms": int((time.perf_counter() - started_at) * 1000),
    }


def build_upload_error_payload(
    stage: str,
    error_code: str,
    message: str,
) -> dict[str, Any]:
    return {
        "success": False,
        "mode": "report_check",
        "stage": stage,
        "display_type": "markdown",
        "answer_markdown": "",
        "saved": False,
        "warning": message,
        "error_code": error_code,
        "message": message,
    }


def vision_enabled() -> bool:
    return os.getenv("ENABLE_VISION", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def build_vision_log_context(
    mode: str,
    stage: str,
    route: str,
    error_code: str,
    started_at: float,
    page_count: int = 0,
    embedded_image_count: int = 0,
    image_count: int = 0,
    image_format: str | None = None,
    image_bytes_total: int = 0,
    teaching_policy: str | None = None,
    guidance_level: int | None = None,
) -> dict[str, Any]:
    context: dict[str, Any] = {
        "mode": mode,
        "stage": stage,
        "route": route,
        "error_code": error_code,
        "duration_ms": int((time.perf_counter() - started_at) * 1000),
        "image_count": image_count,
        "image_bytes_total": image_bytes_total,
    }
    if route == VISION_TUTOR_ROUTE and image_format:
        context["image_format"] = image_format
    if route == VISION_REPORT_ROUTE:
        context["page_count"] = page_count
        context["embedded_image_count"] = embedded_image_count
    if teaching_policy is not None:
        context["teaching_policy"] = teaching_policy
    if guidance_level is not None:
        context["guidance_level"] = guidance_level
    return context


def build_vision_error_payload(
    mode: str,
    stage: str,
    route: str,
    error_code: str,
    message: str,
) -> dict[str, Any]:
    payload = build_frontend_payload(
        mode,
        "",
        {
            "success": False,
            "stage": stage,
            "saved": False,
            "warning": message,
            "need_student_reply": mode == TUTOR_MODE,
        },
    )
    payload["route"] = route
    payload["error_code"] = error_code
    payload["message"] = message
    return payload


def build_document_info(extraction) -> dict[str, Any]:
    return {
        "page_count": extraction.page_count,
        "extracted_text_chars": len(extraction.text.strip()),
        "embedded_image_count": extraction.embedded_image_count,
        "suspected_scanned": extraction.suspected_scanned,
    }


def parse_conversation_history(raw_history: str | None) -> tuple[list[dict[str, str]], str | None]:
    if not raw_history or not raw_history.strip():
        return [], None
    try:
        parsed = json.loads(raw_history)
        if not isinstance(parsed, list):
            raise ValueError("conversation_history must be a list")
        history: list[dict[str, str]] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            content = item.get("content")
            if isinstance(role, str) and isinstance(content, str):
                history.append({"role": role, "content": content})
        return history, None
    except (TypeError, ValueError, json.JSONDecodeError):
        return [], "历史对话格式无效，已忽略历史上下文。"


def build_vision_resources(prompt_path: Path) -> tuple[dict[str, str], str]:
    resources = load_resources()
    return resources, load_text_file(prompt_path)


class VisionConfigurationError(Exception):
    """Raised when the vision client cannot be configured."""


app = FastAPI(title="AI Agent Backend", version="0.2.0")
app.mount(
    STATIC_IMAGES_PREFIX,
    StaticFiles(directory=str(DEFAULT_IMAGES_DIR), check_dir=False),
    name="ai_agent_images",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


SSE_HEADERS = {
    "Content-Type": "text/event-stream; charset=utf-8",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}

VISION_TUTOR_PROMPT_PATH = BASE_DIR / "prompts" / "vision_tutor_prompt.txt"
VISION_REPORT_PROMPT_PATH = BASE_DIR / "prompts" / "vision_report_prompt.txt"
VISION_TUTOR_ROUTE = "vision_tutor"
VISION_REPORT_ROUTE = "vision_report"
VISION_CHAT_STREAM_MODE = "vision_chat"
REPORT_CHECK_VISION_STREAM_MODE = "report_check_vision"
VISION_STREAM_STAGE = "generation"
SINGLE_IMAGE_VISION_STREAM_ROUTE = "single_image_vision"
REPORT_IMAGE_VISION_STREAM_ROUTE = "report_image_vision"
PDF_IMAGE_VISION_STREAM_ROUTE = "pdf_image_vision"
PDF_TEXT_FALLBACK_STREAM_ROUTE = "pdf_text_fallback"
SUPPORTED_IMAGE_MIME_TYPES = frozenset({"image/png", "image/jpeg", "image/webp"})
MAX_VISION_IMAGE_SIZE_BYTES = 5 * 1024 * 1024
MAX_STREAM_VISION_IMAGE_SIZE_BYTES = 15 * 1024 * 1024
MAX_STREAM_PDF_SIZE_BYTES = 30 * 1024 * 1024
MAX_STREAM_VISION_IMAGES_PER_REQUEST = 12
MAX_QUIZ_CONTEXT_CHARS = 32 * 1024


class ChatRequest(BaseModel):
    session_id: str = Field(default="", description="前端会话标识，当前阶段仅透传保留")
    stage: str = Field(default="unknown", description="学习阶段，例如 pre_lab")
    student_input: str = Field(..., min_length=1, description="学生本轮问题")
    conversation_history: list[dict[str, str]] = Field(
        default_factory=list,
        description="历史对话，元素建议包含 role 和 content",
    )
    teaching_policy: str | None = Field(
        default=None,
        description="Optional teaching policy override: direct, guided, or hybrid",
    )
    guidance_level: int | None = Field(
        default=None,
        description="Optional guidance level override from 0 to 3",
    )


class ReportCheckRequest(BaseModel):
    session_id: str = Field(default="", description="前端会话标识，当前阶段仅透传保留")
    stage: str = Field(default="report", description="报告检查阶段")
    report_content: str = Field(..., min_length=1, description="学生报告片段")
    quiz_context: str | None = Field(
        default=None,
        max_length=MAX_QUIZ_CONTEXT_CHARS,
        description="课前预习习题作答与标准答案上下文",
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "ai_agent",
    }


@app.post("/api/agent/chat")
def agent_chat(request: ChatRequest) -> dict[str, Any]:
    try:
        return ask_tutor(
            student_input=request.student_input,
            stage=request.stage,
            conversation_history=request.conversation_history,
            save=True,
            stream=False,
            teaching_policy=request.teaching_policy,
            guidance_level=request.guidance_level,
        )
    except Exception:
        context = {
            "mode": "tutor",
            "stage": request.stage,
            "route": "unknown",
            "error_code": "AGENT_RUNTIME_ERROR",
        }
        logger.exception(
            format_log_message("agent_chat request failed", context),
            extra=context,
        )
        return {
            "success": False,
            "error_code": "AGENT_RUNTIME_ERROR",
            "message": "Agent 处理失败，请稍后重试。",
        }



def build_vision_stream_meta(
    mode: str,
    route: str,
    document_info: dict[str, Any] | None = None,
    vision_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "success": True,
        "mode": mode,
        "route": route,
        "stage": VISION_STREAM_STAGE,
        "display_type": "markdown",
    }
    if document_info is not None:
        payload["document_info"] = document_info
    if vision_info is not None:
        payload["vision_info"] = vision_info
    return payload


def build_vision_stream_done(answer_markdown: str) -> dict[str, Any]:
    return {
        "success": True,
        "answer_markdown": answer_markdown,
    }


def build_vision_stream_error(error_code: str, message: str) -> dict[str, Any]:
    return {
        "success": False,
        "error_code": error_code,
        "message": message,
    }


def close_stream_iterator(iterator: Any) -> None:
    close = getattr(iterator, "close", None)
    if callable(close):
        close()


def iter_chat_vision_stream_events(
    *,
    stage: str,
    student_input: str,
    system_prompt: str,
    user_prompt: str,
    image: VisionImage,
    vision_info: dict[str, Any],
    started_at: float,
    teaching_policy: str | None = None,
    guidance_level: int | None = None,
):
    stream = None
    yield {
        "event": "meta",
        "data": build_vision_stream_meta(
            VISION_CHAT_STREAM_MODE,
            SINGLE_IMAGE_VISION_STREAM_ROUTE,
            vision_info=vision_info,
        ),
    }
    try:
        vision_client = create_vision_client()
        stream = iter_vision_model_stream_chunks(
            vision_client,
            system_prompt,
            user_prompt,
            [image],
            temperature=0.3,
        )
        chunks: list[str] = []
        for chunk in stream:
            chunks.append(chunk)
            yield {"event": "delta", "data": {"text": chunk}}

        markdown = normalize_model_output("".join(chunks).strip())
        save_output(
            format_saved_content("视觉问答", student_input, markdown),
            "tutor_chat",
        )
        context = build_vision_log_context(
            VISION_CHAT_STREAM_MODE,
            stage,
            SINGLE_IMAGE_VISION_STREAM_ROUTE,
            "OK",
            started_at,
            image_count=1,
            image_format=image.mime,
            image_bytes_total=len(image.bytes_data),
            teaching_policy=teaching_policy,
            guidance_level=guidance_level,
        )
        logger.info(format_log_message("chat_vision_stream completed", context), extra=context)
        yield {"event": "done", "data": build_vision_stream_done(markdown)}
    except (GeneratorExit, asyncio.CancelledError):
        context = build_vision_log_context(
            VISION_CHAT_STREAM_MODE,
            stage,
            SINGLE_IMAGE_VISION_STREAM_ROUTE,
            "CANCELLED",
            started_at,
            image_count=1,
            image_format=image.mime,
            image_bytes_total=len(image.bytes_data),
            teaching_policy=teaching_policy,
            guidance_level=guidance_level,
        )
        logger.info(format_log_message("chat_vision_stream cancelled", context), extra=context)
        raise
    except Exception:
        context = build_vision_log_context(
            VISION_CHAT_STREAM_MODE,
            stage,
            SINGLE_IMAGE_VISION_STREAM_ROUTE,
            "VISION_API_ERROR",
            started_at,
            image_count=1,
            image_format=image.mime,
            image_bytes_total=len(image.bytes_data),
            teaching_policy=teaching_policy,
            guidance_level=guidance_level,
        )
        logger.exception(format_log_message("chat_vision_stream failed", context), extra=context)
        yield {
            "event": "error",
            "data": build_vision_stream_error(
                "VISION_API_ERROR",
                "视觉模型暂时不可用，请稍后重试。",
            ),
        }
    finally:
        if stream is not None:
            close_stream_iterator(stream)


@app.post("/api/agent/chat/vision/stream")
async def agent_chat_vision_stream(
    image: UploadFile | None = File(default=None),
    session_id: str | None = Form(default=None),
    stage: str = Form(default="unknown"),
    student_input: str | None = Form(default=None),
    conversation_history: str | None = Form(default="[]"),
    teaching_policy: str | None = Form(default=None),
    guidance_level: int | None = Form(default=None),
):
    started_at = time.perf_counter()

    if not vision_enabled():
        payload = build_vision_error_payload(
            TUTOR_MODE,
            stage,
            VISION_TUTOR_ROUTE,
            "VISION_DISABLED",
            "视觉能力未启用。",
        )
        context = build_vision_log_context(
            VISION_CHAT_STREAM_MODE,
            stage,
            SINGLE_IMAGE_VISION_STREAM_ROUTE,
            "VISION_DISABLED",
            started_at,
        )
        logger.warning(format_log_message("chat_vision_stream disabled", context), extra=context)
        return JSONResponse(status_code=404, content=payload)

    if image is None or not session_id or not session_id.strip():
        return build_vision_error_payload(
            TUTOR_MODE,
            stage,
            VISION_TUTOR_ROUTE,
            "VALIDATION_ERROR",
            "请上传图片并提供 session_id。",
        )

    if not student_input or not student_input.strip():
        return build_vision_error_payload(
            TUTOR_MODE,
            stage,
            VISION_TUTOR_ROUTE,
            "VALIDATION_ERROR",
            "问题内容不能为空。",
        )

    image_format = image.content_type or ""
    if image_format not in SUPPORTED_IMAGE_MIME_TYPES:
        return build_vision_error_payload(
            TUTOR_MODE,
            stage,
            VISION_TUTOR_ROUTE,
            "IMAGE_FORMAT_UNSUPPORTED",
            "仅支持 PNG、JPEG 或 WebP 图片。",
        )

    image_bytes = await image.read()
    if not image_bytes:
        return build_vision_error_payload(
            TUTOR_MODE,
            stage,
            VISION_TUTOR_ROUTE,
            "VALIDATION_ERROR",
            "图片文件不能为空。",
        )

    if len(image_bytes) > MAX_STREAM_VISION_IMAGE_SIZE_BYTES:
        return build_vision_error_payload(
            TUTOR_MODE,
            stage,
            VISION_TUTOR_ROUTE,
            "IMAGE_TOO_LARGE",
            "单张图片不能超过 15MB。",
        )

    history, _history_warning = parse_conversation_history(conversation_history)
    trimmed_history = _trim_prompt_history(history)
    try:
        resources, vision_template = build_vision_resources(VISION_TUTOR_PROMPT_PATH)
        effective_stage, tutor_route = resolve_tutor_stage_and_route(
            stage,
            student_input.strip(),
            history,
            classifier_template=resources.get("stage_classifier_prompt"),
        )
        teaching_decision = build_default_teaching_decision()
        if tutor_route == GUIDED_TUTOR_ROUTE:
            teaching_decision = resolve_teaching_decision(
                effective_stage,
                student_input.strip(),
                history,
                classifier_template=resources.get(
                    "teaching_policy_classifier_prompt"
                ),
                allow_model_fallback=False,
                teaching_policy=teaching_policy,
                guidance_level=guidance_level,
            )
        system_prompt = build_system_prompt(resources)
        knowledge_base = build_rag_augmented_knowledge(
            resources["knowledge_base"],
            student_input.strip(),
            task_type="vision",
        )
        state = prepare_tutor_conversation_state(
            None,
            effective_stage,
            stage,
            history,
            teaching_decision,
        )
        user_prompt = build_tutor_prompt(
            vision_template,
            knowledge_base,
            student_input.strip(),
            {
                **state,
                "route": SINGLE_IMAGE_VISION_STREAM_ROUTE,
                "turn_count": len(history),
            },
            trimmed_history,
            get_stage_instruction(resources, effective_stage),
            get_teaching_instruction(resources, teaching_decision),
        )
    except Exception:
        context = build_vision_log_context(
            VISION_CHAT_STREAM_MODE,
            stage,
            SINGLE_IMAGE_VISION_STREAM_ROUTE,
            "AGENT_RUNTIME_ERROR",
            started_at,
            image_count=1,
            image_format=image_format,
            image_bytes_total=len(image_bytes),
        )
        logger.exception(format_log_message("chat_vision_stream setup failed", context), extra=context)
        return build_vision_error_payload(
            TUTOR_MODE,
            stage,
            VISION_TUTOR_ROUTE,
            "AGENT_RUNTIME_ERROR",
            "Agent 处理失败，请稍后重试。",
        )

    image_payload = VisionImage(bytes_data=image_bytes, mime=image_format)
    vision_info = {
        "image_count": 1,
        "image_format": image_format,
        "image_bytes": len(image_bytes),
    }
    events = iter_chat_vision_stream_events(
        stage=effective_stage,
        student_input=student_input.strip(),
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        image=image_payload,
        vision_info=vision_info,
        started_at=started_at,
        teaching_policy=teaching_decision["policy"],
        guidance_level=teaching_decision["guidance_level"],
    )
    return StreamingResponse(
        iter_sse_response(events),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
@app.post("/api/agent/chat/vision")
async def agent_chat_vision(
    image: UploadFile | None = File(default=None),
    session_id: str | None = Form(default=None),
    stage: str = Form(default="unknown"),
    student_input: str | None = Form(default=None),
    conversation_history: str | None = Form(default="[]"),
):
    started_at = time.perf_counter()

    if not vision_enabled():
        payload = build_vision_error_payload(
            TUTOR_MODE,
            stage,
            VISION_TUTOR_ROUTE,
            "VISION_DISABLED",
            "视觉能力未启用。",
        )
        context = build_vision_log_context(
            TUTOR_MODE,
            stage,
            VISION_TUTOR_ROUTE,
            "VISION_DISABLED",
            started_at,
        )
        logger.warning(format_log_message("chat_vision disabled", context), extra=context)
        return JSONResponse(status_code=404, content=payload)

    if image is None or not session_id or not session_id.strip():
        context = build_vision_log_context(
            TUTOR_MODE,
            stage,
            VISION_TUTOR_ROUTE,
            "VALIDATION_ERROR",
            started_at,
        )
        logger.warning(format_log_message("chat_vision validation failed", context), extra=context)
        return build_vision_error_payload(
            TUTOR_MODE,
            stage,
            VISION_TUTOR_ROUTE,
            "VALIDATION_ERROR",
            "请上传图片并提供 session_id。",
        )

    if not student_input or not student_input.strip():
        context = build_vision_log_context(
            TUTOR_MODE,
            stage,
            VISION_TUTOR_ROUTE,
            "VALIDATION_ERROR",
            started_at,
            image_format=image.content_type,
        )
        logger.warning(format_log_message("chat_vision validation failed", context), extra=context)
        return build_vision_error_payload(
            TUTOR_MODE,
            stage,
            VISION_TUTOR_ROUTE,
            "VALIDATION_ERROR",
            "问题内容不能为空。",
        )

    image_format = image.content_type or ""
    if image_format not in SUPPORTED_IMAGE_MIME_TYPES:
        context = build_vision_log_context(
            TUTOR_MODE,
            stage,
            VISION_TUTOR_ROUTE,
            "IMAGE_FORMAT_UNSUPPORTED",
            started_at,
            image_format=image_format,
        )
        logger.warning(format_log_message("chat_vision validation failed", context), extra=context)
        return build_vision_error_payload(
            TUTOR_MODE,
            stage,
            VISION_TUTOR_ROUTE,
            "IMAGE_FORMAT_UNSUPPORTED",
            "仅支持 PNG、JPEG 或 WebP 图片。",
        )

    image_bytes = await image.read()
    if not image_bytes:
        context = build_vision_log_context(
            TUTOR_MODE,
            stage,
            VISION_TUTOR_ROUTE,
            "VALIDATION_ERROR",
            started_at,
            image_format=image_format,
        )
        logger.warning(format_log_message("chat_vision validation failed", context), extra=context)
        return build_vision_error_payload(
            TUTOR_MODE,
            stage,
            VISION_TUTOR_ROUTE,
            "VALIDATION_ERROR",
            "图片文件不能为空。",
        )

    if len(image_bytes) > MAX_VISION_IMAGE_SIZE_BYTES:
        context = build_vision_log_context(
            TUTOR_MODE,
            stage,
            VISION_TUTOR_ROUTE,
            "IMAGE_TOO_LARGE",
            started_at,
            image_count=1,
            image_format=image_format,
            image_bytes_total=len(image_bytes),
        )
        logger.warning(format_log_message("chat_vision validation failed", context), extra=context)
        return build_vision_error_payload(
            TUTOR_MODE,
            stage,
            VISION_TUTOR_ROUTE,
            "IMAGE_TOO_LARGE",
            "单张图片不能超过 5MB。",
        )

    history, history_warning = parse_conversation_history(conversation_history)
    trimmed_history = _trim_prompt_history(history)

    try:
        resources, vision_template = build_vision_resources(VISION_TUTOR_PROMPT_PATH)
        effective_stage, tutor_route = resolve_tutor_stage_and_route(
            stage,
            student_input.strip(),
            history,
            classifier_template=resources.get("stage_classifier_prompt"),
        )
        teaching_decision = build_default_teaching_decision()
        if tutor_route == GUIDED_TUTOR_ROUTE:
            teaching_decision = resolve_teaching_decision(
                effective_stage,
                student_input.strip(),
                history,
                classifier_template=resources.get(
                    "teaching_policy_classifier_prompt"
                ),
                allow_model_fallback=False,
            )
        system_prompt = build_system_prompt(resources)
        knowledge_base = build_rag_augmented_knowledge(
            resources["knowledge_base"],
            student_input.strip(),
            task_type="vision",
        )
        state = prepare_tutor_conversation_state(
            None,
            effective_stage,
            stage,
            history,
            teaching_decision,
        )
        user_prompt = build_tutor_prompt(
            vision_template,
            knowledge_base,
            student_input.strip(),
            {
                **state,
                "route": VISION_TUTOR_ROUTE,
                "turn_count": len(history),
            },
            trimmed_history,
            get_stage_instruction(resources, effective_stage),
            get_teaching_instruction(resources, teaching_decision),
        )
        vision_client = create_vision_client()
    except RuntimeError:
        context = build_vision_log_context(
            TUTOR_MODE,
            stage,
            VISION_TUTOR_ROUTE,
            "CONFIG_ERROR",
            started_at,
            image_count=1,
            image_format=image_format,
            image_bytes_total=len(image_bytes),
        )
        logger.exception(format_log_message("chat_vision config failed", context), extra=context)
        return build_vision_error_payload(
            TUTOR_MODE,
            stage,
            VISION_TUTOR_ROUTE,
            "CONFIG_ERROR",
            "视觉 AI 服务尚未配置，请检查环境变量。",
        )
    except Exception:
        context = build_vision_log_context(
            TUTOR_MODE,
            stage,
            VISION_TUTOR_ROUTE,
            "AGENT_RUNTIME_ERROR",
            started_at,
            image_count=1,
            image_format=image_format,
            image_bytes_total=len(image_bytes),
        )
        logger.exception(format_log_message("chat_vision setup failed", context), extra=context)
        return build_vision_error_payload(
            TUTOR_MODE,
            stage,
            VISION_TUTOR_ROUTE,
            "AGENT_RUNTIME_ERROR",
            "Agent 处理失败，请稍后重试。",
        )

    try:
        markdown = call_vision_model(
            vision_client,
            system_prompt,
            user_prompt,
            [VisionImage(bytes_data=image_bytes, mime=image_format)],
            temperature=0.3,
        )
    except Exception:
        context = build_vision_log_context(
            TUTOR_MODE,
            stage,
            VISION_TUTOR_ROUTE,
            "VISION_API_ERROR",
            started_at,
            image_count=1,
            image_format=image_format,
            image_bytes_total=len(image_bytes),
        )
        logger.exception(format_log_message("chat_vision model failed", context), extra=context)
        return build_vision_error_payload(
            TUTOR_MODE,
            stage,
            VISION_TUTOR_ROUTE,
            "VISION_API_ERROR",
            "视觉模型暂时不可用，请稍后重试。",
        )

    save_output(format_saved_content("视觉问答", student_input.strip(), markdown), "tutor_chat")
    payload = build_frontend_payload(
        TUTOR_MODE,
        markdown,
        {
            "stage": stage,
            "saved": True,
            "warning": history_warning,
            "need_student_reply": True,
        },
    )
    payload["route"] = VISION_TUTOR_ROUTE
    payload["vision_info"] = {
        "image_count": 1,
        "image_format": image_format,
        "image_bytes": len(image_bytes),
    }

    context = build_vision_log_context(
        TUTOR_MODE,
        effective_stage,
        VISION_TUTOR_ROUTE,
        "OK",
        started_at,
        image_count=1,
        image_format=image_format,
        image_bytes_total=len(image_bytes),
        teaching_policy=teaching_decision["policy"],
        guidance_level=teaching_decision["guidance_level"],
    )
    logger.info(format_log_message("chat_vision completed", context), extra=context)
    return payload


def iter_sse_response(events):
    for event in events:
        yield format_sse_event(event["event"], event["data"])


@app.post("/api/agent/chat/stream")
def agent_chat_stream(request: ChatRequest) -> StreamingResponse:
    events = stream_tutor_events(
        student_input=request.student_input,
        stage=request.stage,
        conversation_history=request.conversation_history,
        save=True,
        teaching_policy=request.teaching_policy,
        guidance_level=request.guidance_level,
    )
    return StreamingResponse(
        iter_sse_response(events),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@app.post("/api/agent/report-check")
def agent_report_check(request: ReportCheckRequest) -> dict[str, Any]:
    try:
        return check_report(
            report_content=request.report_content,
            stage=request.stage,
            save=True,
            stream=False,
            quiz_context=request.quiz_context,
        )
    except Exception:
        context = {
            "mode": "report_check",
            "stage": request.stage,
            "route": "unknown",
            "error_code": "AGENT_RUNTIME_ERROR",
        }
        logger.exception(
            format_log_message("agent_report_check request failed", context),
            extra=context,
        )
        return {
            "success": False,
            "error_code": "AGENT_RUNTIME_ERROR",
            "message": "Agent 处理失败，请稍后重试。",
        }


@app.post("/api/agent/report-check/upload")
async def agent_report_check_upload(
    file: UploadFile | None = File(default=None),
    session_id: str | None = Form(default=None),
    stage: str = Form(default="report"),
) -> dict[str, Any]:
    started_at = time.perf_counter()
    page_count = 0
    embedded_image_count = 0

    if file is None or not session_id or not session_id.strip():
        context = build_upload_log_context(
            stage,
            "VALIDATION_ERROR",
            started_at,
            page_count,
            embedded_image_count,
        )
        logger.warning(
            format_log_message("report_check_upload validation failed", context),
            extra=context,
        )
        return build_upload_error_payload(
            stage,
            "VALIDATION_ERROR",
            "请上传 PDF 文件并提供 session_id。",
        )

    if file.content_type != "application/pdf":
        context = build_upload_log_context(
            stage,
            "VALIDATION_ERROR",
            started_at,
            page_count,
            embedded_image_count,
        )
        logger.warning(
            format_log_message("report_check_upload validation failed", context),
            extra=context,
        )
        return build_upload_error_payload(
            stage,
            "VALIDATION_ERROR",
            "仅支持 PDF 文件。",
        )

    pdf_bytes = await file.read()
    if not pdf_bytes:
        context = build_upload_log_context(
            stage,
            "VALIDATION_ERROR",
            started_at,
            page_count,
            embedded_image_count,
        )
        logger.warning(
            format_log_message("report_check_upload validation failed", context),
            extra=context,
        )
        return build_upload_error_payload(
            stage,
            "VALIDATION_ERROR",
            "PDF 文件不能为空。",
        )

    if len(pdf_bytes) > MAX_PDF_SIZE_BYTES:
        context = build_upload_log_context(
            stage,
            "VALIDATION_ERROR",
            started_at,
            page_count,
            embedded_image_count,
        )
        logger.warning(
            format_log_message("report_check_upload validation failed", context),
            extra=context,
        )
        return build_upload_error_payload(
            stage,
            "VALIDATION_ERROR",
            "PDF 文件不能超过 10MB。",
        )

    try:
        extraction = extract_pdf_text(pdf_bytes)
        page_count = extraction.page_count
        embedded_image_count = extraction.embedded_image_count
    except PdfTooManyPagesError:
        context = build_upload_log_context(
            stage,
            "VALIDATION_ERROR",
            started_at,
            page_count,
            embedded_image_count,
        )
        logger.warning(
            format_log_message("report_check_upload validation failed", context),
            extra=context,
        )
        return build_upload_error_payload(
            stage,
            "VALIDATION_ERROR",
            "PDF 页数不能超过 50 页。",
        )
    except PdfParseError:
        context = build_upload_log_context(
            stage,
            "DOCUMENT_PARSE_ERROR",
            started_at,
            page_count,
            embedded_image_count,
        )
        logger.exception(
            format_log_message("report_check_upload parse failed", context),
            extra=context,
        )
        return build_upload_error_payload(
            stage,
            "DOCUMENT_PARSE_ERROR",
            "PDF 解析失败，请检查文件是否损坏。",
        )

    document_info = {
        "page_count": extraction.page_count,
        "extracted_text_chars": len(extraction.text.strip()),
        "embedded_image_count": extraction.embedded_image_count,
        "suspected_scanned": extraction.suspected_scanned,
    }

    if extraction.suspected_scanned:
        context = build_upload_log_context(
            stage,
            "SCANNED_NOT_SUPPORTED",
            started_at,
            page_count,
            embedded_image_count,
        )
        logger.info(
            format_log_message("report_check_upload scanned detected", context),
            extra=context,
        )
        payload = build_upload_error_payload(
            stage,
            "SCANNED_NOT_SUPPORTED",
            "该 PDF 似为扫描件，请改用 /api/agent/report-check/vision 或手动截图上传。",
        )
        payload["document_info"] = document_info
        return payload

    try:
        payload = check_report(
            report_content=extraction.text,
            stage=stage,
            save=True,
            stream=False,
        )
    except Exception:
        context = build_upload_log_context(
            stage,
            "AGENT_RUNTIME_ERROR",
            started_at,
            page_count,
            embedded_image_count,
        )
        logger.exception(
            format_log_message("report_check_upload request failed", context),
            extra=context,
        )
        payload = build_upload_error_payload(
            stage,
            "AGENT_RUNTIME_ERROR",
            "Agent 处理失败，请稍后重试。",
        )

    payload["document_info"] = document_info
    if payload.get("success"):
        payload["warning"] = (
            f"PDF 共 {extraction.page_count} 页，检测到 "
            f"{extraction.embedded_image_count} 张图片，"
            "本版本仅基于文字内容检查；如需图像识别请改用 /api/agent/report-check/vision。"
        )

    context = build_upload_log_context(
        stage,
        str(payload.get("error_code") or "OK"),
        started_at,
        page_count,
        embedded_image_count,
    )
    logger.info(
        format_log_message("report_check_upload completed", context),
        extra=context,
    )
    return payload


def build_report_vision_info(
    images: list[VisionImage],
    image_source: str,
    truncated: bool,
) -> dict[str, Any]:
    return {
        "image_count": len(images),
        "image_source": image_source,
        "image_bytes_total": sum(len(image.bytes_data) for image in images),
        "truncated": truncated,
    }


def build_report_vision_text(
    stage: str,
    report_content: str,
    image_source: str,
    quiz_context: str | None = None,
) -> tuple[str, str]:
    resources, vision_template = build_vision_resources(VISION_REPORT_PROMPT_PATH)
    system_prompt = build_system_prompt(resources)
    knowledge_base = build_rag_augmented_knowledge(
        resources["knowledge_base"],
        build_report_rag_query(report_content, stage),
        task_type="vision",
    )
    user_prompt = build_report_prompt(
        vision_template,
        knowledge_base,
        (
            f"检查阶段：{stage}\n"
            f"图像来源：{image_source}\n\n"
            f"{report_content}".strip()
        ),
        quiz_context=quiz_context,
    )
    return system_prompt, user_prompt


def call_report_vision(
    stage: str,
    report_content: str,
    images: list[VisionImage],
    image_source: str,
) -> str:
    system_prompt, user_prompt = build_report_vision_text(
        stage,
        report_content,
        image_source,
    )
    try:
        vision_client = create_vision_client()
    except RuntimeError as exc:
        raise VisionConfigurationError(str(exc)) from exc
    return call_vision_model(
        vision_client,
        system_prompt,
        user_prompt,
        images,
        temperature=0.2,
    )


def build_report_vision_payload(
    stage: str,
    markdown: str,
    report_content: str,
    images: list[VisionImage],
    image_source: str,
    truncated: bool,
    document_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    save_output(
        format_saved_content("视觉报告检查", report_content, markdown),
        "report_check",
    )
    payload = build_frontend_payload(
        REPORT_CHECK_MODE,
        markdown,
        {
            "stage": stage,
            "saved": True,
            "warning": None,
        },
    )
    payload["route"] = VISION_REPORT_ROUTE
    if document_info is not None:
        payload["document_info"] = document_info
    payload["vision_info"] = build_report_vision_info(
        images,
        image_source,
        truncated,
    )
    return payload



def iter_report_vision_stream_events(
    *,
    stage: str,
    route: str,
    report_content: str,
    system_prompt: str,
    user_prompt: str,
    images: list[VisionImage] | None,
    document_info: dict[str, Any] | None,
    vision_info: dict[str, Any] | None,
    started_at: float,
):
    stream = None
    yield {
        "event": "meta",
        "data": build_vision_stream_meta(
            REPORT_CHECK_VISION_STREAM_MODE,
            route,
            document_info=document_info,
            vision_info=vision_info,
        ),
    }
    try:
        chunks: list[str] = []
        if route == PDF_TEXT_FALLBACK_STREAM_ROUTE:
            model_client = create_client()
            stream = iter_model_stream_chunks(
                model_client,
                system_prompt,
                user_prompt,
                0.2,
            )
        else:
            vision_client = create_vision_client()
            stream = iter_vision_model_stream_chunks(
                vision_client,
                system_prompt,
                user_prompt,
                images or [],
                temperature=0.2,
            )

        for chunk in stream:
            chunks.append(chunk)
            yield {"event": "delta", "data": {"text": chunk}}

        markdown = normalize_model_output("".join(chunks).strip())
        save_output(
            format_saved_content("视觉报告检查", report_content, markdown),
            "report_check",
        )
        context = build_vision_log_context(
            REPORT_CHECK_VISION_STREAM_MODE,
            stage,
            route,
            "OK",
            started_at,
            page_count=(document_info or {}).get("page_count", 0),
            embedded_image_count=(document_info or {}).get("embedded_image_count", 0),
            image_count=len(images or []),
            image_bytes_total=sum(len(image.bytes_data) for image in (images or [])),
        )
        logger.info(format_log_message("report_check_vision_stream completed", context), extra=context)
        yield {"event": "done", "data": build_vision_stream_done(markdown)}
    except (GeneratorExit, asyncio.CancelledError):
        context = build_vision_log_context(
            REPORT_CHECK_VISION_STREAM_MODE,
            stage,
            route,
            "CANCELLED",
            started_at,
            page_count=(document_info or {}).get("page_count", 0),
            embedded_image_count=(document_info or {}).get("embedded_image_count", 0),
            image_count=len(images or []),
            image_bytes_total=sum(len(image.bytes_data) for image in (images or [])),
        )
        logger.info(format_log_message("report_check_vision_stream cancelled", context), extra=context)
        raise
    except Exception:
        context = build_vision_log_context(
            REPORT_CHECK_VISION_STREAM_MODE,
            stage,
            route,
            "VISION_API_ERROR",
            started_at,
            page_count=(document_info or {}).get("page_count", 0),
            embedded_image_count=(document_info or {}).get("embedded_image_count", 0),
            image_count=len(images or []),
            image_bytes_total=sum(len(image.bytes_data) for image in (images or [])),
        )
        logger.exception(format_log_message("report_check_vision_stream failed", context), extra=context)
        yield {
            "event": "error",
            "data": build_vision_stream_error(
                "VISION_API_ERROR",
                "视觉模型暂时不可用，请稍后重试。",
            ),
        }
    finally:
        if stream is not None:
            close_stream_iterator(stream)


@app.post("/api/agent/report-check/vision/stream")
async def agent_report_check_vision_stream(
    file: UploadFile | None = File(default=None),
    session_id: str | None = Form(default=None),
    stage: str = Form(default="report"),
    quiz_context: str | None = Form(default=None, max_length=MAX_QUIZ_CONTEXT_CHARS),
):
    started_at = time.perf_counter()

    if not vision_enabled():
        payload = build_vision_error_payload(
            REPORT_CHECK_MODE,
            stage,
            VISION_REPORT_ROUTE,
            "VISION_DISABLED",
            "视觉能力未启用。",
        )
        context = build_vision_log_context(
            REPORT_CHECK_VISION_STREAM_MODE,
            stage,
            REPORT_IMAGE_VISION_STREAM_ROUTE,
            "VISION_DISABLED",
            started_at,
        )
        logger.warning(format_log_message("report_check_vision_stream disabled", context), extra=context)
        return JSONResponse(status_code=404, content=payload)

    if file is None or not session_id or not session_id.strip():
        return build_vision_error_payload(
            REPORT_CHECK_MODE,
            stage,
            VISION_REPORT_ROUTE,
            "VALIDATION_ERROR",
            "请上传文件并提供 session_id。",
        )

    content_type = file.content_type or ""
    file_bytes = await file.read()
    if not file_bytes:
        return build_vision_error_payload(
            REPORT_CHECK_MODE,
            stage,
            VISION_REPORT_ROUTE,
            "VALIDATION_ERROR",
            "文件不能为空。",
        )

    if content_type in SUPPORTED_IMAGE_MIME_TYPES:
        if len(file_bytes) > MAX_STREAM_VISION_IMAGE_SIZE_BYTES:
            return build_vision_error_payload(
                REPORT_CHECK_MODE,
                stage,
                VISION_REPORT_ROUTE,
                "IMAGE_TOO_LARGE",
                "单张图片不能超过 15MB。",
            )

        images = [VisionImage(bytes_data=file_bytes, mime=content_type)]
        report_content = "学生上传的是报告图片，请根据图片中的可见文字、数据和图像内容进行检查。"
        system_prompt, user_prompt = build_report_vision_text(
            stage,
            report_content,
            "upload",
            quiz_context=quiz_context,
        )
        events = iter_report_vision_stream_events(
            stage=stage,
            route=REPORT_IMAGE_VISION_STREAM_ROUTE,
            report_content="学生上传的是报告图片。",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            images=images,
            document_info=None,
            vision_info=build_report_vision_info(images, "upload", False),
            started_at=started_at,
        )
        return StreamingResponse(
            iter_sse_response(events),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )

    if content_type != "application/pdf":
        return build_vision_error_payload(
            REPORT_CHECK_MODE,
            stage,
            VISION_REPORT_ROUTE,
            "IMAGE_FORMAT_UNSUPPORTED",
            "仅支持 PDF、PNG、JPEG 或 WebP 文件。",
        )

    if len(file_bytes) > MAX_STREAM_PDF_SIZE_BYTES:
        return build_vision_error_payload(
            REPORT_CHECK_MODE,
            stage,
            VISION_REPORT_ROUTE,
            "VALIDATION_ERROR",
            "PDF 文件不能超过 30MB。",
        )

    page_count = 0
    embedded_image_count = 0
    try:
        extraction = extract_pdf_text(file_bytes)
        page_count = extraction.page_count
        embedded_image_count = extraction.embedded_image_count
    except PdfTooManyPagesError:
        return build_vision_error_payload(
            REPORT_CHECK_MODE,
            stage,
            VISION_REPORT_ROUTE,
            "VALIDATION_ERROR",
            "PDF 页数不能超过 50 页。",
        )
    except PdfParseError:
        context = build_vision_log_context(
            REPORT_CHECK_VISION_STREAM_MODE,
            stage,
            PDF_IMAGE_VISION_STREAM_ROUTE,
            "DOCUMENT_PARSE_ERROR",
            started_at,
            page_count,
            embedded_image_count,
        )
        logger.exception(format_log_message("report_check_vision_stream parse failed", context), extra=context)
        return build_vision_error_payload(
            REPORT_CHECK_MODE,
            stage,
            VISION_REPORT_ROUTE,
            "DOCUMENT_PARSE_ERROR",
            "PDF 解析失败，请检查文件是否损坏。",
        )

    document_info = build_document_info(extraction)
    if not extraction.suspected_scanned and extraction.embedded_image_count == 0:
        try:
            resources = load_resources()
            system_prompt = build_system_prompt(resources)
            knowledge_base = build_rag_augmented_knowledge(
                resources["knowledge_base"],
                build_report_rag_query(extraction.text.strip(), stage),
                task_type="vision",
            )
            user_prompt = build_report_prompt(
                resources["report_prompt"],
                knowledge_base,
                extraction.text.strip(),
                quiz_context=quiz_context,
            )
        except Exception:
            context = build_vision_log_context(
                REPORT_CHECK_VISION_STREAM_MODE,
                stage,
                PDF_TEXT_FALLBACK_STREAM_ROUTE,
                "AGENT_RUNTIME_ERROR",
                started_at,
                page_count,
                embedded_image_count,
            )
            logger.exception(format_log_message("report_check_vision_stream setup failed", context), extra=context)
            return build_vision_error_payload(
                REPORT_CHECK_MODE,
                stage,
                VISION_REPORT_ROUTE,
                "AGENT_RUNTIME_ERROR",
                "Agent 处理失败，请稍后重试。",
            )

        events = iter_report_vision_stream_events(
            stage=stage,
            route=PDF_TEXT_FALLBACK_STREAM_ROUTE,
            report_content=extraction.text.strip(),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            images=None,
            document_info=document_info,
            vision_info=None,
            started_at=started_at,
        )
        return StreamingResponse(
            iter_sse_response(events),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )

    try:
        if extraction.suspected_scanned:
            images = rasterize_scanned_pdf(
                file_bytes,
                dpi=150,
                limit=MAX_STREAM_VISION_IMAGES_PER_REQUEST,
            )
            image_source = "pdf_rasterized"
            truncated = len(images) < extraction.page_count
            report_content = "该 PDF 疑似扫描件，请根据页面图像中的可见内容进行报告检查。"
        else:
            images = extract_embedded_images(
                file_bytes,
                limit=MAX_STREAM_VISION_IMAGES_PER_REQUEST,
            )
            image_source = "pdf_embedded"
            truncated = len(images) < extraction.embedded_image_count
            report_content = (
                "PDF 已抽取出文字和嵌入图片。请结合文字内容与图片证据进行报告检查。\n\n"
                f"抽取文字：\n{extraction.text}"
            )
        if not images:
            raise PdfParseError("No vision images extracted from PDF.")
    except PdfParseError:
        context = build_vision_log_context(
            REPORT_CHECK_VISION_STREAM_MODE,
            stage,
            PDF_IMAGE_VISION_STREAM_ROUTE,
            "DOCUMENT_PARSE_ERROR",
            started_at,
            page_count,
            embedded_image_count,
        )
        logger.exception(format_log_message("report_check_vision_stream image extraction failed", context), extra=context)
        return build_vision_error_payload(
            REPORT_CHECK_MODE,
            stage,
            VISION_REPORT_ROUTE,
            "DOCUMENT_PARSE_ERROR",
            "PDF 图像提取失败，请检查文件或改为截图上传。",
        )

    try:
        system_prompt, user_prompt = build_report_vision_text(
            stage,
            report_content,
            image_source,
            quiz_context=quiz_context,
        )
    except Exception:
        context = build_vision_log_context(
            REPORT_CHECK_VISION_STREAM_MODE,
            stage,
            PDF_IMAGE_VISION_STREAM_ROUTE,
            "AGENT_RUNTIME_ERROR",
            started_at,
            page_count,
            embedded_image_count,
            len(images),
            image_bytes_total=sum(len(image.bytes_data) for image in images),
        )
        logger.exception(format_log_message("report_check_vision_stream setup failed", context), extra=context)
        return build_vision_error_payload(
            REPORT_CHECK_MODE,
            stage,
            VISION_REPORT_ROUTE,
            "AGENT_RUNTIME_ERROR",
            "Agent 处理失败，请稍后重试。",
        )

    events = iter_report_vision_stream_events(
        stage=stage,
        route=PDF_IMAGE_VISION_STREAM_ROUTE,
        report_content=report_content,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        images=images,
        document_info=document_info,
        vision_info=build_report_vision_info(images, image_source, truncated),
        started_at=started_at,
    )
    return StreamingResponse(
        iter_sse_response(events),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
@app.post("/api/agent/report-check/vision")
async def agent_report_check_vision(
    file: UploadFile | None = File(default=None),
    session_id: str | None = Form(default=None),
    stage: str = Form(default="report"),
):
    started_at = time.perf_counter()
    page_count = 0
    embedded_image_count = 0

    if not vision_enabled():
        payload = build_vision_error_payload(
            REPORT_CHECK_MODE,
            stage,
            VISION_REPORT_ROUTE,
            "VISION_DISABLED",
            "视觉能力未启用。",
        )
        context = build_vision_log_context(
            REPORT_CHECK_MODE,
            stage,
            VISION_REPORT_ROUTE,
            "VISION_DISABLED",
            started_at,
        )
        logger.warning(format_log_message("report_check_vision disabled", context), extra=context)
        return JSONResponse(status_code=404, content=payload)

    if file is None or not session_id or not session_id.strip():
        context = build_vision_log_context(
            REPORT_CHECK_MODE,
            stage,
            VISION_REPORT_ROUTE,
            "VALIDATION_ERROR",
            started_at,
        )
        logger.warning(format_log_message("report_check_vision validation failed", context), extra=context)
        return build_vision_error_payload(
            REPORT_CHECK_MODE,
            stage,
            VISION_REPORT_ROUTE,
            "VALIDATION_ERROR",
            "请上传文件并提供 session_id。",
        )

    content_type = file.content_type or ""
    file_bytes = await file.read()
    if not file_bytes:
        context = build_vision_log_context(
            REPORT_CHECK_MODE,
            stage,
            VISION_REPORT_ROUTE,
            "VALIDATION_ERROR",
            started_at,
        )
        logger.warning(format_log_message("report_check_vision validation failed", context), extra=context)
        return build_vision_error_payload(
            REPORT_CHECK_MODE,
            stage,
            VISION_REPORT_ROUTE,
            "VALIDATION_ERROR",
            "文件不能为空。",
        )

    if content_type in SUPPORTED_IMAGE_MIME_TYPES:
        if len(file_bytes) > MAX_VISION_IMAGE_SIZE_BYTES:
            context = build_vision_log_context(
                REPORT_CHECK_MODE,
                stage,
                VISION_REPORT_ROUTE,
                "IMAGE_TOO_LARGE",
                started_at,
                image_count=1,
                image_bytes_total=len(file_bytes),
            )
            logger.warning(format_log_message("report_check_vision validation failed", context), extra=context)
            return build_vision_error_payload(
                REPORT_CHECK_MODE,
                stage,
                VISION_REPORT_ROUTE,
                "IMAGE_TOO_LARGE",
                "单张图片不能超过 5MB。",
            )

        images = [VisionImage(bytes_data=file_bytes, mime=content_type)]
        try:
            markdown = call_report_vision(
                stage,
                "学生上传的是报告图片，请根据图片中的可见文字、数据和图像内容进行检查。",
                images,
                "upload",
            )
            payload = build_report_vision_payload(
                stage,
                markdown,
                "学生上传的是报告图片。",
                images,
                "upload",
                False,
            )
        except VisionConfigurationError:
            context = build_vision_log_context(
                REPORT_CHECK_MODE,
                stage,
                VISION_REPORT_ROUTE,
                "CONFIG_ERROR",
                started_at,
                image_count=1,
                image_bytes_total=len(file_bytes),
            )
            logger.exception(format_log_message("report_check_vision config failed", context), extra=context)
            return build_vision_error_payload(
                REPORT_CHECK_MODE,
                stage,
                VISION_REPORT_ROUTE,
                "CONFIG_ERROR",
                "视觉 AI 服务尚未配置，请检查环境变量。",
            )
        except Exception:
            context = build_vision_log_context(
                REPORT_CHECK_MODE,
                stage,
                VISION_REPORT_ROUTE,
                "VISION_API_ERROR",
                started_at,
                image_count=1,
                image_bytes_total=len(file_bytes),
            )
            logger.exception(format_log_message("report_check_vision model failed", context), extra=context)
            return build_vision_error_payload(
                REPORT_CHECK_MODE,
                stage,
                VISION_REPORT_ROUTE,
                "VISION_API_ERROR",
                "视觉模型暂时不可用，请稍后重试。",
            )

        context = build_vision_log_context(
            REPORT_CHECK_MODE,
            stage,
            VISION_REPORT_ROUTE,
            "OK",
            started_at,
            image_count=1,
            image_bytes_total=len(file_bytes),
        )
        logger.info(format_log_message("report_check_vision completed", context), extra=context)
        return payload

    if content_type != "application/pdf":
        context = build_vision_log_context(
            REPORT_CHECK_MODE,
            stage,
            VISION_REPORT_ROUTE,
            "IMAGE_FORMAT_UNSUPPORTED",
            started_at,
        )
        logger.warning(format_log_message("report_check_vision validation failed", context), extra=context)
        return build_vision_error_payload(
            REPORT_CHECK_MODE,
            stage,
            VISION_REPORT_ROUTE,
            "IMAGE_FORMAT_UNSUPPORTED",
            "仅支持 PDF、PNG、JPEG 或 WebP 文件。",
        )

    if len(file_bytes) > MAX_PDF_SIZE_BYTES:
        context = build_vision_log_context(
            REPORT_CHECK_MODE,
            stage,
            VISION_REPORT_ROUTE,
            "VALIDATION_ERROR",
            started_at,
        )
        logger.warning(format_log_message("report_check_vision validation failed", context), extra=context)
        return build_vision_error_payload(
            REPORT_CHECK_MODE,
            stage,
            VISION_REPORT_ROUTE,
            "VALIDATION_ERROR",
            "PDF 文件不能超过 10MB。",
        )

    try:
        extraction = extract_pdf_text(file_bytes)
        page_count = extraction.page_count
        embedded_image_count = extraction.embedded_image_count
    except PdfTooManyPagesError:
        context = build_vision_log_context(
            REPORT_CHECK_MODE,
            stage,
            VISION_REPORT_ROUTE,
            "VALIDATION_ERROR",
            started_at,
            page_count,
            embedded_image_count,
        )
        logger.warning(format_log_message("report_check_vision validation failed", context), extra=context)
        return build_vision_error_payload(
            REPORT_CHECK_MODE,
            stage,
            VISION_REPORT_ROUTE,
            "VALIDATION_ERROR",
            "PDF 页数不能超过 50 页。",
        )
    except PdfParseError:
        context = build_vision_log_context(
            REPORT_CHECK_MODE,
            stage,
            VISION_REPORT_ROUTE,
            "DOCUMENT_PARSE_ERROR",
            started_at,
            page_count,
            embedded_image_count,
        )
        logger.exception(format_log_message("report_check_vision parse failed", context), extra=context)
        return build_vision_error_payload(
            REPORT_CHECK_MODE,
            stage,
            VISION_REPORT_ROUTE,
            "DOCUMENT_PARSE_ERROR",
            "PDF 解析失败，请检查文件是否损坏。",
        )

    document_info = build_document_info(extraction)

    if not extraction.suspected_scanned and extraction.embedded_image_count == 0:
        payload = dict(
            check_report(
                report_content=extraction.text,
                stage=stage,
                save=True,
                stream=False,
            )
        )
        payload["route"] = VISION_REPORT_ROUTE
        payload["document_info"] = document_info
        if payload.get("success"):
            payload["warning"] = (
                f"PDF 共 {extraction.page_count} 页，文字提取 "
                f"{len(extraction.text.strip())} 字符，未触发视觉模型；"
                "如需图像分析请上传带嵌入图的 PDF。"
            )
        context = build_vision_log_context(
            REPORT_CHECK_MODE,
            stage,
            VISION_REPORT_ROUTE,
            str(payload.get("error_code") or "OK"),
            started_at,
            page_count,
            embedded_image_count,
        )
        logger.info(format_log_message("report_check_vision completed", context), extra=context)
        return payload

    try:
        if extraction.suspected_scanned:
            images = rasterize_scanned_pdf(
                file_bytes,
                dpi=150,
                limit=MAX_VISION_IMAGES_PER_REQUEST,
            )
            image_source = "pdf_rasterized"
            truncated = len(images) < extraction.page_count
            report_content = "该 PDF 疑似扫描件，请根据页面图像中的可见内容进行报告检查。"
        else:
            images = extract_embedded_images(
                file_bytes,
                limit=MAX_VISION_IMAGES_PER_REQUEST,
            )
            image_source = "pdf_embedded"
            truncated = len(images) < extraction.embedded_image_count
            report_content = (
                "PDF 已抽取出文字和嵌入图片。请结合文字内容与图片证据进行报告检查。\n\n"
                f"抽取文字：\n{extraction.text}"
            )
        if not images:
            raise PdfParseError("No vision images extracted from PDF.")
    except PdfParseError:
        context = build_vision_log_context(
            REPORT_CHECK_MODE,
            stage,
            VISION_REPORT_ROUTE,
            "DOCUMENT_PARSE_ERROR",
            started_at,
            page_count,
            embedded_image_count,
        )
        logger.exception(format_log_message("report_check_vision image extraction failed", context), extra=context)
        return build_vision_error_payload(
            REPORT_CHECK_MODE,
            stage,
            VISION_REPORT_ROUTE,
            "DOCUMENT_PARSE_ERROR",
            "PDF 图像提取失败，请检查文件或改为截图上传。",
        )

    image_bytes_total = sum(len(image.bytes_data) for image in images)
    try:
        markdown = call_report_vision(stage, report_content, images, image_source)
        payload = build_report_vision_payload(
            stage,
            markdown,
            report_content,
            images,
            image_source,
            truncated,
            document_info,
        )
    except VisionConfigurationError:
        context = build_vision_log_context(
            REPORT_CHECK_MODE,
            stage,
            VISION_REPORT_ROUTE,
            "CONFIG_ERROR",
            started_at,
            page_count,
            embedded_image_count,
            len(images),
            image_bytes_total=image_bytes_total,
        )
        logger.exception(format_log_message("report_check_vision config failed", context), extra=context)
        return build_vision_error_payload(
            REPORT_CHECK_MODE,
            stage,
            VISION_REPORT_ROUTE,
            "CONFIG_ERROR",
            "视觉 AI 服务尚未配置，请检查环境变量。",
        )
    except Exception:
        context = build_vision_log_context(
            REPORT_CHECK_MODE,
            stage,
            VISION_REPORT_ROUTE,
            "VISION_API_ERROR",
            started_at,
            page_count,
            embedded_image_count,
            len(images),
            image_bytes_total=image_bytes_total,
        )
        logger.exception(format_log_message("report_check_vision model failed", context), extra=context)
        return build_vision_error_payload(
            REPORT_CHECK_MODE,
            stage,
            VISION_REPORT_ROUTE,
            "VISION_API_ERROR",
            "视觉模型暂时不可用，请稍后重试。",
        )

    context = build_vision_log_context(
        REPORT_CHECK_MODE,
        stage,
        VISION_REPORT_ROUTE,
        "OK",
        started_at,
        page_count,
        embedded_image_count,
        len(images),
        image_bytes_total=image_bytes_total,
    )
    logger.info(format_log_message("report_check_vision completed", context), extra=context)
    return payload


@app.post("/api/agent/report-check/stream")
def agent_report_check_stream(request: ReportCheckRequest) -> StreamingResponse:
    try:
        events = stream_report_events(
            report_content=request.report_content,
            stage=request.stage,
            save=True,
            quiz_context=request.quiz_context,
        )
        return StreamingResponse(
            iter_sse_response(events),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )
    except Exception:
        context = {
            "mode": "report_check",
            "stage": request.stage,
            "error_code": "AGENT_RUNTIME_ERROR",
        }
        logger.exception(
            format_log_message("agent_report_check_stream request failed", context),
            extra=context,
        )
        events = [
            {
                "event": "error",
                "data": {
                    "success": False,
                    "error_code": "AGENT_RUNTIME_ERROR",
                    "message": "Agent 处理失败，请稍后重试。",
                },
            }
        ]
        return StreamingResponse(
            iter_sse_response(events),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )
