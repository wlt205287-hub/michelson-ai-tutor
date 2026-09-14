from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

from pypdf import PdfReader

from vision_client import VisionImage


MAX_PDF_SIZE_BYTES = 10 * 1024 * 1024
MAX_PDF_PAGES = 50
MIN_EXTRACTED_TEXT_CHARS = 200
MAX_VISION_IMAGES_PER_REQUEST = 8


@dataclass
class PdfExtractionResult:
    text: str
    page_count: int
    embedded_image_count: int
    suspected_scanned: bool


class PdfParseError(Exception):
    """Raised when pypdf cannot parse the supplied PDF bytes."""


class PdfTooManyPagesError(Exception):
    """Raised when a PDF exceeds the supported page limit."""


try:
    from pdf2image import convert_from_bytes as _pdf2image_convert_from_bytes
except ImportError:
    _pdf2image_convert_from_bytes = None


def convert_from_bytes(pdf_bytes: bytes, dpi: int = 150):
    if _pdf2image_convert_from_bytes is None:
        raise PdfParseError("pdf2image is not installed.")
    return _pdf2image_convert_from_bytes(pdf_bytes, dpi=dpi)


def extract_pdf_text(pdf_bytes: bytes) -> PdfExtractionResult:
    """Extract text and lightweight document metadata from PDF bytes."""
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        page_count = len(reader.pages)
        if page_count > MAX_PDF_PAGES:
            raise PdfTooManyPagesError(f"PDF page count exceeds {MAX_PDF_PAGES}.")

        text_parts: list[str] = []
        embedded_image_count = 0
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                text_parts.append(page_text.strip())
            embedded_image_count += len(getattr(page, "images", []))

        text = "\n\n".join(text_parts)
        return PdfExtractionResult(
            text=text,
            page_count=page_count,
            embedded_image_count=embedded_image_count,
            suspected_scanned=len(text.strip()) < MIN_EXTRACTED_TEXT_CHARS,
        )
    except PdfTooManyPagesError:
        raise
    except Exception as exc:
        raise PdfParseError("Failed to parse PDF.") from exc


def _detect_image_mime(data: bytes, name: str | None = None) -> str:
    normalized_name = (name or "").lower()
    if normalized_name.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    if normalized_name.endswith(".webp"):
        return "image/webp"
    if normalized_name.endswith(".png"):
        return "image/png"
    if data.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def extract_embedded_images(
    pdf_bytes: bytes,
    limit: int = MAX_VISION_IMAGES_PER_REQUEST,
) -> list[VisionImage]:
    """Extract embedded PDF images as GLM-compatible image payloads."""
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        page_count = len(reader.pages)
        if page_count > MAX_PDF_PAGES:
            raise PdfTooManyPagesError(f"PDF page count exceeds {MAX_PDF_PAGES}.")

        images: list[VisionImage] = []
        for page in reader.pages:
            for image_file in getattr(page, "images", []):
                data = getattr(image_file, "data", b"") or b""
                if not data:
                    continue
                images.append(
                    VisionImage(
                        bytes_data=data,
                        mime=_detect_image_mime(
                            data,
                            getattr(image_file, "name", None),
                        ),
                    )
                )
                if len(images) >= limit:
                    return images
        return images
    except PdfTooManyPagesError:
        raise
    except Exception as exc:
        raise PdfParseError("Failed to extract PDF images.") from exc


def rasterize_scanned_pdf(
    pdf_bytes: bytes,
    dpi: int = 150,
    limit: int = MAX_VISION_IMAGES_PER_REQUEST,
) -> list[VisionImage]:
    """Rasterize scanned PDF pages to PNG images for vision models."""
    try:
        rendered_pages = convert_from_bytes(pdf_bytes, dpi=dpi)
        images: list[VisionImage] = []
        for page in rendered_pages[:limit]:
            buffer = BytesIO()
            page.save(buffer, format="PNG")
            images.append(VisionImage(bytes_data=buffer.getvalue(), mime="image/png"))
        return images
    except Exception as exc:
        raise PdfParseError("Failed to rasterize PDF.") from exc
