from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from pypdf import PdfReader, PdfWriter
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
    NumberObject,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from document_loader import (
    MAX_PDF_PAGES,
    PdfParseError,
    PdfTooManyPagesError,
    extract_pdf_text,
)


def build_text_pdf(text: str, page_count: int = 1) -> bytes:
    writer = PdfWriter()
    for _ in range(page_count):
        page = writer.add_blank_page(width=612, height=792)
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {
                NameObject("/Font"): DictionaryObject(
                    {NameObject("/F1"): font}
                )
            }
        )
        stream = DecodedStreamObject()
        escaped_text = (
            text.replace("\\", "\\\\")
            .replace("(", "\\(")
            .replace(")", "\\)")
        )
        stream.set_data(
            f"BT /F1 12 Tf 72 720 Td ({escaped_text}) Tj ET".encode("latin-1")
        )
        page[NameObject("/Contents")] = writer._add_object(stream)

    buffer = BytesIO()
    writer.write(buffer)
    result = buffer.getvalue()
    assert PdfReader(BytesIO(result)).pages, "build_text_pdf produced unreadable PDF"
    return result


def build_pdf_with_image() -> bytes:
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)

    image = DecodedStreamObject()
    image.set_data(bytes([255, 0, 0]))
    image.update(
        {
            NameObject("/Type"): NameObject("/XObject"),
            NameObject("/Subtype"): NameObject("/Image"),
            NameObject("/Width"): NumberObject(1),
            NameObject("/Height"): NumberObject(1),
            NameObject("/ColorSpace"): NameObject("/DeviceRGB"),
            NameObject("/BitsPerComponent"): NumberObject(8),
        }
    )
    image_ref = writer._add_object(image)

    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): font}),
            NameObject("/XObject"): DictionaryObject({NameObject("/Im1"): image_ref}),
        }
    )
    stream = DecodedStreamObject()
    text = "michelson report " * 20
    stream.set_data(
        (
            f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET "
            "q 10 0 0 10 72 680 cm /Im1 Do Q"
        ).encode("latin-1")
    )
    page[NameObject("/Contents")] = writer._add_object(stream)

    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


class DocumentLoaderTests(TestCase):
    def test_extract_simple_text_pdf(self) -> None:
        pdf_bytes = build_text_pdf("michelson interferometer report " * 12)

        result = extract_pdf_text(pdf_bytes)

        self.assertEqual(result.page_count, 1)
        self.assertGreaterEqual(len(result.text.strip()), 200)
        self.assertEqual(result.embedded_image_count, 0)
        self.assertFalse(result.suspected_scanned)

    def test_extract_too_few_chars_marked_scanned(self) -> None:
        pdf_bytes = build_text_pdf("short report")

        result = extract_pdf_text(pdf_bytes)

        self.assertEqual(result.page_count, 1)
        self.assertLess(len(result.text.strip()), 200)
        self.assertTrue(result.suspected_scanned)

    def test_extract_counts_embedded_images(self) -> None:
        pdf_bytes = build_pdf_with_image()

        result = extract_pdf_text(pdf_bytes)

        self.assertGreaterEqual(result.embedded_image_count, 1)

    def test_extract_corrupt_pdf_raises_parse_error(self) -> None:
        with patch("pypdf._reader.logger_warning"):
            with self.assertRaises(PdfParseError):
                extract_pdf_text(b"not a pdf")

    def test_extract_too_many_pages_raises(self) -> None:
        pdf_bytes = build_text_pdf(
            "michelson interferometer report " * 12,
            page_count=MAX_PDF_PAGES + 1,
        )

        with self.assertRaises(PdfTooManyPagesError):
            extract_pdf_text(pdf_bytes)
