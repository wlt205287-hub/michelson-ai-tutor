from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from PIL import Image
from pypdf import PdfWriter
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
    NumberObject,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from document_loader import extract_embedded_images, rasterize_scanned_pdf
from vision_client import VisionImage


def build_pdf_with_images(image_count: int) -> bytes:
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    xobjects = DictionaryObject()
    draw_commands: list[str] = []

    for index in range(image_count):
        image = DecodedStreamObject()
        image.set_data(bytes([index % 256, 0, 0]))
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
        name = NameObject(f"/Im{index + 1}")
        xobjects[name] = writer._add_object(image)
        draw_commands.append(f"q 10 0 0 10 72 {700 - index} cm {name} Do Q")

    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/XObject"): xobjects}
    )
    stream = DecodedStreamObject()
    stream.set_data(" ".join(draw_commands).encode("latin-1"))
    page[NameObject("/Contents")] = writer._add_object(stream)

    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


class DocumentLoaderVisionTests(TestCase):
    def test_extract_embedded_images_respects_limit(self) -> None:
        pdf_bytes = build_pdf_with_images(10)

        images = extract_embedded_images(pdf_bytes, limit=3)

        self.assertEqual(len(images), 3)
        self.assertTrue(all(isinstance(image, VisionImage) for image in images))

    def test_extract_embedded_images_detects_mime(self) -> None:
        pdf_bytes = build_pdf_with_images(1)

        images = extract_embedded_images(pdf_bytes)

        self.assertEqual(images[0].mime, "image/png")
        self.assertTrue(images[0].bytes_data.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_rasterize_scanned_pdf_calls_pdf2image_with_dpi(self) -> None:
        rendered_pages = [Image.new("RGB", (1, 1), "white") for _ in range(10)]

        with patch("document_loader.convert_from_bytes", return_value=rendered_pages) as convert:
            images = rasterize_scanned_pdf(b"%PDF-1.4", dpi=150, limit=8)

        convert.assert_called_once_with(b"%PDF-1.4", dpi=150)
        self.assertEqual(len(images), 8)
        self.assertEqual(images[0].mime, "image/png")
        self.assertTrue(images[0].bytes_data.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_rasterize_scanned_pdf_respects_limit(self) -> None:
        rendered_pages = [Image.new("RGB", (1, 1), "white") for _ in range(3)]

        with patch("document_loader.convert_from_bytes", return_value=rendered_pages):
            images = rasterize_scanned_pdf(b"%PDF-1.4", limit=2)

        self.assertEqual(len(images), 2)
