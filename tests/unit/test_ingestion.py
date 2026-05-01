from __future__ import annotations

from PIL import Image
import pytest

from ingestion.exceptions import InvalidFileError, PDFLoadingError
from ingestion.file_validator import validate_file
from ingestion.pdf_loader import load_document_images


def test_validate_file_accepts_png(sample_image_path):
    validated = validate_file(sample_image_path)
    assert validated == sample_image_path.resolve()


def test_validate_file_accepts_pdf(sample_pdf_path, monkeypatch):
    monkeypatch.setattr("ingestion.file_validator.pdfinfo_from_path", lambda path: {"Pages": 1})
    validated = validate_file(sample_pdf_path)
    assert validated == sample_pdf_path.resolve()


def test_validate_file_rejects_invalid_file_type(tmp_path):
    invalid_path = tmp_path / "document.txt"
    invalid_path.write_text("not supported", encoding="utf-8")

    with pytest.raises(InvalidFileError, match="Unsupported file extension"):
        validate_file(invalid_path)


def test_validate_file_rejects_corrupted_file(corrupted_file_path):
    with pytest.raises(InvalidFileError, match="Image validation failed"):
        validate_file(corrupted_file_path)


def test_load_document_images_returns_single_rgb_image(sample_image_path):
    images = load_document_images(sample_image_path)

    assert len(images) == 1
    assert images[0].mode == "RGB"
    assert images[0].size == (400, 200)


def test_load_document_images_converts_pdf_pages(sample_pdf_path, monkeypatch):
    monkeypatch.setattr(
        "ingestion.pdf_loader.convert_from_path",
        lambda *args, **kwargs: [
            Image.new("RGB", (100, 120), color="white"),
            Image.new("RGB", (100, 120), color="white"),
        ],
    )

    images = load_document_images(sample_pdf_path)

    assert len(images) == 2
    assert all(image.mode == "RGB" for image in images)


def test_load_document_images_raises_pdf_loading_error(sample_pdf_path, monkeypatch):
    def _raise(*args, **kwargs):
        raise RuntimeError("bad pdf")

    def _raise_fitz(*args, **kwargs):
        raise RuntimeError("fitz bad pdf")

    monkeypatch.setattr("ingestion.pdf_loader.convert_from_path", _raise)
    monkeypatch.setattr("ingestion.pdf_loader.fitz.open", _raise_fitz)

    with pytest.raises(PDFLoadingError, match="Unable to convert PDF to images"):
        load_document_images(sample_pdf_path)
