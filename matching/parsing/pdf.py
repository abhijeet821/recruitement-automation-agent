"""
PDF text extraction, with an honest quality signal.

The predecessor of this module silently returned ``""`` on any failure, so a
scanned resume and an empty upload were indistinguishable from a genuinely weak
candidate — all three scored zero. Here extraction always reports *why* it
produced what it produced, and the pipeline propagates that as low confidence
instead of a low score.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

logger = logging.getLogger(__name__)

MAX_PAGES = 15
# Below this, a PDF almost certainly holds page images rather than a text layer.
MIN_CHARS_FOR_TEXT_LAYER = 200
# Ligature and mojibake repairs common in PDF text layers.
_FIXES = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", " ": " ", "•": "\n- ",
}
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")


@dataclass
class PDFExtraction:
    text: str = ""
    page_count: int = 0
    pages_read: int = 0
    ok: bool = False
    # "ok" | "encrypted" | "no_text_layer" | "corrupt" | "missing" | "empty"
    reason: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def looks_scanned(self) -> bool:
        return self.reason == "no_text_layer"


def clean_pdf_text(text: str) -> str:
    for bad, good in _FIXES.items():
        text = text.replace(bad, good)
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def extract_pdf_text(path: str | Path, max_pages: int = MAX_PAGES) -> PDFExtraction:
    """Extract text from a PDF, never raising — failures come back as a reason."""
    path = Path(path)

    if not path.exists():
        return PDFExtraction(ok=False, reason="missing", warnings=[f"No file at {path}"])
    if path.stat().st_size == 0:
        return PDFExtraction(ok=False, reason="empty", warnings=["File is zero bytes"])

    try:
        reader = PdfReader(str(path))
    except (PdfReadError, OSError, ValueError) as exc:
        logger.warning("Unreadable PDF %s: %s", path.name, exc)
        return PDFExtraction(ok=False, reason="corrupt", warnings=[str(exc)])

    warnings: list[str] = []

    if reader.is_encrypted:
        # Many resumes are "encrypted" with an empty owner password, which
        # pypdf can open; only give up if that fails.
        try:
            reader.decrypt("")
        except Exception:  # noqa: BLE001
            return PDFExtraction(
                ok=False, reason="encrypted", warnings=["PDF is password protected"]
            )

    try:
        total_pages = len(reader.pages)
    except Exception as exc:  # noqa: BLE001
        return PDFExtraction(ok=False, reason="corrupt", warnings=[str(exc)])

    if total_pages > max_pages:
        warnings.append(f"Resume has {total_pages} pages; only the first {max_pages} were read")

    chunks: list[str] = []
    pages_read = 0
    for index, page in enumerate(reader.pages[:max_pages]):
        try:
            chunks.append(page.extract_text() or "")
            pages_read += 1
        except Exception as exc:  # noqa: BLE001 - one bad page must not lose the rest
            warnings.append(f"Page {index + 1} could not be read: {exc}")

    text = clean_pdf_text("\n\n".join(c for c in chunks if c.strip()))

    if len(text) < MIN_CHARS_FOR_TEXT_LAYER:
        warnings.append(
            "Almost no extractable text — the resume is probably a scan or an image. "
            "OCR would be required to read it."
        )
        return PDFExtraction(
            text=text, page_count=total_pages, pages_read=pages_read,
            ok=False, reason="no_text_layer", warnings=warnings,
        )

    return PDFExtraction(
        text=text, page_count=total_pages, pages_read=pages_read,
        ok=True, reason="ok", warnings=warnings,
    )
