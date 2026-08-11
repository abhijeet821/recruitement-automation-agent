"""Unstructured input -> structured records."""

from matching.parsing.contacts import extract_contacts, find_github_username
from matching.parsing.jobspec import parse_jobspec
from matching.parsing.pdf import PDFExtraction, extract_pdf_text
from matching.parsing.resume import parse_resume

__all__ = [
    "extract_contacts",
    "find_github_username",
    "parse_jobspec",
    "PDFExtraction",
    "extract_pdf_text",
    "parse_resume",
]
