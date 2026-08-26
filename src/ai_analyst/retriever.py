"""Lightweight RAG layer over RetailSync AI project documentation.

Uses simple keyword-based retrieval (BM25-like) to find relevant documentation
chunks for user questions. No external vector database required.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
DOCS_DIR = os.path.join(_project_root, "docs")


@dataclass
class DocChunk:
    """A chunk of documentation content."""
    source: str
    content: str
    score: float = 0.0


def _load_docs() -> list[DocChunk]:
    """Load and chunk all markdown documentation files."""
    chunks: list[DocChunk] = []
    if not os.path.isdir(DOCS_DIR):
        return chunks
    for filename in os.listdir(DOCS_DIR):
        if not filename.endswith(".md"):
            continue
        filepath = os.path.join(DOCS_DIR, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            for para in paragraphs:
                if len(para) < 40:
                    continue
                chunks.append(DocChunk(source=filename, content=para))
        except Exception as exc:
            logger.warning("Failed to load doc %s: %s", filename, exc)
    return chunks


def _tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9_]+", text.lower())
    stop = {"the", "and", "for", "with", "this", "that", "from", "are", "was", "were",
            "has", "have", "had", "not", "but", "what", "when", "where", "which", "who",
            "how", "why", "all", "any", "each", "every", "both", "few", "more", "most",
            "other", "some", "such", "than", "too", "very", "can", "will", "just", "into",
            "also", "only", "over", "such", "into", "through"}
    return {w for w in words if w not in stop and len(w) > 2}


def _score_chunk(query: str, chunk: DocChunk, top_k: int) -> float:
    query_tokens = _tokenize(query)
    chunk_tokens = _tokenize(chunk.content)
    if not query_tokens:
        return 0.0
    overlap = query_tokens & chunk_tokens
    score = len(overlap) / len(query_tokens)
    source_bonus = 0.1 if chunk.source in ("data_dictionary.md", "inventory_methodology.md", "forecasting_methodology.md") else 0.0
    return score + source_bonus


def retrieve(query: str, top_k: int = 3, max_chunk_chars: int = 800) -> list[DocChunk]:
    """Retrieve the most relevant documentation chunks for a query."""
    chunks = _load_docs()
    if not chunks:
        return []
    scored = []
    for chunk in chunks:
        score = _score_chunk(query, chunk, top_k)
        if score > 0:
            content = chunk.content[:max_chunk_chars]
            if len(chunk.content) > max_chunk_chars:
                content += "..."
            scored.append(DocChunk(source=chunk.source, content=content, score=score))
    scored.sort(key=lambda c: c.score, reverse=True)
    return scored[:top_k]


def format_context(chunks: list[DocChunk]) -> str:
    """Format retrieved chunks as context for the LLM."""
    if not chunks:
        return ""
    lines = ["Relevant project documentation:"]
    for chunk in chunks:
        lines.append(f"\n[Source: {chunk.source}]\n{chunk.content}")
    return "\n".join(lines)
