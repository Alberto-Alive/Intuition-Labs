"""Document chunking for dense / hybrid retrieval."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..schema import Document


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str


def chunk_documents(docs: List[Document], chunk_size: int = 240,
                    overlap: int = 40) -> List[Chunk]:
    chunks: List[Chunk] = []
    for d in docs:
        text = d.text
        # prepend title so chunk carries doc context
        body = f"[{d.title}] {text}"
        if len(body) <= chunk_size:
            chunks.append(Chunk(f"{d.doc_id}#0", d.doc_id, body))
            continue
        start, k = 0, 0
        while start < len(body):
            piece = body[start:start + chunk_size]
            chunks.append(Chunk(f"{d.doc_id}#{k}", d.doc_id, piece))
            k += 1
            start += chunk_size - overlap
    return chunks
