"""Embed every passage once and search them by cosine similarity.

The index is a matrix of unit vectors (``vectors.npy``), the passages in the
same order (``chunks.parquet``), and ``meta.json`` naming the build and the
embedding model it came from. At this size a brute-force dot product is fast
enough that no vector database is needed.

The index holds game text, so it lives under ``.cache/``, outside both git and
the ``dist/`` folder that ``push`` uploads.
"""

import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from xivlore.rag import ollama
from xivlore.rag.chunks import Chunk, chunk_corpus

BATCH_SIZE = 64

Embed = Callable[[list[str]], list[list[float]]]


# embeddinggemma was trained with these task prefixes on queries and documents.
def document_prompt(chunk: Chunk) -> str:
    return f"title: {chunk.title} | text: {chunk.text}"


def query_prompt(question: str) -> str:
    return f"task: search result | query: {question}"


def build_index(
    dist_dir: Path,
    index_dir: Path,
    embed: Embed = ollama.embed,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Chunk and embed the build in ``dist_dir`` and write the index to ``index_dir``."""
    built_at = _build_stamp(dist_dir)
    chunks = chunk_corpus(dist_dir)
    vectors = []
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        vectors.extend(embed([document_prompt(c) for c in batch]))
        if progress:
            progress(start + len(batch), len(chunks))

    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(index_dir / "vectors.npy", _normalise(np.asarray(vectors, dtype=np.float32)))
    pq.write_table(pa.Table.from_pylist([asdict(c) for c in chunks]),
                   index_dir / "chunks.parquet", compression="zstd")
    meta = {
        "built_at": built_at,
        "embed_model": ollama.EMBED_MODEL,
        "chunks": len(chunks),
    }
    (index_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


class Index:
    def __init__(self, index_dir: Path):
        if not (index_dir / "meta.json").exists():
            raise FileNotFoundError(f"no index in {index_dir}; run `xivlore index` first")
        self.meta = json.loads((index_dir / "meta.json").read_text())
        self.vectors = np.load(index_dir / "vectors.npy")
        rows = pq.read_table(index_dir / "chunks.parquet").to_pylist()
        self.chunks = [Chunk(**row) for row in rows]

    def is_stale(self, dist_dir: Path) -> bool:
        """True if ``dist_dir`` or the embedding model changed since indexing."""
        return (self.meta["built_at"] != _build_stamp(dist_dir)
                or self.meta["embed_model"] != ollama.EMBED_MODEL)

    def search(
        self, question: str, k: int = 8, embed: Embed = ollama.embed_and_unload
    ) -> list[tuple[Chunk, float]]:
        query = _normalise(np.asarray(embed([query_prompt(question)]), dtype=np.float32))[0]
        scores = self.vectors @ query
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top]


def _normalise(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def _build_stamp(dist_dir: Path) -> str:
    return json.loads((dist_dir / "manifest.json").read_text())["built_at"]
