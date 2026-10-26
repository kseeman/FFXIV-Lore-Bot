"""Search passages by meaning (embeddings), by keyword, or by both.

The index is a matrix of unit vectors (``vectors.npy``), the passages in the
same order (``chunks.parquet``), a SQLite full-text index over the same
passages (``keywords.sqlite``), and ``meta.json`` naming the build and the
embedding model it came from. At this size a brute-force dot product is fast
enough that no vector database is needed.

Embeddings find passages that say what the question asks in other words;
keywords find exact names that embeddings blur. Hybrid search merges the two
rankings by reciprocal rank fusion, so a passage near the top of either list
ranks high without comparing cosine and BM25 scores directly.

The index holds game text, so it lives under ``.cache/``, outside both git and
the ``dist/`` folder that ``push`` uploads.
"""

import json
import re
import sqlite3
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from xivlore.rag import DEFAULT_SEARCH, SEARCH_MODES, ollama
from xivlore.rag.chunks import Chunk, chunk_corpus

BATCH_SIZE = 64
# Candidates taken from each ranking before fusing, and the usual RRF constant.
FUSION_DEPTH = 50
RRF_K = 60

# Keep apostrophes and hyphens inside tokens: Y'shtola, Kan-E-Senna.
KEYWORD_TOKENIZER = "unicode61 tokenchars '''-'"
STOPWORDS = frozenset("""
    a an and are as at be by did do does for from had has have he her his how i in
    is it its me my of on or she that the their them they this to was were what
    when where which who whom whose why will with you your
""".split())

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
    build_keyword_index(index_dir, chunks)
    meta = {
        "built_at": built_at,
        "embed_model": ollama.EMBED_MODEL,
        "chunks": len(chunks),
    }
    (index_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def build_keyword_index(index_dir: Path, chunks: list[Chunk]) -> None:
    """Write the full-text index; each passage's rowid is its position in ``chunks``."""
    path = index_dir / "keywords.sqlite"
    partial = path.with_suffix(".partial")
    partial.unlink(missing_ok=True)
    db = sqlite3.connect(partial)
    with db:
        db.execute(
            f'CREATE VIRTUAL TABLE passages USING fts5(text, tokenize="{KEYWORD_TOKENIZER}")'
        )
        db.executemany("INSERT INTO passages (rowid, text) VALUES (?, ?)",
                       ((i, c.text) for i, c in enumerate(chunks)))
    db.close()
    partial.replace(path)


def keyword_query(question: str) -> str | None:
    """An FTS5 query matching any of the question's words, each as a prefix.

    A prefix match lets "Y'shtola" find "Y'shtola's". None if no words are left.
    """
    words = []
    for word in re.findall(r"[\w'-]+", question.lower().replace("’", "'")):
        word = word.strip("'-").removesuffix("'s")
        if word and word not in STOPWORDS and word not in words:
            words.append(word)
    return " OR ".join(f'"{w}"*' for w in words) or None


class Index:
    def __init__(self, index_dir: Path):
        if not (index_dir / "meta.json").exists():
            raise FileNotFoundError(f"no index in {index_dir}; run `xivlore index` first")
        self.meta = json.loads((index_dir / "meta.json").read_text())
        self.vectors = np.load(index_dir / "vectors.npy")
        rows = pq.read_table(index_dir / "chunks.parquet").to_pylist()
        self.chunks = [Chunk(**row) for row in rows]
        # Indexes built before keyword search get one here; it takes seconds.
        if not (index_dir / "keywords.sqlite").exists():
            build_keyword_index(index_dir, self.chunks)
        self.keywords = sqlite3.connect(index_dir / "keywords.sqlite")

    def is_stale(self, dist_dir: Path) -> bool:
        """True if ``dist_dir`` or the embedding model changed since indexing."""
        return (self.meta["built_at"] != _build_stamp(dist_dir)
                or self.meta["embed_model"] != ollama.EMBED_MODEL)

    def search(
        self,
        question: str,
        k: int = 8,
        embed: Embed = ollama.embed_and_unload,
        mode: str = DEFAULT_SEARCH,
    ) -> list[tuple[Chunk, float]]:
        """The top ``k`` passages with their scores, higher meaning closer.

        What a score means depends on ``mode``: cosine similarity, negated
        BM25, or fused reciprocal rank.
        """
        if mode == "vector":
            ranked = self._vector_ranking(question, k, embed)
        elif mode == "keyword":
            ranked = self._keyword_ranking(question, k)
        elif mode == "hybrid":
            fused: dict[int, float] = {}
            for ranking in (self._vector_ranking(question, FUSION_DEPTH, embed),
                            self._keyword_ranking(question, FUSION_DEPTH)):
                for rank, (i, _) in enumerate(ranking, 1):
                    fused[i] = fused.get(i, 0.0) + 1 / (RRF_K + rank)
            ranked = sorted(fused.items(), key=lambda item: -item[1])[:k]
        else:
            raise ValueError(f"unknown search mode {mode!r}; expected one of {SEARCH_MODES}")
        return [(self.chunks[i], score) for i, score in ranked]

    def _vector_ranking(self, question: str, n: int, embed: Embed) -> list[tuple[int, float]]:
        query = _normalise(np.asarray(embed([query_prompt(question)]), dtype=np.float32))[0]
        scores = self.vectors @ query
        return [(int(i), float(scores[i])) for i in np.argsort(-scores)[:n]]

    def _keyword_ranking(self, question: str, n: int) -> list[tuple[int, float]]:
        query = keyword_query(question)
        if query is None:
            return []
        rows = self.keywords.execute(
            "SELECT rowid, bm25(passages) FROM passages WHERE passages MATCH ?"
            " ORDER BY bm25(passages) LIMIT ?",
            (query, n),
        )
        return [(i, -score) for i, score in rows]


def _normalise(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def _build_stamp(dist_dir: Path) -> str:
    return json.loads((dist_dir / "manifest.json").read_text())["built_at"]
