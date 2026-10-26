"""Split the built corpus into passages small enough to embed and retrieve.

Transcripts run from a few lines to a whole quest chain, so they are cut into
overlapping windows on line boundaries. Every window repeats the transcript's
title and header line, so a passage from mid-scene still says where it is from.
Descriptions are short and each becomes one passage.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

MAX_CHARS = 1500
OVERLAP_CHARS = 300


@dataclass(frozen=True)
class Chunk:
    id: str
    source_id: str
    category: str
    title: str
    text: str


def chunk_corpus(dist_dir: Path) -> list[Chunk]:
    """Every passage from the ``transcripts`` and ``descriptions`` configs."""
    chunks = []
    for row in pq.read_table(dist_dir / "transcripts").to_pylist():
        chunks.extend(transcript_chunks(row))
    for row in pq.read_table(dist_dir / "descriptions").to_pylist():
        chunks.append(description_chunk(row))
    return chunks


def transcript_chunks(row: dict) -> Iterator[Chunk]:
    head, _, body = row["text"].partition("\n\n")
    for n, window in enumerate(windows(body.split("\n"), MAX_CHARS - len(head) - 2)):
        yield Chunk(
            id=f"{row['id']}#{n}",
            source_id=row["id"],
            category=row["category"],
            title=row["title"],
            text=f"{head}\n\n{window}",
        )


def description_chunk(row: dict) -> Chunk:
    return Chunk(
        id=row["id"],
        source_id=row["id"],
        category=row["category"],
        title=row["name"],
        text=f"{row['name']} ({row['category'].replace('_', ' ')})\n{row['text']}",
    )


def windows(lines: list[str], max_chars: int) -> Iterator[str]:
    """Pack ``lines`` into windows of at most ``max_chars``, overlapping slightly.

    A single line longer than ``max_chars`` becomes a window of its own rather
    than being cut mid-sentence.
    """
    lines = [l for l in lines if l.strip()]
    start = 0
    while start < len(lines):
        end, size = start, 0
        while end < len(lines) and (end == start or size + len(lines[end]) + 1 <= max_chars):
            size += len(lines[end]) + 1
            end += 1
        yield "\n".join(lines[start:end])
        if end == len(lines):
            return
        # Step back over trailing lines for overlap, but always move forward.
        back, carried = end, 0
        while back - 1 > start and carried + len(lines[back - 1]) + 1 <= OVERLAP_CHARS:
            back -= 1
            carried += len(lines[back]) + 1
        # Carrying lines the next one cannot join would repeat them on their own.
        if carried + len(lines[end]) + 1 > max_chars:
            back = end
        start = back
