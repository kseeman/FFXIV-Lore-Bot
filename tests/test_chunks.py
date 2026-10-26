import pyarrow as pa
import pyarrow.parquet as pq

from xivlore.build import DESCRIPTION_SCHEMA, TRANSCRIPT_SCHEMA
from xivlore.rag import chunks
from xivlore.rag.chunks import chunk_corpus, transcript_chunks, windows


def test_windows_fit_and_overlap(monkeypatch):
    monkeypatch.setattr(chunks, "OVERLAP_CHARS", 10)
    lines = [f"line {i}" for i in range(10)]  # 6 chars each, 7 with newline
    out = list(windows(lines, 24))
    assert out[0] == "line 0\nline 1\nline 2"
    assert out[1].startswith("line 2\n")  # last line carried over
    assert all(len(w) <= 24 for w in out)
    assert out[-1].endswith("line 9")


def test_windows_keep_long_line_whole():
    assert list(windows(["x" * 50, "short"], 20)) == ["x" * 50, "short"]


def test_windows_skip_blank_lines():
    assert list(windows(["a", "", "  ", "b"], 100)) == ["a\nb"]


def test_transcript_chunks_repeat_header(monkeypatch):
    monkeypatch.setattr(chunks, "MAX_CHARS", 60)
    row = {
        "id": "quest/X", "category": "quest", "title": "Way of the Archer",
        "text": "Way of the Archer\nGridania\n\n"
                + "\n".join(f"Luciane: Line number {i}." for i in range(4)),
    }
    out = list(transcript_chunks(row))
    assert len(out) > 1
    assert [c.id for c in out] == [f"quest/X#{n}" for n in range(len(out))]
    assert all(c.text.startswith("Way of the Archer\nGridania\n\n") for c in out)
    assert all(c.source_id == "quest/X" and c.title == "Way of the Archer" for c in out)


def test_chunk_corpus_reads_both_configs(tmp_path):
    transcript = dict.fromkeys(TRANSCRIPT_SCHEMA.names) | {
        "id": "quest/X", "category": "quest", "title": "T", "text": "T\n\nA: hi",
    }
    description = {
        "id": "TripleTriadCard/1/Description", "category": "triple_triad_card",
        "sheet": "TripleTriadCard",
        "row_id": 1, "name": "Moogle", "field": "Description", "text": "Kupo.",
    }
    for name, schema, row in (("transcripts", TRANSCRIPT_SCHEMA, transcript),
                              ("descriptions", DESCRIPTION_SCHEMA, description)):
        (tmp_path / name).mkdir()
        pq.write_table(pa.Table.from_pylist([row], schema=schema),
                       tmp_path / name / "train.parquet")
    out = chunk_corpus(tmp_path)
    assert [c.text for c in out] == ["T\n\nA: hi", "Moogle (triple triad card)\nKupo."]
