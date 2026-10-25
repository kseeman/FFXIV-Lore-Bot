"""Assemble extracted records into the Hugging Face dataset layout.

``dist/`` gets one directory of Parquet per config, a ``manifest.json``
describing the source snapshot, and the dataset card (``README.md``).
"""

import json
from collections.abc import Iterable
from dataclasses import asdict
from datetime import UTC, datetime
from itertools import groupby
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from xivlore import card
from xivlore.sources.datamining import descriptions, dialogue
from xivlore.sources.datamining.snapshot import Snapshot

DIALOGUE_SCHEMA = pa.schema([
    ("id", pa.string()),
    ("category", pa.string()),
    ("script", pa.string()),
    ("index", pa.int32()),
    ("key", pa.string()),
    ("kind", pa.string()),
    ("speaker_key", pa.string()),
    ("speaker", pa.string()),
    ("display_speaker", pa.string()),
    ("text", pa.string()),
    ("quest_id", pa.int32()),
    ("quest_name", pa.string()),
    ("expansion", pa.string()),
    ("journal_genre", pa.string()),
    ("journal_category", pa.string()),
    ("journal_section", pa.string()),
    ("place", pa.string()),
    ("level", pa.int16()),
])

TRANSCRIPT_SCHEMA = pa.schema([
    ("id", pa.string()),
    ("category", pa.string()),
    ("script", pa.string()),
    ("title", pa.string()),
    ("quest_id", pa.int32()),
    ("expansion", pa.string()),
    ("journal_genre", pa.string()),
    ("journal_category", pa.string()),
    ("journal_section", pa.string()),
    ("place", pa.string()),
    ("level", pa.int16()),
    ("speakers", pa.list_(pa.string())),
    ("n_lines", pa.int32()),
    ("text", pa.string()),
])

DESCRIPTION_SCHEMA = pa.schema([
    ("id", pa.string()),
    ("category", pa.string()),
    ("sheet", pa.string()),
    ("row_id", pa.int32()),
    ("name", pa.string()),
    ("field", pa.string()),
    ("text", pa.string()),
])

# Line kinds that read as part of a scene. Objectives and system prompts are
# UI chatter; they stay in the dialogue config but not in transcripts.
TRANSCRIPT_KINDS = {"journal", "dialogue", "choice_prompt", "choice"}


def transcripts(lines: Iterable[dialogue.Line]) -> list[dict]:
    """Collapse each script's lines into one readable document.

    ``lines`` must be grouped by script, as ``dialogue.extract`` yields them.
    """
    documents = []
    for _, group in groupby(lines, key=lambda l: (l.category, l.script)):
        group = [l for l in group if l.kind in TRANSCRIPT_KINDS]
        spoken = [l for l in group if l.kind == "dialogue"]
        if not spoken:
            continue
        first = group[0]
        title = first.quest_name or first.script
        header = " · ".join(
            v for v in (first.journal_section, first.journal_genre, first.expansion,
                        first.place) if v
        )
        journal = [l.text for l in group if l.kind == "journal"]
        body = [_render(l) for l in group if l.kind != "journal"]
        sections = [title + (f"\n{header}" if header else ""), *journal, "\n".join(body)]
        documents.append({
            "id": f"{first.category}/{first.script}",
            "category": first.category,
            "script": first.script,
            "title": title,
            "quest_id": first.quest_id,
            "expansion": first.expansion,
            "journal_genre": first.journal_genre,
            "journal_category": first.journal_category,
            "journal_section": first.journal_section,
            "place": first.place,
            "level": first.level,
            "speakers": list(dict.fromkeys(l.speaker for l in spoken)),
            "n_lines": len(spoken),
            "text": "\n\n".join(sections),
        })
    return documents


def _render(line: dialogue.Line) -> str:
    if line.kind == "dialogue":
        return f"{line.speaker}: {line.text}"
    if line.kind == "choice":
        return f"> {line.text}"
    return line.text


def build(snapshot: Snapshot, out_dir: Path) -> dict:
    """Extract everything from ``snapshot`` and write the dataset to ``out_dir``."""
    lines = list(dialogue.extract(snapshot.csv_dir))
    configs = {
        "dialogue": (DIALOGUE_SCHEMA, [asdict(l) for l in lines]),
        "transcripts": (TRANSCRIPT_SCHEMA, transcripts(lines)),
        "descriptions": (
            DESCRIPTION_SCHEMA,
            [asdict(d) for d in descriptions.extract(snapshot.csv_dir)],
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    for name, (schema, rows) in configs.items():
        config_dir = out_dir / name
        config_dir.mkdir(exist_ok=True)
        for stale in config_dir.glob("*.parquet"):
            stale.unlink()
        pq.write_table(
            pa.Table.from_pylist(rows, schema=schema),
            config_dir / "train-00000-of-00001.parquet",
            compression="zstd",
        )
        stats[name] = {"rows": len(rows), "characters": sum(len(r["text"]) for r in rows)}

    manifest = {
        "source": {
            "repository": "https://github.com/xivapi/ffxiv-datamining",
            "commit": snapshot.commit,
            "commit_date": snapshot.commit_date,
            "game_version": snapshot.game_version,
        },
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "configs": stats,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out_dir / "README.md").write_text(card.render(manifest))
    return manifest
