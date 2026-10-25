import json

import pyarrow.parquet as pq
import pytest
from conftest import write_csv

from xivlore import hub
from xivlore.build import build
from xivlore.sources.datamining.descriptions import SPECS
from xivlore.sources.datamining.snapshot import Snapshot


@pytest.fixture
def built(csv_dir, tmp_path):
    # Empty stand-ins for description sheets the shared fixture does not define.
    columns: dict[str, set[str]] = {}
    for spec in SPECS:
        columns.setdefault(spec.text_sheet, set()).update(spec.text_columns)
        if spec.name_reference:
            columns[spec.text_sheet].add(spec.name_reference)
        columns.setdefault(spec.name_sheet, set()).add(spec.name_column)
    for sheet, names in columns.items():
        if not (csv_dir / f"{sheet}.csv").exists():
            write_csv(csv_dir / f"{sheet}.csv", [["#", *names], [0, *[""] * len(names)]])
    snap = Snapshot(csv_dir.parent.parent, "d9582a624eaec69d", "2026-09-30", "7.56h1")
    out = tmp_path / "dist"
    return out, build(snap, out)


def test_build_writes_configs_and_manifest(built):
    out, manifest = built
    assert json.loads((out / "manifest.json").read_text()) == manifest
    assert manifest["source"]["game_version"] == "7.56h1"
    assert {k: v["rows"] for k, v in manifest["configs"].items()} == {
        "dialogue": 10, "transcripts": 3, "descriptions": 4,
    }
    dialogue = pq.read_table(out / "dialogue").to_pylist()
    assert dialogue[0]["quest_name"] == "Way of the Archer"


def test_quest_transcript(built):
    out, _ = built
    quest = pq.read_table(out / "transcripts").to_pylist()[0]
    assert quest["id"] == "quest/ClsArc000_00021"
    assert quest["speakers"] == ["Luciane"]
    assert quest["n_lines"] == 2
    assert quest["text"] == (
        "Way of the Archer\n"
        "Class & Job Quests · Archer Quests · A Realm Reborn · Gridania\n\n"
        "Athelyna wishes you to join the guild.\n\n"
        "Luciane: Greetings, adventurer.\n"
        "Will you dedicate yourself to archery?\n"
        "> Yes\n"
        "> No\n"
        "Luciane: <sigh> Very well."
    )


def test_card_metadata(built):
    out, _ = built
    readme = (out / "README.md").read_text()
    assert readme.startswith("---\n")
    assert "config_name: transcripts\n  data_files: transcripts/*.parquet\n  default: true" in readme
    assert "[`d9582a62`]" in readme
    assert "$" not in readme


class FakeApi:
    def __init__(self, private):
        self.private, self.uploaded = private, False

    def create_repo(self, *args, **kwargs):
        pass

    def repo_info(self, *args, **kwargs):
        return type("Info", (), {"private": self.private})()

    def upload_folder(self, **kwargs):
        self.uploaded = True
        return type("Commit", (), {"commit_url": "https://hf.co/commit"})()


def test_push_refuses_public_repo(built, monkeypatch):
    out, _ = built
    api = FakeApi(private=False)
    monkeypatch.setattr(hub, "HfApi", lambda: api)
    with pytest.raises(hub.NotPrivateError):
        hub.push(out, "someone/ffxiv-lore")
    assert not api.uploaded


def test_push_private_repo_names_it_in_card(built, monkeypatch):
    out, _ = built
    api = FakeApi(private=True)
    monkeypatch.setattr(hub, "HfApi", lambda: api)
    assert hub.push(out, "someone/ffxiv-lore") == "https://hf.co/commit"
    assert api.uploaded
    assert 'load_dataset("someone/ffxiv-lore", "transcripts"' in (out / "README.md").read_text()
