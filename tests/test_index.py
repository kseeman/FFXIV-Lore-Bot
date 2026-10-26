import json

import pytest

from conftest import fake_embed

from xivlore.rag.index import Index, build_index


@pytest.fixture
def index_dir(dist_dir, tmp_path):
    out = tmp_path / "index"
    build_index(dist_dir, out, embed=fake_embed)
    return out


def test_build_writes_meta(index_dir):
    meta = json.loads((index_dir / "meta.json").read_text())
    assert meta["chunks"] == 3
    assert meta["built_at"] == "2026-10-11T15:50:59+00:00"


def test_build_reports_progress(dist_dir, tmp_path):
    calls = []
    build_index(dist_dir, tmp_path / "index", embed=fake_embed,
                progress=lambda done, total: calls.append((done, total)))
    assert calls[-1] == (3, 3)


def test_search_ranks_by_similarity(index_dir):
    results = Index(index_dir).search("how much gil?", k=2, embed=fake_embed)
    assert [c.title for c, _ in results][0] == "Gil"
    assert len(results) == 2
    assert results[0][1] > results[1][1]


def test_stale_when_dist_rebuilt(index_dir, dist_dir):
    index = Index(index_dir)
    assert not index.is_stale(dist_dir)
    (dist_dir / "manifest.json").write_text(json.dumps({"built_at": "later"}))
    assert index.is_stale(dist_dir)


def test_missing_index(tmp_path):
    with pytest.raises(FileNotFoundError, match="xivlore index"):
        Index(tmp_path)
