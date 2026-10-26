import json

import pytest

from conftest import fake_embed

from xivlore.rag.index import Index, build_index, keyword_query


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


def test_stale_when_embed_model_changes(index_dir, dist_dir, monkeypatch):
    monkeypatch.setattr("xivlore.rag.ollama.EMBED_MODEL", "another-embedder")
    assert Index(index_dir).is_stale(dist_dir)


def test_missing_index(tmp_path):
    with pytest.raises(FileNotFoundError, match="xivlore index"):
        Index(tmp_path)


def test_keyword_query_drops_stopwords_and_possessives():
    assert keyword_query("Who is Y\u2019shtola's sister?") == '"y\'shtola"* OR "sister"*'
    assert keyword_query("Who leads the Order of the Twin Adder?") == (
        '"leads"* OR "order"* OR "twin"* OR "adder"*'
    )
    assert keyword_query("Who is it?") is None


def test_keyword_search_finds_names(index_dir):
    results = Index(index_dir).search("Tell me about Gil", k=1, mode="keyword")
    assert [c.title for c, _ in results] == ["Gil"]


def test_keyword_search_without_words_finds_nothing(index_dir):
    assert Index(index_dir).search("Who is it?", mode="keyword") == []


def test_hybrid_fuses_both_rankings(index_dir, monkeypatch):
    index = Index(index_dir)
    monkeypatch.setattr(index, "_vector_ranking", lambda q, n, embed: [(0, 0.9), (1, 0.8)])
    monkeypatch.setattr(index, "_keyword_ranking", lambda q, n: [(1, 5.0), (2, 4.0)])
    results = index.search("q", k=3, mode="hybrid")
    # Passage 1 ranks in both lists, so it beats passage 0, which tops only one.
    assert [index.chunks.index(c) for c, _ in results] == [1, 0, 2]
    assert results[0][1] == pytest.approx(1 / 62 + 1 / 61)


def test_unknown_search_mode(index_dir):
    with pytest.raises(ValueError, match="unknown search mode"):
        Index(index_dir).search("q", mode="fuzzy", embed=fake_embed)


def test_index_without_keywords_gets_one(index_dir):
    (index_dir / "keywords.sqlite").unlink()
    results = Index(index_dir).search("gil", k=1, mode="keyword")
    assert [c.title for c, _ in results] == ["Gil"]
