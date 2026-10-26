import json

from conftest import fake_embed

from xivlore.rag.evaluate import Question, load_questions, report, run
from xivlore.rag.index import Index, build_index


def test_load_questions(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text(
        json.dumps({"id": "a", "question": "Q?", "answer": "A.", "must_mention": ["x"]})
        + "\n\n"
        + json.dumps({"id": "b", "question": "Q?", "answer": "A.",
                      "must_mention": [["y", "z"]], "must_not_mention": ["w"],
                      "source_title": "Gil"})
        + "\n"
    )
    a, b = load_questions(path)
    assert a.must_mention == (("x",),) and a.must_not_mention == ()
    assert a.source_titles == ()
    assert b.must_mention == (("y", "z"),) and b.must_not_mention == ("w",)
    assert b.source_titles == ("Gil",)


def test_passes_needs_every_group_ignoring_case():
    q = Question("q", "Q?", "A.", (("exarch",), ("crystal",)))
    assert q.passes("The Crystal Exarch leads it.")
    assert not q.passes("The Exarch leads it.")


def test_passes_accepts_any_alternative():
    q = Question("q", "Q?", "A.", (("Dawnservant", "throne"),))
    assert q.passes("She seeks her father's throne.")
    assert q.passes("She would be Dawnservant.")
    assert not q.passes("She seeks a crown.")


def test_passes_rejects_forbidden_words():
    q = Question("q", "Q?", "A.", (("Scion",),), must_not_mention=("Lalafell",))
    assert q.passes("A Miqo'te Scion.")
    assert not q.passes("A Lalafell Scion.")


def test_passes_treats_curly_apostrophe_as_straight():
    assert Question("q", "Q?", "A.", (("G'raha",),)).passes("He is G\u2019raha Tia.")


def test_run_scores_both_modes(dist_dir, tmp_path):
    build_index(dist_dir, tmp_path / "index", embed=fake_embed)

    def chat(messages):
        # Only the grounded prompt carries excerpts, so only it "knows" the answer.
        return "It is currency." if "Excerpts:" in messages[-1]["content"] else "No idea."

    questions = [
        Question("gil", "What is gil?", "Currency.", (("currency",),), source_titles=("Gil",)),
        Question("moogle", "What says kupo?", "A moogle.", (("moogle",),),
                 source_titles=("Nope",)),
    ]
    results = run(questions, Index(tmp_path / "index"), k=1, chat=chat, embed=fake_embed)
    assert [(r.plain_pass, r.grounded_pass, r.retrieved_source) for r in results] == [
        (False, True, True),
        (False, False, False),
    ]
    text = report(results)
    assert "no retrieval 0/2, with retrieval 1/2, source retrieved 1/2" in text
    assert "## gil: What is gil?" in text


def test_run_answers_at_temperature_zero(dist_dir, tmp_path, monkeypatch):
    build_index(dist_dir, tmp_path / "index", embed=fake_embed)
    temperatures = []
    monkeypatch.setattr("xivlore.rag.ollama.chat",
                        lambda messages, **opts: temperatures.append(opts) or "")
    run([Question("q", "Q?", "A.", ())], Index(tmp_path / "index"), embed=fake_embed)
    assert temperatures == [{"temperature": 0}, {"temperature": 0}]
