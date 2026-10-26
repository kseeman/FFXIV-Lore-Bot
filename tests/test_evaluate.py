import json

from conftest import fake_embed

from xivlore.rag.evaluate import Question, load_questions, mentions_all, report, run
from xivlore.rag.index import Index, build_index


def test_load_questions(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text(
        json.dumps({"id": "a", "question": "Q?", "answer": "A.", "must_mention": ["x"]})
        + "\n\n"
        + json.dumps({"id": "b", "question": "Q?", "answer": "A.", "must_mention": [],
                      "source_title": "Gil"})
        + "\n"
    )
    a, b = load_questions(path)
    assert a.must_mention == ("x",) and a.source_title is None
    assert b.source_title == "Gil"


def test_mentions_all_ignores_case():
    assert mentions_all("The Crystal Exarch leads it.", ("exarch", "crystal"))
    assert not mentions_all("A council leads it.", ("Exarch",))


def test_run_scores_both_modes(dist_dir, tmp_path):
    build_index(dist_dir, tmp_path / "index", embed=fake_embed)

    def chat(messages):
        # Only the grounded prompt carries excerpts, so only it "knows" the answer.
        return "It is currency." if "Excerpts:" in messages[-1]["content"] else "No idea."

    questions = [
        Question("gil", "What is gil?", "Currency.", ("currency",), source_title="Gil"),
        Question("moogle", "What says kupo?", "A moogle.", ("moogle",), source_title="Nope"),
    ]
    results = run(questions, Index(tmp_path / "index"), k=1, chat=chat, embed=fake_embed)
    assert [(r.plain_pass, r.grounded_pass, r.retrieved_source) for r in results] == [
        (False, True, True),
        (False, False, False),
    ]
    text = report(results)
    assert "no retrieval 0/2, with retrieval 1/2, source retrieved 1/2" in text
    assert "## gil: What is gil?" in text
