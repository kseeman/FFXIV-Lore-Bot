from conftest import fake_embed

from xivlore.rag.answer import GROUNDED_SYSTEM, PLAIN_SYSTEM, ask
from xivlore.rag.index import Index, build_index


class FakeChat:
    def __init__(self):
        self.messages = None

    def __call__(self, messages):
        self.messages = messages
        return "An answer."


def test_ask_without_index_sends_question_only():
    chat = FakeChat()
    answer = ask("Who is Y'shtola?", None, chat=chat)
    assert answer.text == "An answer."
    assert answer.passages == []
    assert chat.messages == [
        {"role": "system", "content": PLAIN_SYSTEM},
        {"role": "user", "content": "Who is Y'shtola?"},
    ]


def test_ask_with_index_numbers_excerpts(dist_dir, tmp_path):
    build_index(dist_dir, tmp_path / "index", embed=fake_embed)
    chat = FakeChat()
    answer = ask("What is gil?", Index(tmp_path / "index"), k=2, chat=chat, embed=fake_embed)
    assert [c.title for c, _ in answer.passages][0] == "Gil"
    system, user = chat.messages
    assert system["content"] == GROUNDED_SYSTEM
    assert user["content"].startswith(
        "Excerpts:\n\n[1] Gil (item)\nStandard Eorzean currency.\n\n[2] "
    )
    assert user["content"].endswith("\n\nQuestion: What is gil?")
