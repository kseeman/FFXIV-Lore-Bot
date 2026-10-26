"""Answer a lore question, optionally grounded in passages retrieved from the index."""

from dataclasses import dataclass, field

from xivlore.rag import ollama
from xivlore.rag.chunks import Chunk
from xivlore.rag.index import Embed, Index

GROUNDED_SYSTEM = """\
You answer questions about the lore of Final Fantasy XIV using only the numbered \
excerpts of game text you are given. Cite the excerpts you rely on by number, like [2]. \
If the excerpts do not contain the answer, say that you could not find it in the game \
text instead of guessing."""

PLAIN_SYSTEM = "You answer questions about the lore of Final Fantasy XIV."


@dataclass
class Answer:
    text: str
    # Retrieved passages with their similarity scores, numbered from 1 in the prompt.
    passages: list[tuple[Chunk, float]] = field(default_factory=list)


def ask(
    question: str,
    index: Index | None,
    k: int = 8,
    chat=ollama.chat,
    embed: Embed = ollama.embed_and_unload,
) -> Answer:
    """Answer ``question`` from the top ``k`` passages, or from the model alone if no index."""
    if index is None:
        return Answer(chat([
            {"role": "system", "content": PLAIN_SYSTEM},
            {"role": "user", "content": question},
        ]))
    passages = index.search(question, k, embed=embed)
    excerpts = "\n\n".join(f"[{n}] {chunk.text}" for n, (chunk, _) in enumerate(passages, 1))
    text = chat([
        {"role": "system", "content": GROUNDED_SYSTEM},
        {"role": "user", "content": f"Excerpts:\n\n{excerpts}\n\nQuestion: {question}"},
    ])
    return Answer(text, passages)
