"""Score the model on lore questions with known answers, with and without retrieval.

Questions live in a JSON Lines file, one object per line:

    {"id": "...", "question": "...", "answer": "...",
     "must_mention": ["...", ["...", "..."]], "must_not_mention": ["..."],
     "source_title": "... or [...]"}

An answer passes if it mentions every ``must_mention`` entry and none of the
optional ``must_not_mention`` words, ignoring case. An entry that is a list
passes on any one of its alternatives. Where ``source_title`` names the quest
or item the answer comes from (or a list of them), the run also checks whether
retrieval found it.
Keyword matching is crude, so the full answers are written to a report for
reading through.

The model answers at temperature 0, so a change in score comes from a change
in the code or questions rather than chance.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from xivlore.rag import ollama
from xivlore.rag.answer import Answer, ask
from xivlore.rag.index import Index


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    answer: str
    # Each entry is a group of alternatives; every group must be matched.
    must_mention: tuple[tuple[str, ...], ...]
    must_not_mention: tuple[str, ...] = ()
    source_titles: tuple[str, ...] = ()

    def passes(self, text: str) -> bool:
        # Models often write the curly apostrophe; game names use the straight one.
        text = text.lower().replace("\u2019", "'")
        return (all(any(k.lower() in text for k in group) for group in self.must_mention)
                and not any(k.lower() in text for k in self.must_not_mention))


@dataclass
class Result:
    question: Question
    plain: Answer
    grounded: Answer

    @property
    def plain_pass(self) -> bool:
        return self.question.passes(self.plain.text)

    @property
    def grounded_pass(self) -> bool:
        return self.question.passes(self.grounded.text)

    @property
    def retrieved_source(self) -> bool | None:
        if not self.question.source_titles:
            return None
        wanted = {t.lower() for t in self.question.source_titles}
        return any(chunk.title.lower() in wanted for chunk, _ in self.grounded.passages)


def load_questions(path: Path) -> list[Question]:
    questions = []
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            questions.append(Question(
                id=row["id"],
                question=row["question"],
                answer=row["answer"],
                must_mention=tuple(_strings(m) for m in row["must_mention"]),
                must_not_mention=tuple(row.get("must_not_mention", ())),
                source_titles=_strings(row.get("source_title", ())),
            ))
    return questions


def _strings(value: str | list[str]) -> tuple[str, ...]:
    return (value,) if isinstance(value, str) else tuple(value)


def greedy_chat(messages: list[dict]) -> str:
    return ollama.chat(messages, temperature=0)


def run(
    questions: list[Question], index: Index, k: int = 8, chat=greedy_chat, **kwargs
) -> list[Result]:
    return [
        Result(q, ask(q.question, None, chat=chat, **kwargs),
               ask(q.question, index, k, chat=chat, **kwargs))
        for q in questions
    ]


def report(results: list[Result]) -> str:
    """A Markdown report with each question's expected and actual answers."""
    def mark(ok: bool | None) -> str:
        return {True: "pass", False: "FAIL", None: "n/a"}[ok]

    lines = ["# Lore eval", "", summary(results), ""]
    for r in results:
        q = r.question
        lines += [
            f"## {q.id}: {q.question}",
            "",
            f"**Expected:** {q.answer} ({criteria(q)})",
            "",
            f"### No retrieval: {mark(r.plain_pass)}",
            "",
            r.plain.text,
            "",
            f"### With retrieval: {mark(r.grounded_pass)}"
            f" (source retrieved: {mark(r.retrieved_source)})",
            "",
            r.grounded.text,
            "",
            "Passages:",
            *[f"{n}. {c.title} (`{c.id}`, {s:.3f})"
              for n, (c, s) in enumerate(r.grounded.passages, 1)],
            "",
        ]
    return "\n".join(lines)


def criteria(q: Question) -> str:
    text = "must mention: " + ", ".join(" or ".join(group) for group in q.must_mention)
    if q.must_not_mention:
        text += "; must not mention: " + ", ".join(q.must_not_mention)
    return text


def summary(results: list[Result]) -> str:
    n = len(results)
    plain = sum(r.plain_pass for r in results)
    grounded = sum(r.grounded_pass for r in results)
    sourced = [r.retrieved_source for r in results if r.retrieved_source is not None]
    text = f"no retrieval {plain}/{n}, with retrieval {grounded}/{n}"
    if sourced:
        text += f", source retrieved {sum(sourced)}/{len(sourced)}"
    return text
