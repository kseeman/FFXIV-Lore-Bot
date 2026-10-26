"""Score the model on lore questions with known answers, with and without retrieval.

Questions live in a JSON Lines file, one object per line:

    {"id": "...", "question": "...", "answer": "...",
     "must_mention": ["..."], "source_title": "..."}

An answer passes if it mentions every ``must_mention`` keyword (ignoring case).
Where ``source_title`` names the quest or item the answer comes from, the run
also checks whether retrieval found it. Keyword matching is crude, so the full
answers are written to a report for reading through.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from xivlore.rag.answer import Answer, ask
from xivlore.rag.index import Index


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    answer: str
    must_mention: tuple[str, ...]
    source_title: str | None = None


@dataclass
class Result:
    question: Question
    plain: Answer
    grounded: Answer

    @property
    def plain_pass(self) -> bool:
        return mentions_all(self.plain.text, self.question.must_mention)

    @property
    def grounded_pass(self) -> bool:
        return mentions_all(self.grounded.text, self.question.must_mention)

    @property
    def retrieved_source(self) -> bool | None:
        if not self.question.source_title:
            return None
        wanted = self.question.source_title.lower()
        return any(chunk.title.lower() == wanted for chunk, _ in self.grounded.passages)


def load_questions(path: Path) -> list[Question]:
    questions = []
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            questions.append(Question(
                id=row["id"],
                question=row["question"],
                answer=row["answer"],
                must_mention=tuple(row["must_mention"]),
                source_title=row.get("source_title"),
            ))
    return questions


def mentions_all(text: str, keywords: tuple[str, ...]) -> bool:
    return all(k.lower() in text.lower() for k in keywords)


def run(questions: list[Question], index: Index, k: int = 8, **ollama) -> list[Result]:
    return [
        Result(q, ask(q.question, None, **ollama), ask(q.question, index, k, **ollama))
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
            f"**Expected:** {q.answer} (must mention: {', '.join(q.must_mention)})",
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


def summary(results: list[Result]) -> str:
    n = len(results)
    plain = sum(r.plain_pass for r in results)
    grounded = sum(r.grounded_pass for r in results)
    sourced = [r.retrieved_source for r in results if r.retrieved_source is not None]
    text = f"no retrieval {plain}/{n}, with retrieval {grounded}/{n}"
    if sourced:
        text += f", source retrieved {sum(sourced)}/{len(sourced)}"
    return text
