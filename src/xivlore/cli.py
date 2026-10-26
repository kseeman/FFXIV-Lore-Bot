"""xivlore: fetch game text, build the lore corpus, publish it, and ask it questions."""

import argparse
import sys
from pathlib import Path

from xivlore.sources.datamining import snapshot

DEFAULT_CACHE = Path(".cache/ffxiv-datamining")
DEFAULT_OUT = Path("dist")
DEFAULT_INDEX = Path(".cache/index")
DEFAULT_QUESTIONS = Path("eval/questions.jsonl")
DEFAULT_REPORTS = Path(".cache/eval")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="xivlore", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    fetch = commands.add_parser("fetch", help="download or update the ffxiv-datamining CSVs")
    fetch.add_argument("--ref", default="master", help="branch, tag, or full commit SHA")
    fetch.add_argument("--cache", type=Path, default=DEFAULT_CACHE)

    build = commands.add_parser("build", help="extract the corpus into Parquet configs")
    build.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    build.add_argument("--out", type=Path, default=DEFAULT_OUT)

    push = commands.add_parser("push", help="upload a build to a private HF dataset")
    push.add_argument("repo_id", help="e.g. your-username/ffxiv-lore")
    push.add_argument("--out", type=Path, default=DEFAULT_OUT)

    index = commands.add_parser("index", help="embed a build for retrieval with Ollama")
    index.add_argument("--out", type=Path, default=DEFAULT_OUT)
    index.add_argument("--index", type=Path, default=DEFAULT_INDEX)

    ask = commands.add_parser("ask", help="answer a lore question with the local model")
    ask.add_argument("question")
    ask.add_argument("-k", type=int, default=8, help="passages to retrieve")
    ask.add_argument("--no-rag", action="store_true", help="ask the model without retrieval")
    ask.add_argument("--show-context", action="store_true", help="print retrieved passages")
    ask.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ask.add_argument("--index", type=Path, default=DEFAULT_INDEX)

    evaluate = commands.add_parser("eval", help="score answers with and without retrieval")
    evaluate.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    evaluate.add_argument("-k", type=int, default=8, help="passages to retrieve")
    evaluate.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    evaluate.add_argument("--reports", type=Path, default=DEFAULT_REPORTS)

    args = parser.parse_args(argv)
    if args.command == "fetch":
        try:
            snap = snapshot.fetch(args.cache, args.ref)
        except RuntimeError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        print(f"ffxiv-datamining {snap.commit[:8]} ({snap.commit_date}), "
              f"game {snap.game_version or 'unknown'}")
    elif args.command == "build":
        from xivlore.build import build as run_build

        manifest = run_build(snapshot.open_snapshot(args.cache), args.out)
        for name, stats in manifest["configs"].items():
            print(f"{name:>13}: {stats['rows']:>9,} rows {stats['characters']:>12,} chars")
        print(f"wrote {args.out}/")
    elif args.command == "push":
        from xivlore.hub import NotPrivateError, push as run_push

        try:
            print(run_push(args.out, args.repo_id))
        except NotPrivateError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
    elif args.command == "index":
        from xivlore.rag.index import build_index
        from xivlore.rag.ollama import OllamaError

        def progress(done: int, total: int) -> None:
            print(f"\rembedded {done:,}/{total:,}", end="", file=sys.stderr, flush=True)

        try:
            meta = build_index(args.out, args.index, progress=progress)
        except OllamaError as e:
            print(f"\nerror: {e}", file=sys.stderr)
            return 1
        print(f"\nindexed {meta['chunks']:,} passages into {args.index}/")
    elif args.command == "ask":
        from xivlore.rag.answer import ask as run_ask
        from xivlore.rag.index import Index
        from xivlore.rag.ollama import OllamaError

        index = None
        if not args.no_rag:
            try:
                index = Index(args.index)
            except FileNotFoundError as e:
                print(f"error: {e}", file=sys.stderr)
                return 1
            if index.is_stale(args.out):
                print(f"warning: {args.out}/ was rebuilt since indexing; run `xivlore index`",
                      file=sys.stderr)
        try:
            answer = run_ask(args.question, index, k=args.k)
        except OllamaError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        if args.show_context:
            for n, (chunk, score) in enumerate(answer.passages, 1):
                print(f"--- [{n}] {chunk.id} ({score:.3f})\n{chunk.text}\n")
        print(answer.text)
        if answer.passages:
            print("\nSources:")
            for n, (chunk, _) in enumerate(answer.passages, 1):
                print(f"  [{n}] {chunk.title} ({chunk.source_id})")
    elif args.command == "eval":
        from datetime import datetime

        from xivlore.rag import evaluate as ev
        from xivlore.rag.index import Index
        from xivlore.rag.ollama import OllamaError

        try:
            index = Index(args.index)
            results = ev.run(ev.load_questions(args.questions), index, k=args.k)
        except (FileNotFoundError, OllamaError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        for r in results:
            print(f"{r.question.id:<30} no retrieval {'pass' if r.plain_pass else 'FAIL'}"
                  f"  with retrieval {'pass' if r.grounded_pass else 'FAIL'}")
        print(ev.summary(results))
        # Reports quote game text, so they stay under .cache/ with the index.
        args.reports.mkdir(parents=True, exist_ok=True)
        path = args.reports / f"{datetime.now():%Y%m%d-%H%M%S}.md"
        path.write_text(ev.report(results))
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
