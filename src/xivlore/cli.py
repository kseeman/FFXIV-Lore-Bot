"""xivlore: fetch game text, build the lore corpus, push it to Hugging Face."""

import argparse
import sys
from pathlib import Path

from xivlore.sources.datamining import snapshot

DEFAULT_CACHE = Path(".cache/ffxiv-datamining")
DEFAULT_OUT = Path("dist")


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
    return 0


if __name__ == "__main__":
    sys.exit(main())
