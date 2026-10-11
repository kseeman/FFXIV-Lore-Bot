"""Snapshot of the game's text sheets from xivapi/ffxiv-datamining.

The repository mirrors the client's Excel sheets as CSV, one directory per
language. We sparse-checkout only the English sheets: ~220 MB instead of the
full multi-language repository.
"""

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

REPO_URL = "https://github.com/xivapi/ffxiv-datamining.git"
LANGUAGE = "en"

# Enough history to find the most recent global patch commit; regional client
# updates ("TC 7.25", "KR 7.2") are interleaved with global ones ("7.56h1").
HISTORY_DEPTH = 30
GLOBAL_PATCH = re.compile(r"^(\d+\.\d+\w*)\b")


@dataclass(frozen=True)
class Snapshot:
    root: Path
    commit: str
    commit_date: str
    game_version: str | None

    @property
    def csv_dir(self) -> Path:
        return self.root / "csv" / LANGUAGE


def _git(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def fetch(cache_dir: Path, ref: str = "master") -> Snapshot:
    """Clone or update the sparse checkout in ``cache_dir`` and pin it to ``ref``."""
    if not (cache_dir / ".git").exists():
        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        _git(
            "clone", "--quiet", "--filter=blob:none", "--sparse",
            f"--depth={HISTORY_DEPTH}", "--no-checkout", REPO_URL, str(cache_dir),
        )
        _git("sparse-checkout", "set", f"csv/{LANGUAGE}", cwd=cache_dir)
    _git("fetch", "--quiet", f"--depth={HISTORY_DEPTH}", "origin", ref, cwd=cache_dir)
    _git("checkout", "--quiet", "--detach", "FETCH_HEAD", cwd=cache_dir)
    return open_snapshot(cache_dir)


def open_snapshot(cache_dir: Path) -> Snapshot:
    """Describe an existing checkout without touching the network."""
    if not (cache_dir / ".git").exists():
        raise FileNotFoundError(f"no datamining checkout at {cache_dir}; run `xivlore fetch`")
    commit, commit_date = _git("log", "-1", "--format=%H %cs", cwd=cache_dir).split()
    subjects = _git("log", "--format=%s", cwd=cache_dir).splitlines()
    game_version = next(
        (m.group(1) for s in subjects if (m := GLOBAL_PATCH.match(s))), None
    )
    return Snapshot(cache_dir, commit, commit_date, game_version)
