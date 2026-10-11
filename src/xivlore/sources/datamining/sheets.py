"""Readers for the two CSV layouts in ffxiv-datamining.

Game data sheets (``Quest.csv``) have a header row whose first column is ``#``,
the row id. Script text sheets (``quest/000/ClsArc000_00021.csv``) have no
header: every row is ``index,key,text``.
"""

import csv
from collections.abc import Iterator
from pathlib import Path

Row = dict[str, str]


def read_sheet(path: Path) -> dict[int, Row]:
    """Read a game data sheet into ``{row_id: {column: value}}``."""
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        if header[0] != "#":
            raise ValueError(f"{path} is not a headed sheet (first column {header[0]!r})")
        return {int(row[0]): dict(zip(header, row)) for row in reader}


def require_columns(sheet: dict[int, Row], name: str, *columns: str) -> None:
    """Fail loudly when an upstream schema change drops a column we rely on."""
    present = next(iter(sheet.values()), {})
    missing = [c for c in columns if c not in present]
    if missing:
        raise KeyError(f"{name} is missing columns {missing}; upstream schema changed?")


def read_script(path: Path) -> Iterator[tuple[int, str, str]]:
    """Yield ``(index, key, text)`` from a script text sheet."""
    with path.open(encoding="utf-8", newline="") as f:
        for index, key, text in csv.reader(f):
            yield int(index), key, text
