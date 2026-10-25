"""Flavour text attached to game entities: items, minions, mounts, cards, fish...

Some sheets keep the name and the text in the same row. Others split them: the
name lives in ``Mount`` and the text in ``MountTransient``, sharing a row id,
or the text row points at its name through a reference column.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from xivlore.text import clean, is_filler

from .sheets import read_sheet, require_columns


@dataclass(frozen=True)
class Spec:
    category: str
    text_sheet: str
    text_columns: tuple[str, ...]
    name_sheet: str
    name_column: str
    # Column in the text sheet holding the name sheet's row id. None: same row id.
    name_reference: str | None = None


SPECS = [
    Spec("item", "Item", ("Description",), "Item", "Name"),
    Spec("key_item", "EventItemHelp", ("Description",), "EventItem", "Name"),
    Spec("minion", "CompanionTransient", ("Description", "DescriptionEnhanced", "Tooltip"),
         "Companion", "Singular"),
    Spec("mount", "MountTransient", ("Description", "DescriptionEnhanced", "Tooltip"),
         "Mount", "Singular"),
    Spec("fashion_accessory", "OrnamentTransient", ("Text",), "Ornament", "Singular"),
    Spec("facewear", "Glasses", ("Description",), "Glasses", "Name"),
    Spec("triple_triad_card", "TripleTriadCard", ("Description",), "TripleTriadCard", "Name"),
    Spec("fish", "FishParameter", ("Text",), "Item", "Name", name_reference="Item"),
    Spec("orchestrion", "Orchestrion", ("Description",), "Orchestrion", "Name"),
    Spec("sightseeing", "Adventure", ("Impression", "Description"), "Adventure", "Name"),
    Spec("fate", "Fate", ("Description",), "Fate", "Name"),
    Spec("leve", "Leve", ("Description",), "Leve", "Name"),
]


@dataclass
class Description:
    id: str
    category: str
    sheet: str
    row_id: int
    name: str
    field: str
    text: str


def extract(csv_dir: Path, specs: list[Spec] = SPECS) -> Iterator[Description]:
    for spec in specs:
        text_rows = read_sheet(csv_dir / f"{spec.text_sheet}.csv")
        require_columns(
            text_rows, spec.text_sheet, *spec.text_columns,
            *([spec.name_reference] if spec.name_reference else []),
        )
        name_rows = (
            text_rows if spec.name_sheet == spec.text_sheet
            else read_sheet(csv_dir / f"{spec.name_sheet}.csv")
        )
        require_columns(name_rows, spec.name_sheet, spec.name_column)

        for row_id, row in text_rows.items():
            name_id = int(row[spec.name_reference]) if spec.name_reference else row_id
            name_row = name_rows.get(name_id)
            name = clean(name_row[spec.name_column]) if name_row else ""
            if not name or is_filler(name):
                continue
            for column in spec.text_columns:
                text = clean(row[column])
                if not text or is_filler(text):
                    continue
                yield Description(
                    id=f"{spec.text_sheet}/{row_id}/{column}",
                    category=spec.category,
                    sheet=spec.text_sheet,
                    row_id=row_id,
                    name=name,
                    field=column,
                    text=text,
                )
