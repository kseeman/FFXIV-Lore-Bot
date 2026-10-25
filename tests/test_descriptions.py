import pytest

from xivlore.sources.datamining.descriptions import SPECS, extract
from xivlore.sources.datamining.sheets import read_sheet, require_columns


def specs(*categories):
    return [s for s in SPECS if s.category in categories]


def test_same_sheet_names(csv_dir):
    rows = list(extract(csv_dir, specs("item")))
    assert [(r.id, r.name, r.text) for r in rows] == [
        ("Item/1/Description", "Gil", "Standard Eorzean currency."),
    ]


def test_referenced_names(csv_dir):
    rows = list(extract(csv_dir, specs("fish")))
    assert [(r.name, r.text) for r in rows] == [("Rainbow Trout", "A trout of many colours.")]


def test_transient_sheet_names_and_multiple_fields(csv_dir):
    rows = list(extract(csv_dir, specs("mount")))
    assert [(r.name, r.field, r.text) for r in rows] == [
        ("company chocobo", "Description", "Summon your chocobo."),
        ("company chocobo", "DescriptionEnhanced", "A loyal bird."),
    ]


def test_require_columns_reports_schema_drift(csv_dir):
    sheet = read_sheet(csv_dir / "Mount.csv")
    with pytest.raises(KeyError, match="Lore"):
        require_columns(sheet, "Mount", "Singular", "Lore")
