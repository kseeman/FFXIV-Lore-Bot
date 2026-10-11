"""A miniature ffxiv-datamining ``csv/en`` tree with just enough rows to join."""

import csv
from pathlib import Path

import pytest


def write_csv(path: Path, rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(rows)


@pytest.fixture
def csv_dir(tmp_path: Path) -> Path:
    root = tmp_path / "csv" / "en"
    write_csv(root / "ENpcResident.csv", [
        ["#", "Singular", "Plural"],
        [0, "", ""],
        [1, "Luciane", ""],
        [2, "Y'shtola", ""],
        [3, "Zhloe Aliapoh", ""],
        [4, "wandering minstrel", ""],
    ])
    write_csv(root / "BNpcName.csv", [["#", "Singular", "Plural"], [1, "Ifrit", ""]])
    write_csv(root / "ExVersion.csv", [
        ["#", "Name"], [0, "A Realm Reborn"], [1, "Heavensward"], [2, "Stormblood"],
        [3, "Shadowbringers"],
    ])
    write_csv(root / "PlaceName.csv", [["#", "Name"], [0, ""], [39, "Gridania"]])
    write_csv(root / "JournalSection.csv", [["#", "Name"], [2, "Class & Job Quests"]])
    write_csv(root / "JournalCategory.csv", [
        ["#", "Name", "JournalSection"], [10, "Disciple of War Quests", 2],
    ])
    write_csv(root / "JournalGenre.csv", [
        ["#", "Name", "JournalCategory"], [0, "", 0], [20, "Archer Quests", 10],
    ])
    write_csv(root / "Quest.csv", [
        ["#", "Name", "Id", "JournalGenre", "PlaceName", "ClassJobLevel[0]", "Expansion"],
        [65536, "", "", 0, 0, 0, 0],
        [65621, "Way of the Archer", "ClsArc000_00021", 20, 39, 1, 0],
    ])
    write_csv(root / "quest" / "000" / "ClsArc000_00021.csv", [
        [0, "TEXT_CLSARC000_00021_SEQ_00", "Athelyna wishes you to join the guild."],
        [1, "TEXT_CLSARC000_00021_SEQ_01", ""],
        [2, "TEXT_CLSARC000_00021_TODO_00", "Speak with Luciane."],
        [3, "TEXT_CLSARC000_00021_LUCIANE_000_10", "Greetings, **adventurer**."],
        [4, "TEXT_CLSARC000_00021_Q1_000_1", "Will you dedicate yourself to archery?"],
        [5, "TEXT_CLSARC000_00021_A1_000_1", "Yes"],
        [6, "TEXT_CLSARC000_00021_A1_000_2", "No"],
        [7, "TEXT_CLSARC000_00021_LUCIANE_000_11", "<sigh> Very well."],
        [8, "TEXT_CLSARC000_00021_LUCIANE_000_12", "To be deleted."],
        [9, "TEXT_CLSARC000_00021_ATHELYNA_000_1", "（★未使用／削除予定★）"],
    ])
    write_csv(root / "cut_scene" / "052" / "VoiceMan_05200.csv", [
        [0, "TEXT_VOICEMAN_05200_000010_YSHTOLA", "We must hurry."],
        [1, "TEXT_VOICEMAN_05200_000020_MYSTERYVOICE", "(-???-)Wait..."],
    ])
    write_csv(root / "custom" / "003" / "CtsSfsCharacter1_00386.csv", [
        [0, "TEXT_CTSSFSCHARACTER1_00386_LCUT_RANKUP_ZHLOEALIAPOH_001_000", "Splendid!"],
    ])
    write_csv(root / "custom" / "000" / "CmnDefLinkShell_00006.csv", [
        [0, "TEXT_CMNDEFLINKSHELL_00006_ENTER_GUIDE_001", "To join a linkshell..."],
    ])
    write_csv(root / "shop" / "ShopMenu_00001.csv", [
        [0, "TEXT_SHOPMENU_00001_UI_TITLE", "Crystal Exchange"],
    ])
    write_csv(root / "Item.csv", [
        ["#", "Singular", "Description", "Name"],
        [0, "", "", ""],
        [1, "gil", "Standard Eorzean currency.", "Gil"],
        [4869, "", "", "Rainbow Trout"],
    ])
    write_csv(root / "FishParameter.csv", [
        ["#", "Text", "Item"], [1, "A trout of many colours.", 4869], [2, "", 0],
    ])
    write_csv(root / "Mount.csv", [["#", "Singular"], [1, "company chocobo"]])
    write_csv(root / "MountTransient.csv", [
        ["#", "Description", "DescriptionEnhanced", "Tooltip"],
        [1, "Summon your chocobo.", "A loyal bird.", ""],
    ])
    return root
