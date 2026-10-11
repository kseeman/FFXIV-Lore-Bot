import pytest

from xivlore.sources.datamining.dialogue import (
    SpeakerNames, classify_roles, cutscene_expansion, extract, parse_roles,
)


@pytest.mark.parametrize("key, roles", [
    ("TEXT_CLSARC000_00021_LUCIANE_000_10", ["LUCIANE"]),
    ("TEXT_VOICEMAN_02200_000010_THANCRED", ["THANCRED"]),
    ("TEXT_HEAVNZ411_01829__BOTANISTMOG01829_000_010", ["BOTANISTMOG01829"]),
    ("TEXT_CTSSFSCHARACTER1_00386_LCUT_RANKUP_ZHLOEALIAPOH_001_000",
     ["LCUT", "RANKUP", "ZHLOEALIAPOH"]),
    ("NOT_A_TEXT_KEY", []),
])
def test_parse_roles(key, roles):
    assert parse_roles(key) == roles


@pytest.mark.parametrize("roles, kind", [
    (["SEQ"], "journal"),
    (["TODO"], "objective"),
    (["QIB", "TODO"], "objective"),
    (["Q1"], "choice_prompt"),
    (["A12"], "choice"),
    (["POP", "MESSAGE"], "system"),
    (["LEAVE", "GUIDE", "SYS"], "system"),
    (["ALPHINAUD"], "dialogue"),
    ([], "unknown"),
])
def test_classify_roles(roles, kind):
    assert classify_roles(roles) == kind


def test_speaker_names_resolution():
    names = SpeakerNames(["Y'shtola", "G'raha Tia", "wandering minstrel", "Dwarf Master"])
    assert names.resolve(["YSHTOLA"]) == ("YSHTOLA", "Y'shtola", True)
    assert names.resolve(["GRAHATIA"]) == ("GRAHATIA", "G'raha Tia", True)
    assert names.resolve(["WANDERINGMINSTREL"])[1] == "Wandering minstrel"
    assert names.resolve(["DWARFMASTER03671"])[1:] == ("Dwarf Master", True)
    assert names.resolve(["LCUT", "RANKUP", "GRAHATIA"])[:2] == ("GRAHATIA", "G'raha Tia")
    assert names.resolve(["MYSTERYVOICE"]) == ("MYSTERYVOICE", "Mysteryvoice", False)


def test_cutscene_expansion():
    expansions = {0: "A Realm Reborn", 3: "Shadowbringers"}
    assert cutscene_expansion("VoiceMan_05200", expansions) == "Shadowbringers"
    assert cutscene_expansion("VoiceMan_02200", expansions) == "A Realm Reborn"
    assert cutscene_expansion("ClsArc000_00021", expansions) is None


def test_extract_quest_lines(csv_dir):
    lines = [l for l in extract(csv_dir) if l.category == "quest"]
    assert [(l.index, l.kind, l.speaker, l.text) for l in lines] == [
        (0, "journal", None, "Athelyna wishes you to join the guild."),
        (2, "objective", None, "Speak with Luciane."),
        (3, "dialogue", "Luciane", "Greetings, adventurer."),
        (4, "choice_prompt", None, "Will you dedicate yourself to archery?"),
        (5, "choice", None, "Yes"),
        (6, "choice", None, "No"),
        (7, "dialogue", "Luciane", "<sigh> Very well."),
    ]
    line = lines[2]
    assert line.id == "quest/000/ClsArc000_00021#3"
    assert (line.quest_id, line.quest_name, line.expansion) == (
        65621, "Way of the Archer", "A Realm Reborn",
    )
    assert (line.journal_genre, line.journal_category, line.journal_section) == (
        "Archer Quests", "Disciple of War Quests", "Class & Job Quests",
    )
    assert (line.place, line.level) == ("Gridania", 1)


def test_extract_other_folders(csv_dir):
    lines = [l for l in extract(csv_dir) if l.category != "quest"]
    assert [(l.category, l.speaker, l.expansion, l.text) for l in lines] == [
        ("cutscene", "Y'shtola", "Shadowbringers", "We must hurry."),
        ("cutscene", "Mysteryvoice", "Shadowbringers", "Wait..."),
        ("npc", "Zhloe Aliapoh", None, "Splendid!"),
    ]
