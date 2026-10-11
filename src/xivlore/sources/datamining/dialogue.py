"""Scripted dialogue: quests, voiced cutscenes, and duty/NPC conversations.

Every line's key encodes who says it, e.g. ``TEXT_CLSARC000_00021_LUCIANE_000_10``:
``TEXT``, the script name, the script number, then role tokens. The first role
token is either an NPC's name in upper case without punctuation, or one of the
non-dialogue markers in ``ROLE_KINDS``. Some keys put a scene label before the
name (``LCUT_RANKUP_ZHLOEALIAPOH``), so the speaker is the first role token
that matches a known name.
"""

import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from xivlore.text import clean, is_filler

from .sheets import read_script, read_sheet, require_columns

# Script folders worth keeping, and the corpus category each one maps to. The
# rest (shop, warp, transport, system, ...) are menu strings, not lore.
FOLDERS = {
    "quest": "quest",
    "cut_scene": "cutscene",
    "custom": "npc",
    "opening": "npc",
    "dungeon": "duty",
    "raid": "duty",
    "content": "duty",
}

ROLE_KINDS = [
    (re.compile(r"SEQ"), "journal"),
    (re.compile(r"TODO|QIB"), "objective"),
    (re.compile(r"SYSTEM|POP|POPMESSAGE|MENU"), "system"),
    (re.compile(r"Q\d+"), "choice_prompt"),
    (re.compile(r"A\d+"), "choice"),
]

# Voiced cutscene scripts are numbered by patch: VoiceMan_05300 is patch 5.3.
# Expansions are numbered from 2.x (A Realm Reborn = ExVersion row 0).
_CUTSCENE_PATCH = re.compile(r"VoiceMan_0(\d)\d", re.IGNORECASE)
_FIRST_EXPANSION_MAJOR = 2
_SYSTEM_SUFFIXES = {"SYS", "SYSTEM"}


@dataclass
class Line:
    id: str
    category: str
    script: str
    index: int
    key: str
    kind: str
    speaker_key: str | None
    speaker: str | None
    text: str
    quest_id: int | None = None
    quest_name: str | None = None
    expansion: str | None = None
    journal_genre: str | None = None
    journal_category: str | None = None
    journal_section: str | None = None
    place: str | None = None
    level: int | None = None


def parse_roles(key: str) -> list[str]:
    """Return the role tokens of a line key.

    Drop ``TEXT`` and the script name, then skip numeric and empty tokens
    (script, scene and line numbers). Quests put the role before the line
    numbers, cutscenes after them.
    """
    tokens = key.split("_")
    if len(tokens) < 3 or tokens[0] != "TEXT":
        return []
    return [t for t in tokens[2:] if t and not t.isdigit()]


def classify_roles(roles: list[str]) -> str:
    if not roles:
        return "unknown"
    for pattern, kind in ROLE_KINDS:
        if pattern.fullmatch(roles[0]):
            return kind
    if _SYSTEM_SUFFIXES.intersection(roles):
        return "system"
    return "dialogue"


def _normalise_name(name: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", name.upper())


def _normalise_role(role: str) -> str:
    # Disambiguating suffixes: DWARFMASTER03671 is the same NPC as DWARFMASTER.
    return re.sub(r"\d+$", "", role)


class SpeakerNames:
    """Map role tokens like ``YSHTOLA`` back to display names like ``Y'shtola``."""

    def __init__(self, names: list[str]) -> None:
        spellings: dict[str, Counter[str]] = {}
        for name in names:
            name = clean(name)
            if name and not is_filler(name):
                spellings.setdefault(_normalise_name(name), Counter())[name] += 1
        self._names = {
            key: _capitalise(counts.most_common(1)[0][0])
            for key, counts in spellings.items()
        }

    @classmethod
    def from_csv_dir(cls, csv_dir: Path) -> "SpeakerNames":
        names = [
            row["Singular"]
            for sheet in ("ENpcResident", "BNpcName")
            for row in read_sheet(csv_dir / f"{sheet}.csv").values()
        ]
        return cls(names)

    def resolve(self, roles: list[str]) -> tuple[str, str, bool]:
        """Return ``(role key, display name, resolved)`` for a line's role tokens.

        The first token matching a known name wins; if none does, the first
        token is title-cased as a best guess.
        """
        for role in roles:
            if name := self._names.get(_normalise_name(_normalise_role(role))):
                return role, name, True
        return roles[0], _normalise_role(roles[0]).title(), False


def _capitalise(name: str) -> str:
    return name[:1].upper() + name[1:]


@dataclass
class QuestInfo:
    quest_id: int
    name: str
    expansion: str | None
    journal_genre: str | None
    journal_category: str | None
    journal_section: str | None
    place: str | None
    level: int | None


def load_quests(csv_dir: Path) -> dict[str, QuestInfo]:
    """Quest metadata keyed by upper-cased script name (``CLSARC000_00021``)."""
    quests = read_sheet(csv_dir / "Quest.csv")
    require_columns(
        quests, "Quest", "Id", "Name", "Expansion", "JournalGenre", "PlaceName",
        "ClassJobLevel[0]",
    )
    expansions = _names(csv_dir, "ExVersion")
    places = _names(csv_dir, "PlaceName")
    genres = read_sheet(csv_dir / "JournalGenre.csv")
    categories = read_sheet(csv_dir / "JournalCategory.csv")
    sections = _names(csv_dir, "JournalSection")

    info = {}
    for row_id, row in quests.items():
        if not row["Id"] or not row["Name"]:
            continue
        genre = genres.get(int(row["JournalGenre"]))
        category = categories.get(int(genre["JournalCategory"])) if genre else None
        info[row["Id"].upper()] = QuestInfo(
            quest_id=row_id,
            name=clean(row["Name"]),
            expansion=expansions.get(int(row["Expansion"])),
            journal_genre=clean(genre["Name"]) or None if genre else None,
            journal_category=clean(category["Name"]) or None if category else None,
            journal_section=(
                sections.get(int(category["JournalSection"])) if category else None
            ),
            place=places.get(int(row["PlaceName"])),
            level=int(row["ClassJobLevel[0]"]) or None,
        )
    return info


def _names(csv_dir: Path, sheet: str) -> dict[int, str]:
    rows = read_sheet(csv_dir / f"{sheet}.csv")
    return {row_id: name for row_id, row in rows.items() if (name := clean(row["Name"]))}


def cutscene_expansion(script: str, expansions: dict[int, str]) -> str | None:
    match = _CUTSCENE_PATCH.match(script)
    if not match:
        return None
    return expansions.get(int(match.group(1)) - _FIRST_EXPANSION_MAJOR)


def extract(csv_dir: Path) -> Iterator[Line]:
    """Yield every non-empty, translated line from the kept script folders."""
    speakers = SpeakerNames.from_csv_dir(csv_dir)
    quests = load_quests(csv_dir)
    expansions = _names(csv_dir, "ExVersion")

    for folder, category in FOLDERS.items():
        for path in sorted((csv_dir / folder).rglob("*.csv")):
            script = path.stem
            quest = quests.get(script.upper()) if category == "quest" else None
            expansion = (
                quest.expansion if quest
                else cutscene_expansion(script, expansions) if category == "cutscene"
                else None
            )
            sheet = path.relative_to(csv_dir).with_suffix("").as_posix()
            for index, key, raw in read_script(path):
                text = clean(raw)
                if not text or is_filler(text):
                    continue
                roles = parse_roles(key)
                kind = classify_roles(roles)
                speaker_key = speaker = None
                if kind == "dialogue":
                    speaker_key, speaker, _ = speakers.resolve(roles)
                yield Line(
                    id=f"{sheet}#{index}",
                    category=category,
                    script=script,
                    index=index,
                    key=key,
                    kind=kind,
                    speaker_key=speaker_key,
                    speaker=speaker,
                    text=text,
                    quest_id=quest.quest_id if quest else None,
                    quest_name=quest.name if quest else None,
                    expansion=expansion,
                    journal_genre=quest.journal_genre if quest else None,
                    journal_category=quest.journal_category if quest else None,
                    journal_section=quest.journal_section if quest else None,
                    place=quest.place if quest else None,
                    level=quest.level if quest else None,
                )
