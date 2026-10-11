"""The Hugging Face dataset card (README.md) for a built corpus."""

from string import Template

_SIZE_CATEGORIES = [
    (1_000, "n<1K"), (10_000, "1K<n<10K"), (100_000, "10K<n<100K"),
    (1_000_000, "100K<n<1M"), (10_000_000, "1M<n<10M"), (100_000_000, "10M<n<100M"),
]

_TEMPLATE = Template("""\
---
pretty_name: Final Fantasy XIV Lore Corpus
language:
- en
license: other
license_name: proprietary-square-enix
task_categories:
- text-generation
- question-answering
tags:
- ffxiv
- final-fantasy
- video-games
- dialogue
- lore
size_categories:
- $size_category
configs:
- config_name: transcripts
  data_files: transcripts/*.parquet
  default: true
- config_name: dialogue
  data_files: dialogue/*.parquet
- config_name: descriptions
  data_files: descriptions/*.parquet
---

# Final Fantasy XIV Lore Corpus

English in-game text from *Final Fantasy XIV*: quest and cutscene dialogue,
quest journals, and the flavour text attached to items, minions, mounts, cards,
fish, and places. Built for training and grounding a lore-aware chat model.

Game version **$game_version**, extracted from
[xivapi/ffxiv-datamining](https://github.com/xivapi/ffxiv-datamining) at commit
[`$commit_short`](https://github.com/xivapi/ffxiv-datamining/commit/$commit)
($commit_date). Built $built_at by `xivlore`.

## Configs

| Config | Rows | Characters | One row is |
|---|---:|---:|---|
| `transcripts` | $transcripts_rows | $transcripts_chars | a whole quest, cutscene, or NPC conversation as `Speaker: line` text |
| `dialogue` | $dialogue_rows | $dialogue_chars | a single line, with speaker, line kind, and quest metadata |
| `descriptions` | $descriptions_rows | $descriptions_chars | one flavour-text field of one game entity |

```python
from datasets import load_dataset

transcripts = load_dataset("$repo_id", "transcripts", split="train")
lines = load_dataset("$repo_id", "dialogue", split="train")
```

### `dialogue`

- `category`: `quest`, `cutscene` (voiced main scenario scenes), `npc`
  (scripted NPC conversations), or `duty` (dungeon and raid scripts). Shared
  NPC interface scripts (retainers, inns, housing menus) are left out.
- `kind`: `dialogue` (spoken by `speaker`), `journal` (quest log summary),
  `objective`, `choice_prompt`, `choice` (a player response option), `system`
  (tutorial and UI prompts), or `unknown`.
- `speaker_key` is the role token from the game's line key (`YSHTOLA`);
  `speaker` is the display name it resolves to (`Y'shtola`).
  `display_speaker` is the name the game shows instead, when it overrides it:
  `???` for a speaker not yet revealed to the player.
- Quest lines carry `quest_name`, `expansion`, `journal_section`
  (e.g. *Main Scenario*), `journal_category`, `journal_genre`, `place`, and
  `level`. Cutscene `expansion` is inferred from the patch in the script name.
- `id` is `<sheet path>#<row>` and is stable across rebuilds of the same game
  version.

### `transcripts`

Dialogue grouped by script in game order: title, a metadata line, the journal
summary, then each spoken line as `Speaker: text`. Choices appear as the prompt
followed by `> option` lines. Objectives and system prompts are left out.

### `descriptions`

`category` is one of `item`, `key_item`, `minion`, `mount`,
`fashion_accessory`, `facewear`, `triple_triad_card`, `fish`, `orchestrion`,
`sightseeing`, `fate`, `leve`. `field` names the source column; minions and
mounts have several (`Description`, `DescriptionEnhanced`, `Tooltip`).

## Processing

- Icon glyphs (Unicode private use area) and `**bold**` markers are removed.
  Vocal cues such as `<sigh>` are kept.
- Untranslated developer text and placeholders ("To be deleted.", "Report if
  displayed.") are dropped.
- The player character's name appears as the literal word `Player`. Other
  in-game substitutions (item names, counts) were removed upstream, which
  leaves occasional gaps such as "You have already received .".
- Speaker names are matched against the game's NPC and enemy name sheets. When
  a role does not match a name (roughly a fifth of dialogue lines), `speaker`
  is the title-cased `speaker_key`, e.g. `Mysteryvoice`.

## Licence

All text is © SQUARE ENIX. This dataset is not licensed by Square Enix and no
rights are granted with it. It is shared privately for non-commercial education
and research. Square Enix's
[FINAL FANTASY XIV Materials Usage Policy](https://support.na.square-enix.com/rule.php?id=5382&tag=authc)
governs fan use of game materials; check it before any public or commercial use
of this data or of models trained on it.
""")


def render(manifest: dict, repo_id: str = "<user>/ffxiv-lore") -> str:
    source, configs = manifest["source"], manifest["configs"]
    values = {
        "size_category": _size_category(max(c["rows"] for c in configs.values())),
        "game_version": source["game_version"] or "unknown",
        "commit": source["commit"],
        "commit_short": source["commit"][:8],
        "commit_date": source["commit_date"],
        "built_at": manifest["built_at"],
        "repo_id": repo_id,
    }
    for name, stats in configs.items():
        values[f"{name}_rows"] = f"{stats['rows']:,}"
        values[f"{name}_chars"] = f"{stats['characters']:,}"
    return _TEMPLATE.substitute(values)


def _size_category(rows: int) -> str:
    return next((label for limit, label in _SIZE_CATEGORIES if rows < limit), "100M<n<1B")
