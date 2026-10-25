# ffxivLoreBot

A Final Fantasy XIV lore chat model, starting with its training corpus.

## Corpus builder (`xivlore`)

`xivlore` compiles English in-game text into a Hugging Face dataset:

| Config | One row is |
|---|---|
| `transcripts` (default) | a quest, voiced cutscene, or NPC conversation as `Speaker: line` text |
| `dialogue` | one line, with speaker, line kind, and quest metadata |
| `descriptions` | flavour text for an item, minion, mount, card, fish, place, FATE, or leve |

The source is [xivapi/ffxiv-datamining](https://github.com/xivapi/ffxiv-datamining),
a CSV mirror of the game's text sheets. Only the English CSVs are downloaded,
about 220 MB.

```sh
uv sync
uv run xivlore fetch                 # sparse-clone or update into .cache/
uv run xivlore build                 # write Parquet configs and dataset card to dist/
uv run hf auth login                 # once, with a write token
uv run xivlore push <user>/ffxiv-lore
```

`fetch --ref <commit>` pins an upstream commit. Every build records the
upstream commit and game patch in `dist/manifest.json` and in the dataset card.

Then, from any project:

```python
from datasets import load_dataset

transcripts = load_dataset("<user>/ffxiv-lore", split="train")
lines = load_dataset("<user>/ffxiv-lore", "dialogue", split="train")
```

A private dataset needs `hf auth login` (or `HF_TOKEN`) on the machine that loads it.

### Copyright

The game text is © SQUARE ENIX and is not licensed for redistribution. `push`
creates the dataset private and refuses to upload to a public repository.
Neither the raw CSVs (`.cache/`) nor the built dataset (`dist/`) are committed
here.

### Adding a source

Each source lives in `src/xivlore/sources/<name>/` and yields records for the
build step (`src/xivlore/build.py`) to write as Parquet configs.

### Tests

```sh
uv run pytest
```
