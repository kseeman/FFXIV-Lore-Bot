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

## Asking questions (`xivlore ask`)

A retrieval prototype answers lore questions with a local model through
[Ollama](https://ollama.com): it finds the passages of game text closest to the
question and has the model answer from them, citing each one.

```sh
ollama pull embeddinggemma           # embeds passages and questions
ollama pull gemma4:12b-it-qat        # writes the answers
uv run xivlore index                 # embed a build into .cache/index/ (about 12 minutes)
uv run xivlore ask "Who leads the Crystarium?"
uv run xivlore ask --no-rag "Who leads the Crystarium?"   # the model alone
```

`--show-context` prints the retrieved passages. The default model fits a
12 GB GPU; the embedder is unloaded after each question to leave it room.

`xivlore eval` runs every question in `eval/questions.jsonl` with and without
retrieval and writes the answers side by side to `.cache/eval/`. Each question
has an expected answer and keywords a correct answer must mention. The index
and reports quote game text, so both stay under `.cache/`. The questions file
is committed, so it holds only questions and short answers, not quoted text.

## Tests

```sh
uv run pytest
```
