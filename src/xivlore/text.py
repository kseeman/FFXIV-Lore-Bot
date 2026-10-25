"""Normalisation applied to every piece of game text entering the corpus."""

import re

# Button and status glyphs live in the Unicode private use area; they render as
# icons in game and as noise everywhere else.
_PRIVATE_USE = re.compile(r"[\ue000-\uf8ff]")
_BOLD = re.compile(r"\*\*")
_TRAILING_SPACE = re.compile(r"[ \t]+$", re.MULTILINE)
# English sheets carry some untranslated developer placeholders such as
# "（★未使用／削除予定★）" ("unused / to be deleted"). Match kana and kanji only:
# English text does use full-width forms such as "～" and "／".
_JAPANESE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]")
# Developer filler left in cut or unvoiced lines.
_PLACEHOLDER = re.compile(
    r"(?i)report if displayed|isn't the text you're looking for"
    r"|^\W*(marked for deletion|to be deleted|deleted?|placeholder|unused|test)\W*$"
)


def clean(text: str) -> str:
    """Strip rendering artefacts, keeping the words and in-line cues like <sigh>."""
    text = _PRIVATE_USE.sub("", text)
    text = _BOLD.sub("", text)
    text = _TRAILING_SPACE.sub("", text)
    return text.strip()


def is_untranslated(text: str) -> bool:
    return bool(_JAPANESE.search(text))


def is_filler(text: str) -> bool:
    """True for text that is not real game prose: untranslated or a placeholder."""
    return is_untranslated(text) or bool(_PLACEHOLDER.search(text))
