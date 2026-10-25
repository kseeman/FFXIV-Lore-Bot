import pytest

from xivlore.text import clean, is_filler


def test_clean_strips_icons_bold_and_trailing_space():
    assert clean(" **Quest Sync**  \nPress  to continue.  ") == (
        "Quest Sync\nPress  to continue."
    )


def test_clean_keeps_inline_cues_and_dashes():
    assert clean("<sigh> Very well─if you insist.") == "<sigh> Very well─if you insist."


@pytest.mark.parametrize("text", [
    "deleted", "To be deleted.", "(Marked for deletion)", "placeholder", "TEST",
    "No subtitles. Report if displayed.", "[GASP] Report if displayed.",
    "This isn't the text you're looking for.", "（★未使用／削除予定★）",
])
def test_is_filler(text):
    assert is_filler(text)


@pytest.mark.parametrize("text", [
    "Greetings, adventurer.", "The test of the Twelve awaits.", "Nothing was deleted.",
    "O～hohoho! Are those crates I spy?", "Torn from the Heavens／The Dark Colossus",
])
def test_is_not_filler(text):
    assert not is_filler(text)
