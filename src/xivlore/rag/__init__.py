"""Retrieval-augmented answering over the built corpus with a local Ollama model."""

SEARCH_MODES = ("vector", "keyword", "hybrid")
# Hybrid scored 16/23 on the eval against 18/23 for vector alone: keywords found
# exact terms (Endsinger, sundered) but common words like "Limsa" swamped others.
DEFAULT_SEARCH = "vector"
