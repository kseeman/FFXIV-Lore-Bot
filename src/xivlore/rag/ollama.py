"""Minimal client for a local Ollama server's embed and chat endpoints."""

import json
import urllib.error
import urllib.request

HOST = "http://localhost:11434"
EMBED_MODEL = "embeddinggemma"
CHAT_MODEL = "gemma4:12b-it-qat"


class OllamaError(RuntimeError):
    pass


def embed(texts: list[str], model: str = EMBED_MODEL) -> list[list[float]]:
    return _post("/api/embed", {"model": model, "input": texts})["embeddings"]


def embed_and_unload(texts: list[str]) -> list[list[float]]:
    """Embed, then free the embedding model's VRAM so the chat model fits on the GPU."""
    return _post("/api/embed", {
        "model": EMBED_MODEL, "input": texts, "keep_alive": 0,
    })["embeddings"]


def chat(
    messages: list[dict],
    model: str = CHAT_MODEL,
    num_ctx: int = 6144,
    temperature: float | None = None,
) -> str:
    options = {"num_ctx": num_ctx}
    if temperature is not None:
        options["temperature"] = temperature
    response = _post("/api/chat", {
        "model": model,
        "messages": messages,
        "stream": False,
        # Gemma 4 thinks by default: on the eval that took 3-4x as long for the
        # same score, and now and then left no answer after the thinking.
        "think": False,
        "options": options,
    })
    return response["message"]["content"]


def _post(path: str, payload: dict) -> dict:
    request = urllib.request.Request(
        HOST + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request) as response:
            return json.load(response)
    except urllib.error.HTTPError as e:
        raise OllamaError(f"{path}: {e.code} {e.read().decode(errors='replace')}") from e
    except urllib.error.URLError as e:
        raise OllamaError(f"cannot reach Ollama at {HOST} ({e.reason}); is it running?") from e
