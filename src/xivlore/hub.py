"""Publish a built corpus to a private Hugging Face dataset repository."""

import json
from pathlib import Path

from huggingface_hub import HfApi

from xivlore import card


class NotPrivateError(RuntimeError):
    pass


def push(out_dir: Path, repo_id: str) -> str:
    """Upload ``out_dir`` to ``repo_id``, creating it private if needed.

    Refuses to upload to a repository that is not private: the corpus is
    Square Enix copyright and is not licensed for public redistribution.
    """
    manifest = json.loads((out_dir / "manifest.json").read_text())
    # Re-render so the card's load_dataset() snippet names the real repository.
    (out_dir / "README.md").write_text(card.render(manifest, repo_id))

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", visibility="private", exist_ok=True)
    if not api.repo_info(repo_id, repo_type="dataset").private:
        raise NotPrivateError(
            f"{repo_id} is public; make it private on Hugging Face before pushing"
        )

    source = manifest["source"]
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=out_dir,
        commit_message=(
            f"Build from ffxiv-datamining {source['commit'][:8]}"
            f" (game {source['game_version'] or 'unknown'})"
        ),
        # Drop data files a previous build wrote but this one did not.
        delete_patterns=["*.parquet"],
    )
    return commit.commit_url
