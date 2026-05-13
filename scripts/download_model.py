"""Download the fine-tuned BART-SAMSum model from Hugging Face Hub.

Use this instead of running `python main.py` if you just want to serve the
existing fine-tuned weights. Saves 2-3 hours of CPU training.

Usage:
    python scripts/download_model.py
    HF_MODEL_REPO=your-name/your-fork python scripts/download_model.py
"""
import os
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = os.environ.get("HF_MODEL_REPO", "biggdaddyy/bart-samsum-finetuned")
MODEL_DIR = Path("artifacts/model_trainer/bart-samsum-model")
TOKENIZER_DIR = Path("artifacts/model_trainer/tokenizer")


def main() -> None:
    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {REPO} → {MODEL_DIR} ...")
    snapshot_download(repo_id=REPO, local_dir=str(MODEL_DIR))

    if TOKENIZER_DIR.exists() or TOKENIZER_DIR.is_symlink():
        TOKENIZER_DIR.unlink() if TOKENIZER_DIR.is_symlink() else None
    if not TOKENIZER_DIR.exists():
        TOKENIZER_DIR.symlink_to(MODEL_DIR.resolve())
    print(f"Done. Tokenizer dir symlinked → {TOKENIZER_DIR}")
    print("Now run: uvicorn app:app --host 0.0.0.0 --port 8080")


if __name__ == "__main__":
    main()
