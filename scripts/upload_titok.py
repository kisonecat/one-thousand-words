"""Publish the int8 TiTok ONNX files to the Hugging Face Hub.

    nix-shell --run 'python scripts/upload_titok.py [repo-name]'

Uses the token saved by `hf auth login`. Creates <you>/<repo-name> (public)
if needed, uploads hf/titok-b64-int8-onnx/README.md plus the two ONNX files,
and prints the commit to pin in web/config.js (VITE_TITOK_BASE).
"""
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
NAME = sys.argv[1] if len(sys.argv) > 1 else "titok-b64-int8-onnx"
FILES = {
    "README.md": ROOT / "hf/titok-b64-int8-onnx/README.md",
    "encoder_int8.onnx": ROOT / "models/titok-b64/encoder_int8.onnx",
    "decoder_int8.onnx": ROOT / "models/titok-b64/decoder_int8.onnx",
}

api = HfApi()
user = api.whoami()["name"]
repo_id = f"{user}/{NAME}"
api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
print(f"uploading to https://huggingface.co/{repo_id} ...", flush=True)

from huggingface_hub import CommitOperationAdd  # noqa: E402

commit = api.create_commit(
    repo_id,
    operations=[CommitOperationAdd(path_in_repo=k, path_or_fileobj=str(v)) for k, v in FILES.items()],
    commit_message="Add int8 ONNX export of TiTok-B-64 (encoder + decoder)",
)
sha = commit.oid
print(f"done: {commit.commit_url}")
print(f"VITE_TITOK_BASE=https://huggingface.co/{repo_id}/resolve/{sha}/")
