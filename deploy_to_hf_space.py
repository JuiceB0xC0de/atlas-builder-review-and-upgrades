#!/usr/bin/env python3
"""
Create the private HF Space and upload the atlas runner files.

Run this from /Users/chiggy/atlasing after `huggingface-cli login`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--space-id", default="juiceb0xc0de/atlasing")
    p.add_argument("--local-dir", default=".", type=Path)
    p.add_argument("--private", action="store_true", default=True)
    args = p.parse_args()

    api = HfApi()

    print(f"[hf] creating space {args.space_id} (private={args.private})")
    try:
        create_repo(
            args.space_id,
            repo_type="space",
            space_sdk="static",
            private=args.private,
            exist_ok=True,
        )
        print(f"[hf] space ready: https://huggingface.co/spaces/{args.space_id}")
    except Exception as exc:
        print(f"[hf] create_repo error: {exc}")
        raise

    print(f"[hf] uploading files from {args.local_dir.resolve()}")
    upload_folder(
        repo_id=args.space_id,
        repo_type="space",
        folder_path=str(args.local_dir),
        path_in_repo="",
        ignore_patterns=["outputs/*", "atlas/*", ".venv/*", "__pycache__/*", "*.pyc", ".DS_Store",
                          "*.tmp", "*.npz", "*.npy", "*.sqlite", "sub_zero_ckpt/*", "l*_census_raw*"],
    )
    print("[hf] upload complete")


if __name__ == "__main__":
    main()
