#!/usr/bin/env python3
"""Finalize partially-written census .npz files from temp .npy batches.

Usage:
    python finalize_census.py runs/llama-3-8b/census-v4

This is useful when the main extractor has finished all forward passes but is
still serializing the final .npz files one layer at a time. It reads the
existing temp .npy files, concatenates them, and writes the compressed .npz
for every incomplete layer in parallel.
"""
from __future__ import annotations

import concurrent.futures
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import orjson


def finalize_layer(tmp_dir: Path, out_path: Path) -> tuple[Path, bool, str, float]:
    """Concatenate temp .npy files into one compressed .npz."""
    import time

    t0 = time.time()
    try:
        if not tmp_dir.exists():
            return out_path, False, "no tmp dir", time.time() - t0

        # F2 renamed per-batch temp files from {k}_batch{idx}.npy to
        # {k}_chunk{idx:06d}.npy. Support BOTH layouts so the recovery tool works
        # on old crashed runs (Half-1 batch naming) and new F2 runs (chunk
        # naming) alike.
        npy_files = sorted(
            list(tmp_dir.glob("*_batch*.npy")) + list(tmp_dir.glob("*_chunk*.npy"))
        )
        if not npy_files:
            return out_path, False, "no temp files", time.time() - t0

        # Drop incomplete scratch files left by interrupted atomic writes.
        for writing in tmp_dir.glob("*.tmp.npy"):
            writing.unlink(missing_ok=True)

        # Group files by field key. The key is everything before the
        # _batch/_chunk suffix; split on whichever delimiter is present.
        field_files: dict[str, list[Path]] = {}
        for p in npy_files:
            stem = p.stem
            if "_chunk" in stem:
                key = stem.rsplit("_chunk", 1)[0]
            else:
                key = stem.rsplit("_batch", 1)[0]
            field_files.setdefault(key, []).append(p)

        final_arrays: dict[str, np.ndarray] = {}
        metadata = None
        for key, paths in field_files.items():
            if key == "_metadata":
                # Metadata is a single JSON array stored as a .npy uint8 buffer.
                meta_arr = np.load(paths[0])
                metadata = orjson.loads(meta_arr.tobytes())
                final_arrays["_metadata"] = meta_arr
                continue

            ordered = sorted(paths)
            if not ordered:
                continue

            # Two-pass preallocation: avoid repeated concatenate copies.
            first = np.load(ordered[0])
            total = sum(int(np.load(p, allow_pickle=False).shape[0]) for p in ordered)
            out_shape = (total,) + first.shape[1:]
            stacked = np.empty(out_shape, dtype=first.dtype)
            del first

            cursor = 0
            for part_path in ordered:
                part = np.load(part_path)
                n = part.shape[0]
                stacked[cursor : cursor + n] = part
                cursor += n
                del part

            if stacked.dtype == np.float32:
                stacked = stacked.astype(np.float16)
            final_arrays[key] = stacked

        if metadata is None:
            # Crash-safe journal fallback: io.py now writes _metadata.jsonl
            # (one orjson line per row, fsync'd per batch) so a hard kill
            # mid-loop -- the OOM-sweep's failure mode -- no longer loses the
            # prompt/seq_len/max_token_idx association. Rebuild the metadata
            # buffer from the journal when _metadata.npy never got written.
            journal = tmp_dir / "_metadata.jsonl"
            if journal.exists():
                rows: list = []
                with journal.open("rb") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            rows.append(orjson.loads(line))
                if rows:
                    meta_arr = np.frombuffer(
                        orjson.dumps(rows, default=str), dtype=np.uint8
                    )
                    metadata = rows
                    final_arrays["_metadata"] = meta_arr
            if metadata is None:
                return out_path, False, "missing metadata file", time.time() - t0

        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Default finalize_census to uncompressed for speed.
        np.savez(out_path, **final_arrays)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return out_path, True, f"{len(metadata)} rows", time.time() - t0

    except Exception as exc:
        return out_path, False, str(exc), time.time() - t0


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Finalize census .npz files from temp .npy batches")
    parser.add_argument("census_dir", type=Path, help="directory containing .tmp layer directories")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="number of layers to finalize in parallel (default: 0 = auto, up to 8)",
    )
    args = parser.parse_args()

    root = args.census_dir
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")

    tmp_dirs = [d for d in root.iterdir() if d.is_dir() and d.suffix == ".tmp"]
    if not tmp_dirs:
        print("no .tmp directories found; nothing to finalize")
        return

    max_workers = args.workers
    if max_workers == 0:
        max_workers = min(8, os.cpu_count() or 1)
    print(f"finalizing {len(tmp_dirs)} layers from {root} with {max_workers} worker(s)")
    print("(each dot is one layer completed)")

    from tqdm import tqdm

    completed = 0
    failed = 0

    def run_one(tmp_dir: Path) -> tuple[Path, bool, str, float]:
        return finalize_layer(tmp_dir, tmp_dir.with_suffix(""))

    with tqdm(total=len(tmp_dirs), unit="layer", desc="finalize") as pbar:
        if max_workers == 1:
            for tmp_dir in sorted(tmp_dirs):
                out_path, ok, msg, elapsed = run_one(tmp_dir)
                pbar.set_postfix({out_path.name: f"{elapsed:.1f}s"})
                pbar.update(1)
                if not ok:
                    pbar.write(f"[FAIL] {out_path.name}: {msg}")
                    failed += 1
                completed += 1
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(run_one, tmp_dir): tmp_dir for tmp_dir in tmp_dirs}
                for fut in concurrent.futures.as_completed(futures):
                    out_path, ok, msg, elapsed = fut.result()
                    pbar.set_postfix({out_path.name: f"{elapsed:.1f}s"})
                    pbar.update(1)
                    if not ok:
                        pbar.write(f"[FAIL] {out_path.name}: {msg}")
                        failed += 1
                    completed += 1

    print(f"done — {completed}/{len(tmp_dirs)} layers finalized ({failed} failed)")


if __name__ == "__main__":
    main()
