from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Iterable


def read_json(path: str | Path) -> Any:
    import orjson

    return orjson.loads(Path(path).read_bytes())


def read_census(path: str | Path) -> list[dict[str, Any]]:
    """Read a census file, transparently handling .json or .npz formats.

    The .npz format stores activation arrays in binary plus a small JSON
    metadata block, which is much smaller and faster than a huge JSON array.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix == ".npz":
        return _read_npz_census(p)
    return read_json(p)


def _read_npz_census(path: Path) -> list[dict[str, Any]]:
    import numpy as np
    import orjson

    z = np.load(path, allow_pickle=False)
    metadata = orjson.loads(z["_metadata"].tobytes())
    records = []
    for i, meta in enumerate(metadata):
        rec = dict(meta)
        for key in z.files:
            if key == "_metadata":
                continue
            arr = z[key]
            # Upcast float16 census arrays back to float32 for downstream code.
            if arr.dtype == np.float16:
                arr = arr.astype(np.float32)
            rec[key] = arr[i].tolist()
        records.append(rec)
    return records


def write_json(path: str | Path, obj: Any) -> None:
    import orjson

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(orjson.dumps(obj, option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY))


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    import orjson

    with Path(path).open("rb") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield orjson.loads(line)


def write_json_array_stream(path: str | Path):
    """Return a tiny context manager for streaming a JSON array incrementally."""

    class _Stream:
        def __init__(self, out: Path):
            self.out = out
            self.handle = None
            self.first = True
            self.count = 0

        def __enter__(self):
            self.out.parent.mkdir(parents=True, exist_ok=True)
            self.handle = self.out.open("wb")
            self.handle.write(b"[")
            return self

        def write(self, obj: Any) -> None:
            import orjson

            if not self.first:
                self.handle.write(b",")
            self.handle.write(orjson.dumps(obj, option=orjson.OPT_SERIALIZE_NUMPY))
            self.first = False
            self.count += 1

        def __exit__(self, exc_type, exc, tb):
            self.handle.write(b"]")
            self.handle.close()

    return _Stream(Path(path))


def write_npz_array_stream(path: str | Path, finalize: bool = True, compressed: bool = False):
    """Context manager that accumulates per-batch arrays and writes a .npz file.

    Each call to write() accepts a list of metadata dicts and a dict of numpy
    arrays for one batch. Per-batch arrays are queued to a background thread
    that immediately saves them to temporary .npy files, so:
      - the main loop is not blocked by disk I/O
      - memory stays flat because arrays are not held in RAM

    At close time the temp files are concatenated into the final .npz. Use
    ``compressed=True`` only when disk space is scarce; it is much slower.
    Activations are stored as float16 to cut file size and write bandwidth;
    the read path upcasts back to float32.

    If ``finalize=False``, the chunk directory is left in place (named like the
    output without ``.npz``) instead of being concatenated and deleted. This
    matches manual consolidation notebooks that glob ``l<N>_census_raw`` dirs.
    """
    import numpy as np
    import orjson
    import shutil
    import threading
    import queue

    class _Stream:
        def __init__(self, out: Path, compressed: bool = False):
            self.out = Path(out)
            self.finalize = finalize
            self.compressed = compressed
            self.metadata: list[dict[str, Any]] = []
            # F1: when finalizing, scratch on a fast LOCAL path (tmpfs if
            # available) so per-batch writes, finalize read-back, and rmtree
            # don't hit the network filesystem the output often lives on (RunPod
            # /workspace is a network mount). Only the final .npz ships to
            # self.out.parent. Non-finalize mode keeps the chunk dir next to the
            # output (it IS the canonical output in that mode -- must persist).
            if not self.finalize:
                self.temp_dir = self.out.with_suffix("")
            else:
                self.temp_dir = self._resolve_temp_dir()
            self.temp_files: dict[str, list[Path]] = {}
            self.batch_index = 0
            # F2: chunked flush -- accumulate this many batches per key before
            # writing one .npy, cutting the temp-file count ~Kx. Env-tunable.
            self._chunk_batches = max(1, int(os.environ.get("ATLAS_CHUNK_BATCHES", "32")))
            self._accum: dict[str, list[Any]] = {}
            self._accum_count = 0
            self._chunk_index = 0
            self._write_queue: queue.Queue[tuple[int, dict[str, Any]] | None] = queue.Queue(maxsize=4)
            self._writer_exc: Exception | None = None
            self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
            # Crash-safe metadata journal: one orjson line per row, flushed every
            # batch. _flush_metadata() only runs at __exit__, so a SIGKILL
            # mid-loop (the OOM-sweep's whole failure mode) loses the in-RAM
            # self.metadata and the _metadata.npy never gets written -- which is
            # exactly how 6 layers lost their metadata in the Half-1 run. This
            # journal is the recovery tool's fallback: every captured batch's
            # prompt/seq_len/max_token_idx survives a hard kill.
            self._meta_journal = None
            self._meta_journal_path: Path | None = None

        def _resolve_temp_dir(self) -> Path:
            import tempfile

            base = os.environ.get("ATLAS_TMP_DIR")
            if not base:
                # Default: scratch on the SAME volume as the output (out.parent).
                # F2's chunked flush cut the temp-file count from ~6726/layer to
                # ~30/layer, so writing scratch next to the output is no longer the
                # 640s/layer finalize killer it was pre-F2 -- large sequential
                # chunk writes and same-volume renames are fine on the network FS.
                # Keeping scratch OFF /dev/shm and /tmp (the container disk) is
                # deliberate: RunPod container disks are small (75GB here) and
                # fill fast, while the 800GB network volume is what the user set up
                # to hold exactly this kind of write. Set ATLAS_TMP_DIR=/dev/shm
                # explicitly if you want the tmpfs speed boost on a small run that
                # fits in RAM.
                base = str(self.out.parent)
            return Path(tempfile.mkdtemp(prefix=f"{self.out.stem}_", dir=base))

        def _writer_loop(self):
            try:
                while True:
                    item = self._write_queue.get()
                    if item is None:
                        self._flush_chunk()
                        break
                    _batch_index, arrays = item
                    for k, arr in arrays.items():
                        self._accum.setdefault(k, []).append(arr)
                    self._accum_count += 1
                    if self._accum_count >= self._chunk_batches:
                        self._flush_chunk()
            except Exception as exc:
                self._writer_exc = exc

        def _flush_chunk(self):
            import numpy as np

            if self._accum_count == 0:
                return
            chunk_idx = self._chunk_index
            for k, arrs in self._accum.items():
                stacked = np.concatenate(arrs, axis=0) if len(arrs) > 1 else arrs[0]
                final = self.temp_dir / f"{k}_chunk{chunk_idx:06d}.npy"
                writing = self.temp_dir / f"{k}_chunk{chunk_idx:06d}.tmp.npy"
                np.save(writing, stacked)
                writing.rename(final)
                self.temp_files.setdefault(k, []).append(final)
            self._accum = {}
            self._accum_count = 0
            self._chunk_index += 1

        def _flush_metadata(self) -> None:
            """Persist metadata to a temp file so recovery scripts can rebuild .npz."""
            import tempfile

            metadata_json = orjson.dumps(self.metadata, default=str)
            fd, tmp_path = tempfile.mkstemp(suffix=".npy", dir=self.temp_dir)
            with os.fdopen(fd, "wb") as f:
                np.save(f, np.frombuffer(metadata_json, dtype=np.uint8))
            final_path = self.temp_dir / "_metadata.npy"
            Path(tmp_path).rename(final_path)
            self.temp_files.setdefault("_metadata", []).append(final_path)

        def __enter__(self):
            self.out.parent.mkdir(parents=True, exist_ok=True)
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            # Open the crash-safe metadata journal in binary-append mode. One
            # orjson line per row, fsync'd per batch in write(). Sits in temp_dir
            # (the fast local path), so it's both crash-safe AND cheap.
            self._meta_journal_path = self.temp_dir / "_metadata.jsonl"
            self._meta_journal = open(self._meta_journal_path, "ab")
            self._writer_thread.start()
            return self

        def write(
            self,
            metadata: list[dict[str, Any]],
            arrays: dict[str, Any],
        ) -> None:
            self.metadata.extend(metadata)
            if self._meta_journal is not None:
                # Append one jsonl line per row and fsync -- a hard kill between
                # batches can drop the OS buffer for the in-flight batch, but
                # every completed batch is on disk. orjson returns bytes, so
                # write straight into the binary handle (no decode round-trip).
                fh = self._meta_journal
                for row in metadata:
                    fh.write(orjson.dumps(row, default=str))
                    fh.write(b"\n")
                fh.flush()
                os.fsync(fh.fileno())
            self._write_queue.put((self.batch_index, arrays), block=True)
            self.batch_index += 1
            if self._writer_exc:
                raise self._writer_exc

        def __exit__(self, exc_type, exc, tb):
            import time

            self._write_queue.put(None)
            self._writer_thread.join()
            # Close the crash-safe journal first -- even on the exception path
            # (e.g. OOM caught upstream), the per-batch rows already on disk
            # stay recoverable.
            if self._meta_journal is not None:
                try:
                    self._meta_journal.close()
                except Exception:
                    pass
                self._meta_journal = None
            # Persist metadata so recovery scripts can rebuild the .npz even if
            # the main process is killed before final concatenation.
            if self.temp_dir.exists():
                self._flush_metadata()
            if exc_type is not None:
                return
            if self._writer_exc:
                raise self._writer_exc
            if not self.finalize:
                print(f"  persisted chunks for {self.out.name} in {self.temp_dir}")
                return
            t0 = time.time()
            final_arrays: dict[str, Any] = {}
            metadata_path = None
            for k, paths in self.temp_files.items():
                if k == "_metadata":
                    metadata_path = paths[0]
                    continue

                ordered = sorted(paths)
                if not ordered:
                    continue

                # Per-token arrays are ragged object arrays and require pickle.
                # Detect that path and build a concatenated object array.
                first = np.load(ordered[0], allow_pickle=True)
                is_object = first.dtype == object
                total = sum(int(np.load(p, allow_pickle=is_object).shape[0]) for p in ordered)

                if is_object:
                    parts = [np.load(p, allow_pickle=True) for p in ordered]
                    stacked = np.concatenate(parts, axis=0)
                    del parts
                else:
                    # Two-pass preallocation for dense numeric arrays.
                    out_shape = (total,) + first.shape[1:]
                    stacked = np.empty(out_shape, dtype=first.dtype)
                    del first

                    cursor = 0
                    for i in range(0, len(ordered), 64):
                        chunk = ordered[i : i + 64]
                        try:
                            parts = [np.load(p) for p in chunk]
                        except EOFError as exc:
                            bad = [str(p) for p in chunk if p.stat().st_size == 0]
                            raise RuntimeError(
                                f"Corrupt/empty temp file(s) for key {k!r}: {bad or chunk}"
                            ) from exc
                        for part in parts:
                            n = part.shape[0]
                            stacked[cursor : cursor + n] = part
                            cursor += n
                        del parts

                if not is_object and stacked.dtype == np.float32:
                    stacked = stacked.astype(np.float16)
                final_arrays[k] = stacked

            if metadata_path is None:
                raise RuntimeError("missing _metadata temp file")
            final_arrays["_metadata"] = np.load(metadata_path)

            # Temp + rename: a killed process must never leave a truncated
            # l<N>_census_raw.npz that --skip-census would treat as complete.
            # np.savez appends ".npz" to a bare path, so hand it an open file
            # handle on a same-directory temp file and os.replace at the end.
            tmp_out = self.out.with_name(self.out.name + ".partial")
            with open(tmp_out, "wb") as fh:
                if self.compressed:
                    np.savez_compressed(fh, **final_arrays)
                else:
                    np.savez(fh, **final_arrays)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_out, self.out)
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            print(f"  wrote {self.out.name} in {time.time() - t0:.1f}s (compressed={self.compressed})")

    return _Stream(Path(path), compressed=compressed)


def census_npz_status(path: str | Path, expected_rows: int | None = None) -> tuple[bool, str]:
    """Decide whether an existing census .npz is complete enough to skip re-extraction.

    Existence alone is not completeness: a run killed mid-finalize used to leave
    a truncated file. We require the zip to open, carry ``_metadata`` plus at
    least one array, have every array's leading dim equal to the metadata row
    count, and (when known) match the corpus size.
    """
    import numpy as np
    import orjson

    p = Path(path)
    if not p.exists():
        return False, "missing"
    if p.with_name(p.name + ".partial").exists():
        return False, "partial file present (finalize was interrupted)"
    try:
        z = np.load(p, allow_pickle=False)
        files = list(z.files)
        if "_metadata" not in files:
            return False, "no _metadata array"
        meta = orjson.loads(z["_metadata"].tobytes())
        n = len(meta)
        arrays = [k for k in files if k != "_metadata"]
        if not arrays:
            return False, "no activation arrays"
        for k in arrays:
            shape = z[k].shape
            if not shape or shape[0] != n:
                return False, f"array {k!r} has {shape[0] if shape else 'no'} rows, metadata has {n}"
        if expected_rows is not None and n != expected_rows:
            return False, f"{n} rows captured, corpus has {expected_rows}"
    except Exception as exc:  # zipfile.BadZipFile, EOFError, ValueError...
        return False, f"unreadable ({exc.__class__.__name__}: {exc})"
    return True, f"complete ({n} rows, {len(arrays)} arrays)"


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"
