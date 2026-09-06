#!/usr/bin/env python3
"""
Split the main prompt corpus into authentic / corporate JSONLs for
compliance-behaviour extraction.

Defaults:
    authentic = Core Technical (code/technical prompts)
    corporate = Creative Writing (narrative/creative prompts)

Override with --authentic-bucket and --corporate-bucket using exact category names.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="prompts/prompts.jsonl")
    p.add_argument("--output-authentic", default="prompts/authentic.jsonl")
    p.add_argument("--output-corporate", default="prompts/corporate.jsonl")
    p.add_argument("--authentic-bucket", default="Core Technical")
    p.add_argument("--corporate-bucket", default="Creative Writing")
    p.add_argument("--max-per-side", type=int, default=300,
                   help="cap prompts per side to keep runtime manageable")
    args = p.parse_args()

    corpus = Path(args.corpus)
    if not corpus.exists():
        raise SystemExit(f"corpus not found: {corpus}")

    authentic_rows, corporate_rows = [], []
    with open(corpus) as f:
        for line in f:
            row = json.loads(line)
            cat = row.get("category", "")
            text = row.get("prompt", "")
            if cat == args.authentic_bucket and text and len(authentic_rows) < args.max_per_side:
                authentic_rows.append({"text": text})
            elif cat == args.corporate_bucket and text and len(corporate_rows) < args.max_per_side:
                corporate_rows.append({"text": text})

    Path(args.output_authentic).write_text("\n".join(json.dumps(r) for r in authentic_rows) + "\n")
    Path(args.output_corporate).write_text("\n".join(json.dumps(r) for r in corporate_rows) + "\n")

    print(f"[split] authentic={args.authentic_bucket}: {len(authentic_rows)} rows -> {args.output_authentic}")
    print(f"[split] corporate={args.corporate_bucket}: {len(corporate_rows)} rows -> {args.output_corporate}")


if __name__ == "__main__":
    main()
