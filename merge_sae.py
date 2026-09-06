#!/usr/bin/env python3
"""Fold SAE score arrays into an atlas SQLite mirror."""
from __future__ import annotations

import argparse
import glob
import os
import re
import sqlite3
from pathlib import Path

import numpy as np

ATLAS = Path("atlas")

DDL = """
CREATE TABLE IF NOT EXISTS sae_features (
    layer           INTEGER,
    variant         TEXT,
    feature_idx     INTEGER,
    topic_fstat     REAL,
    compliance_behaviour_fstat   REAL,
    compliance_behaviour_delta   REAL,
    mean_corp       REAL,
    mean_auth       REAL,
    activation_rate REAL,
    corp_leaning    INTEGER,
    PRIMARY KEY (layer, variant, feature_idx)
);
CREATE INDEX IF NOT EXISTS idx_sae_compliance_behaviour ON sae_features(variant, compliance_behaviour_fstat DESC);
CREATE INDEX IF NOT EXISTS idx_sae_topic ON sae_features(variant, topic_fstat);
"""

GOLD_SQL = """
SELECT layer, feature_idx,
       ROUND(compliance_behaviour_fstat,1) AS compliance_behaviour_F,
       ROUND(topic_fstat,2) AS topic_F,
       ROUND(compliance_behaviour_delta,3) AS delta,
       ROUND(activation_rate,4) AS act_rate
FROM sae_features
WHERE variant = ?
  AND compliance_behaviour_fstat IS NOT NULL
  AND compliance_behaviour_fstat > ?
  AND topic_fstat < ?
ORDER BY compliance_behaviour_fstat DESC
LIMIT ?;
"""


def _load_npz(path: str) -> np.lib.npyio.NpzFile:
    # Security: these score files only need numeric arrays. Keep pickle disabled.
    return np.load(path, allow_pickle=False)


def merge(atlas: Path, variant: str, min_act: float, sae_dir: Path | None = None) -> int:
    db = atlas / "atlas.sqlite"
    if not db.exists():
        raise SystemExit(f"atlas sqlite not found: {db} (run atlas build first)")
    search_dir = sae_dir or atlas
    files = sorted(
        glob.glob(os.path.join(search_dir, f"sae_l*_{variant}.npz")),
        key=lambda p: int(re.search(r"sae_l(\d+)_", p).group(1)),
    )
    if not files:
        raise SystemExit(
            f"no sae_l*_{variant}.npz files in {search_dir}"
        )

    con = sqlite3.connect(db)
    con.executescript(DDL)
    con.execute("DELETE FROM sae_features WHERE variant = ?", (variant,))

    total = 0
    for f in files:
        layer = int(re.search(r"sae_l(\d+)_", f).group(1))
        z = _load_npz(f)
        topic = z["topic_fstat"]
        act = z["activation_rate"]
        has_behavior = "compliance_behaviour_fstat" in z.files
        compliance_behaviour = (
            z["compliance_behaviour_fstat"]
            if has_behavior
            else np.full_like(topic, np.nan)
        )
        delta = (
            z["compliance_behaviour_delta"]
            if has_behavior
            else np.full_like(topic, np.nan)
        )
        mcorp = z["mean_corp"] if has_behavior else np.full_like(topic, np.nan)
        mauth = z["mean_auth"] if has_behavior else np.full_like(topic, np.nan)

        keep = act > min_act
        idx = np.nonzero(keep)[0]
        rows = [
            (
                int(layer),
                variant,
                int(i),
                float(topic[i]),
                None if not has_behavior else float(compliance_behaviour[i]),
                None if not has_behavior else float(delta[i]),
                None if not has_behavior else float(mcorp[i]),
                None if not has_behavior else float(mauth[i]),
                float(act[i]),
                None if not has_behavior else int(delta[i] > 0),
            )
            for i in idx
        ]
        con.executemany(
            "INSERT OR REPLACE INTO sae_features VALUES (?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        total += len(rows)
        print(f"  layer {layer:>2}: {len(rows):>6} features kept (of {len(topic)})")

    con.commit()
    print(f"merged {total} sae_features rows (variant={variant}) -> {db}")
    return total


def show_gold(
    atlas: Path,
    variant: str,
    min_compliance_behaviour: float,
    max_topic: float,
    n: int,
):
    con = sqlite3.connect(atlas / "atlas.sqlite")
    rows = con.execute(
        GOLD_SQL, (variant, min_compliance_behaviour, max_topic, n)
    ).fetchall()
    if not rows:
        print("no surgical-gold features matched (need a compliance_behaviour pass + tune thresholds)")
        return
    print(
        f"\n  SURGICAL GOLD - high compliance_behaviour_F (>{min_compliance_behaviour}) "
        f"+ low topic_F (<{max_topic})"
    )
    print(
        f"  {'layer':>5} {'feat':>6} {'compliance_behaviour_F':>10} "
        f"{'topic_F':>8} {'delta':>7} {'act':>7}"
    )
    for layer, feature_idx, behavior_f, topic_f, delta, act in rows:
        lean = "corp" if (delta or 0) > 0 else "auth"
        print(
            f"  {layer:>5} {feature_idx:>6} {behavior_f:>10} {topic_f:>8} "
            f"{delta:>7} {act:>7}   ({lean})"
        )
    print("\n  -> steer/ablate with the SAE decoder vector W_dec[:, feat] at that layer.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="l0_50", help="SAE variant label, e.g. l0_50, 32x, 64x")
    parser.add_argument("--atlas", default=str(ATLAS))
    parser.add_argument("--sae-dir", default=None, help="directory containing sae_l*_<variant>.npz files")
    parser.add_argument(
        "--min-activation",
        type=float,
        default=0.0,
        help="drop features that fire on <= this fraction of tokens",
    )
    parser.add_argument("--gold", action="store_true", help="just print surgical-gold targets")
    parser.add_argument("--min-compliance_behaviour", type=float, default=10.0)
    parser.add_argument("--max-topic", type=float, default=3.0)
    parser.add_argument("--n", type=int, default=25)
    args = parser.parse_args()
    atlas = Path(args.atlas).expanduser()
    sae_dir = Path(args.sae_dir).expanduser() if args.sae_dir else None

    if args.gold:
        show_gold(atlas, args.variant, args.min_compliance_behaviour, args.max_topic, args.n)
    else:
        merge(atlas, args.variant, args.min_activation, sae_dir)
        show_gold(atlas, args.variant, args.min_compliance_behaviour, args.max_topic, args.n)


if __name__ == "__main__":
    main()
