"""
atlas_store.py
--------------
Shared utilities for the master atlas: paths, manifest schema, loaders,
SQLite mirror, corpus hashing. Imported by build_atlas.py and query_atlas.py.

Atlas layout (rooted at any directory; default ./atlas):

  atlas/
    manifest.json            top-level index (layers / components / corpora)
    model_meta.json          model geometry
    atlas.sqlite             pandas-friendly mirror for cross-cutting queries
    layers/
      <N>/
        meta.json            per-layer capture meta (corpus_hash, batch, etc.)
        census/raw.json      copy of l<N>_census_raw.json
        components/<C>/      taxonomy.json, separation_scores.npy, coactivation.json,
                             code_analysis.json, summary.json
        per_head/<C>.json    8-row per-head table (heads, q, k, v)
        sub_zero/
          scores.json          per-layer scalar scores from Sub-Zero report
          projections.json     per-projection SV lists from Sub-Zero report
    cross_layer/
      taxonomy_by_layer.json
      fstat_top_by_layer.json
      code_entanglement_by_layer.json
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


# Default atlas root if not specified by caller. Keep this repo-local so public
# usage never depends on Rick's machine layout.
DEFAULT_ATLAS_DIR = Path("atlas")

# Components we know how to ingest from the analyzer's outputs.
KNOWN_COMPONENTS    = ["mlp", "gate", "up", "attn", "heads", "q", "k", "v"]
PER_HEAD_COMPONENTS = ["heads", "q", "k", "v"]


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def atlas_paths(root: Path) -> dict[str, Path]:
    """Return the canonical sub-paths inside an atlas root."""
    root = Path(root)
    return {
        "root":         root,
        "manifest":     root / "manifest.json",
        "model_meta":   root / "model_meta.json",
        "sqlite":       root / "atlas.sqlite",
        "layers":       root / "layers",
        "cross_layer":  root / "cross_layer",
    }


def layer_paths(root: Path, layer: int) -> dict[str, Path]:
    """Return the canonical sub-paths for a single layer subtree."""
    L = atlas_paths(root)["layers"] / str(layer)
    return {
        "root":             L,
        "meta":             L / "meta.json",
        "census_raw":       L / "census" / "raw.json",
        "components":       L / "components",
        "per_head":         L / "per_head",
        "sub_zero":         L / "sub_zero",
        "subzero_scores":   L / "sub_zero" / "scores.json",
        "subzero_projs":    L / "sub_zero" / "projections.json",
    }


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Return sha256:<hex> of file contents; bullet-proof corpus identifier."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return f"sha256:{h.hexdigest()}"


# ---------------------------------------------------------------------------
# Manifest schema (small dataclass we serialize as JSON)
# ---------------------------------------------------------------------------

@dataclass
class Manifest:
    model_id: str
    model_meta: dict[str, Any]
    layers: list[int]
    components_per_layer: dict[str, list[str]]
    subzero_layers: list[int]
    corpus: dict[str, Any]
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # canonicalize JSON-unfriendly keys: ints->str in dict-with-int-keys
        d["components_per_layer"] = {str(k): v for k, v in self.components_per_layer.items()}
        return d


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Read/write helpers
# ---------------------------------------------------------------------------

def read_json(path: Path) -> Any:
    import orjson                              # ~2-3x faster on the multi-GB census files
    with open(path, "rb") as f:
        return orjson.loads(f.read())


def write_json(path: Path, obj: Any) -> None:
    import orjson
    path.parent.mkdir(parents=True, exist_ok=True)
    # orjson.dumps -> bytes; OPT_INDENT_2 keeps diffs readable, OPT_SERIALIZE_NUMPY handles np scalars
    opt = orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY
    path.write_bytes(orjson.dumps(obj, option=opt))


def load_manifest(root: Path) -> Manifest | None:
    p = atlas_paths(root)["manifest"]
    if not p.exists():
        return None
    d = read_json(p)
    return Manifest(
        model_id=d["model_id"],
        model_meta=d["model_meta"],
        layers=[int(x) for x in d["layers"]],
        components_per_layer={int(k): v for k, v in d["components_per_layer"].items()},
        subzero_layers=[int(x) for x in d["subzero_layers"]],
        corpus=d.get("corpus", {}),
        updated_at=d["updated_at"],
    )


def save_manifest(root: Path, m: Manifest) -> None:
    write_json(atlas_paths(root)["manifest"], m.to_dict())


# ---------------------------------------------------------------------------
# Iteration helpers
# ---------------------------------------------------------------------------

def iter_layers(root: Path) -> Iterable[int]:
    layers_dir = atlas_paths(root)["layers"]
    if not layers_dir.exists():
        return
    for child in sorted(layers_dir.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else -1):
        if child.is_dir() and child.name.isdigit():
            yield int(child.name)


def iter_components(root: Path, layer: int) -> Iterable[str]:
    comp_dir = layer_paths(root, layer)["components"]
    if not comp_dir.exists():
        return
    for c in KNOWN_COMPONENTS:
        if (comp_dir / c).exists():
            yield c


def load_component_summary(root: Path, layer: int, component: str) -> dict[str, Any] | None:
    p = layer_paths(root, layer)["components"] / component / "summary.json"
    return read_json(p) if p.exists() else None


# ---------------------------------------------------------------------------
# SQLite mirror — narrow, denormalized, optimized for cross-cuts
# ---------------------------------------------------------------------------

SQL_SCHEMA = """
CREATE TABLE IF NOT EXISTS layers (
    layer_id INTEGER PRIMARY KEY,
    model_id TEXT,
    n_prompts INTEGER,
    corpus_hash TEXT,
    captured_at TEXT,
    has_census INTEGER DEFAULT 0,
    has_subzero INTEGER DEFAULT 0,
    is_sacred INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS features (
    layer_id INTEGER,
    component TEXT,
    feature_idx INTEGER,
    taxonomy_class TEXT,
    activation_rate REAL,
    mean_act REAL,
    std_act REAL,
    fstat REAL,
    null_floor REAL,
    q_value REAL,
    survivor INTEGER,
    PRIMARY KEY (layer_id, component, feature_idx)
);
CREATE INDEX IF NOT EXISTS idx_features_layer_comp ON features(layer_id, component);
CREATE INDEX IF NOT EXISTS idx_features_class       ON features(taxonomy_class);

CREATE TABLE IF NOT EXISTS per_head (
    layer_id INTEGER,
    component TEXT,
    head_idx INTEGER,
    dims INTEGER,
    n_specific INTEGER,
    fstat_best REAL,
    fstat_mean REAL,
    top_code_dim INTEGER,
    top_code_spec REAL,
    PRIMARY KEY (layer_id, component, head_idx)
);

CREATE TABLE IF NOT EXISTS subzero_layer (
    layer_id INTEGER PRIMARY KEY,
    classifier_accuracy REAL,
    corp_refusal_angle_deg REAL,
    sv_total INTEGER,
    compliance_behaviour_sv INTEGER,
    compliance_behaviour_pct REAL
);

CREATE TABLE IF NOT EXISTS subzero_svs (
    layer_id INTEGER,
    projection TEXT,
    sv_index INTEGER,
    classifier_score REAL,
    wanda_score REAL,
    dark_variance REAL,
    target_scale REAL,
    PRIMARY KEY (layer_id, projection, sv_index)
);
CREATE INDEX IF NOT EXISTS idx_subzero_svs_layer ON subzero_svs(layer_id);

-- Per-DAS-axis capability entanglement: how much freezing each compliance
-- direction would damage each capability domain (code/math/factual/...).
-- One row per (layer, projection, das_axis, domain).
CREATE TABLE IF NOT EXISTS subzero_capability (
    layer_id INTEGER,
    projection TEXT,
    das_axis INTEGER,
    domain TEXT,
    damage REAL,
    damage_max REAL,
    fence_passed INTEGER,
    frozen INTEGER,
    explained REAL,
    PRIMARY KEY (layer_id, projection, das_axis, domain)
);
CREATE INDEX IF NOT EXISTS idx_subzero_cap_layer ON subzero_capability(layer_id);
CREATE INDEX IF NOT EXISTS idx_subzero_cap_domain ON subzero_capability(domain);

CREATE TABLE IF NOT EXISTS coactivation (
    layer_id INTEGER,
    component TEXT,
    feature_a INTEGER,
    feature_b INTEGER,
    correlation REAL,
    dominant_bucket TEXT
);
CREATE INDEX IF NOT EXISTS idx_coact_layer_comp ON coactivation(layer_id, component);

CREATE TABLE IF NOT EXISTS code_analysis (
    layer_id INTEGER,
    component TEXT,
    feature_idx INTEGER,
    role TEXT,            -- 'entangled' or 'selective'
    PRIMARY KEY (layer_id, component, feature_idx, role)
);

CREATE TABLE IF NOT EXISTS compliance_behaviour_features (
    layer_id INTEGER,
    component TEXT,
    feature_idx INTEGER,
    fstat REAL,
    delta REAL,
    mean_corp REAL,
    mean_auth REAL,
    PRIMARY KEY (layer_id, component, feature_idx)
);
CREATE INDEX IF NOT EXISTS idx_compliance_behaviour_layer_comp ON compliance_behaviour_features(layer_id, component);
CREATE INDEX IF NOT EXISTS idx_compliance_behaviour_fstat      ON compliance_behaviour_features(fstat DESC);

CREATE TABLE IF NOT EXISTS compliance_behaviour_per_head (
    layer_id INTEGER,
    component TEXT,
    head_idx INTEGER,
    head_dim INTEGER,
    fstat_best REAL,
    fstat_mean REAL,
    delta_at_top REAL,
    top_dim INTEGER,
    corp_leaning INTEGER,
    PRIMARY KEY (layer_id, component, head_idx)
);

CREATE TABLE IF NOT EXISTS ov_circuits (
    layer_id INTEGER,
    head_idx INTEGER,
    kv_head INTEGER,
    ov_top_singular_val REAL,
    ov_total_energy REAL,
    ov_spectral_conc REAL,
    ov_eff_rank REAL,
    qk_top_singular_val REAL,
    qk_total_energy REAL,
    qk_spectral_conc REAL,
    qk_eff_rank REAL,
    fc_top_singular_val REAL,
    fc_total_energy REAL,
    fc_spectral_conc REAL,
    fc_eff_rank REAL,
    induction_score REAL,
    compliance_score REAL,
    layer_comp_strength REAL,
    ov_top3_sv TEXT,        -- JSON array [s0, s1, s2]
    qk_top3_sv TEXT,        -- JSON array [s0, s1, s2]
    fc_top3_sv TEXT,        -- JSON array [s0, s1, s2]
    PRIMARY KEY (layer_id, head_idx)
);
CREATE INDEX IF NOT EXISTS idx_ov_compliance ON ov_circuits(compliance_score DESC);
CREATE INDEX IF NOT EXISTS idx_ov_ov_spectral   ON ov_circuits(ov_spectral_conc DESC);
CREATE INDEX IF NOT EXISTS idx_ov_qk_spectral   ON ov_circuits(qk_spectral_conc DESC);
CREATE INDEX IF NOT EXISTS idx_ov_fc_spectral   ON ov_circuits(fc_spectral_conc DESC);
CREATE INDEX IF NOT EXISTS idx_ov_induction     ON ov_circuits(induction_score DESC);

CREATE TABLE IF NOT EXISTS logit_lens (
    layer_id INTEGER,
    component TEXT,
    feature_idx INTEGER,
    fstat REAL,
    promoted_tokens TEXT,      -- JSON array of {token, token_id, logit}
    suppressed_tokens TEXT,    -- JSON array of {token, token_id, logit}
    PRIMARY KEY (layer_id, component, feature_idx)
);
CREATE INDEX IF NOT EXISTS idx_logit_lens_layer ON logit_lens(layer_id, component);
CREATE INDEX IF NOT EXISTS idx_logit_lens_fstat ON logit_lens(fstat DESC);

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
CREATE INDEX IF NOT EXISTS idx_sae_topic   ON sae_features(variant, topic_fstat);
"""


def open_sqlite(root: Path) -> sqlite3.Connection:
    db_path = atlas_paths(root)["sqlite"]
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SQL_SCHEMA)
    return conn
