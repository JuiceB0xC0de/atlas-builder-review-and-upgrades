from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    """Runtime model settings, deliberately outside the extractor code."""

    model_id: str
    revision: str | None = None
    trust_remote_code: bool = False
    dtype: str = "bfloat16"
    device_map: str | None = "auto"
    max_length: int = 512
    attn_implementation: str | None = None
    # Wrap each prompt in the model's chat template (user turn + assistant
    # generation prompt) before tokenizing. For *-Instruct models this captures
    # activations in-distribution; see qwip_atlas.chat_format.
    chat_template: bool = False


@dataclass(frozen=True)
class CorpusSpec:
    """Local JSONL corpus settings."""

    path: Path
    prompt_key: str = "prompt"
    category_key: str = "category"
    bucket_key: str = "bucket"


@dataclass(frozen=True)
class AtlasRunConfig:
    """Config for a local activation-census run."""

    model: ModelSpec
    corpus: CorpusSpec
    layers: list[int]
    outdir: Path
    batch_size: int = 8
    components: set[str] = field(default_factory=lambda: {
        "mlp",
        "gate",
        "up",
        "attn",
        "heads",
        "q",
        "k",
        "v",
    })
    truncate_to_deepest_layer: bool = True
    persist_chunks: bool = False
    timing_every: int = 250
    store_per_token: bool = False
    track_residuals: bool = False
    compressed: bool = False
    # W&B logging for the extraction run. wandb_project=None disables wandb.
    wandb_project: str | None = None
    wandb_entity: str = "ricks-holmberg-juiceb0xc0de"
    wandb_run_name: str | None = None
    wandb_tags: list[str] | None = None
    # Pipeline group: ties this stage's run to the other stages' runs in the
    # W&B UI group pane. app.py sets it once and passes it to every stage.
    wandb_group: str | None = None


@dataclass(frozen=True)
class ComplianceBehaviourRunConfig:
    """Config for a binary behavior-axis extraction run."""

    model: ModelSpec
    positive_corpus: CorpusSpec
    negative_corpus: CorpusSpec
    layers: list[int]
    output: Path
    batch_size: int = 8
    components: set[str] = field(default_factory=lambda: {
        "mlp",
        "gate",
        "up",
        "attn",
        "heads",
        "q",
        "k",
        "v",
    })
    positive_label: str = "positive"
    negative_label: str = "negative"
    truncate_to_deepest_layer: bool = True
    # W&B logging for the compliance run. wandb_project=None disables wandb.
    wandb_project: str | None = None
    wandb_entity: str = "ricks-holmberg-juiceb0xc0de"
    wandb_run_name: str | None = None
    wandb_tags: list[str] | None = None
    # Pipeline group (see AtlasRunConfig.wandb_group) -- ties stages together.
    wandb_group: str | None = None
