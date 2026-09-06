from __future__ import annotations

import argparse
import os
from pathlib import Path

from qwip_atlas.config import AtlasRunConfig, ComplianceBehaviourRunConfig, CorpusSpec, ModelSpec
from qwip_atlas.layers import parse_layer_spec


def _extract_local(args: argparse.Namespace) -> None:
    cfg = AtlasRunConfig(
        model=ModelSpec(
            model_id=args.model,
            revision=args.revision,
            trust_remote_code=args.trust_remote,
            dtype=args.dtype,
            device_map=args.device_map,
            max_length=args.max_length,
            attn_implementation=args.attn_implementation,
            chat_template=args.chat_template,
        ),
        corpus=CorpusSpec(
            path=Path(args.corpus),
            prompt_key=args.prompt_key,
            category_key=args.category_key,
            bucket_key=args.bucket_key,
        ),
        layers=parse_layer_spec(args.layers),
        outdir=Path(args.outdir),
        batch_size=args.batch_size,
        components=set(args.components.split(",")) if args.components else AtlasRunConfig.__dataclass_fields__["components"].default_factory(),
        truncate_to_deepest_layer=not args.no_truncate,
        timing_every=args.timing_every,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_tags=args.wandb_tags.split(",") if args.wandb_tags else None,
    )
    token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    from qwip_atlas.extractors import run_local_census

    run_local_census(cfg, hf_token=token)


def _model_spec_from_args(args: argparse.Namespace) -> ModelSpec:
    return ModelSpec(
        model_id=args.model,
        revision=args.revision,
        trust_remote_code=args.trust_remote,
        dtype=args.dtype,
        device_map=args.device_map,
        max_length=args.max_length,
        attn_implementation=args.attn_implementation,
        chat_template=args.chat_template,
    )


def _compliance_behaviour_local(args: argparse.Namespace) -> None:
    components = (
        set(args.components.split(","))
        if args.components
        else ComplianceBehaviourRunConfig.__dataclass_fields__["components"].default_factory()
    )
    cfg = ComplianceBehaviourRunConfig(
        model=_model_spec_from_args(args),
        positive_corpus=CorpusSpec(
            path=Path(args.positive),
            prompt_key=args.positive_prompt_key or args.prompt_key,
            category_key=args.category_key,
            bucket_key=args.bucket_key,
        ),
        negative_corpus=CorpusSpec(
            path=Path(args.negative),
            prompt_key=args.negative_prompt_key or args.prompt_key,
            category_key=args.category_key,
            bucket_key=args.bucket_key,
        ),
        layers=parse_layer_spec(args.layers),
        output=Path(args.output),
        batch_size=args.batch_size,
        components=components,
        positive_label=args.positive_label,
        negative_label=args.negative_label,
        truncate_to_deepest_layer=not args.no_truncate,
    )
    token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    from qwip_atlas.extractors import run_compliance_behaviour

    run_compliance_behaviour(cfg, hf_token=token)


def main() -> None:
    parser = argparse.ArgumentParser(prog="qwip-atlas")
    sub = parser.add_subparsers(dest="cmd", required=True)

    extract = sub.add_parser("extract-local", help="Run local HF activation census extraction")
    extract.add_argument("--model", required=True, help="HF model id or local model path")
    extract.add_argument("--revision", default=None)
    extract.add_argument("--corpus", required=True, help="Local JSONL corpus")
    extract.add_argument("--layers", required=True, help="Layer spec, e.g. 0,4,10-12")
    extract.add_argument("--outdir", required=True)
    extract.add_argument("--batch-size", type=int, default=8)
    extract.add_argument("--max-length", type=int, default=512)
    extract.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"])
    extract.add_argument("--device-map", default="auto", help="Transformers device_map; pass '' to disable")
    extract.add_argument("--attn-implementation", default=None, choices=["eager", "sdpa", "flash_attention_2"], help="Transformers attention backend")
    extract.add_argument("--trust-remote", action="store_true", help="Enable trust_remote_code for the model")
    extract.add_argument("--chat-template", action="store_true",
                         help="Wrap each prompt in the model's chat template (user turn + assistant "
                              "generation prompt) before tokenizing. For *-Instruct models this "
                              "captures activations in-distribution.")
    extract.add_argument("--components", default=None, help="Comma-separated subset: mlp,gate,up,attn,heads,q,k,v")
    extract.add_argument("--prompt-key", default="prompt")
    extract.add_argument("--category-key", default="category")
    extract.add_argument("--bucket-key", default="bucket")
    extract.add_argument("--hf-token", default=None)
    extract.add_argument("--no-truncate", action="store_true")
    extract.add_argument(
        "--timing-every",
        type=int,
        default=250,
        help="Print extraction timing every N batches; use 0 to disable",
    )
    extract.add_argument("--wandb-project", default=None,
                         help="W&B project for per-batch memory/timing telemetry + OOM capture. "
                              "Entity defaults to ricks-holmberg-juiceb0xc0de.")
    extract.add_argument("--wandb-entity", default="ricks-holmberg-juiceb0xc0de")
    extract.add_argument("--wandb-run-name", default=None)
    extract.add_argument("--wandb-tags", default=None, help="comma-separated W&B tags")
    extract.add_argument("--no-auth-prompt", action="store_true",
                         help="Skip the interactive HF/W&B key prompt; fall back to env vars only.")
    extract.add_argument("--persist-login", action="store_true",
                         help="Cache the pasted keys via huggingface_hub/wandb login so the pod "
                              "stays logged in for later commands and stages.")
    extract.set_defaults(func=_extract_local)

    compliance_behaviour = sub.add_parser(
        "compliance-behaviour-local",
        aliases=["compliance_behaviour-local"],
        help="Run local binary compliance-behaviour extraction",
    )
    compliance_behaviour.add_argument("--model", required=True, help="HF model id or local model path")
    compliance_behaviour.add_argument("--revision", default=None)
    compliance_behaviour.add_argument("--positive", required=True, help="Positive-axis JSONL corpus")
    compliance_behaviour.add_argument("--negative", required=True, help="Negative-axis JSONL corpus")
    compliance_behaviour.add_argument("--layers", required=True, help="Layer spec, e.g. 0,4,10-12")
    compliance_behaviour.add_argument("--output", required=True, help="Output compliance_behaviour_scores.json path")
    compliance_behaviour.add_argument("--batch-size", type=int, default=8)
    compliance_behaviour.add_argument("--max-length", type=int, default=512)
    compliance_behaviour.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"])
    compliance_behaviour.add_argument("--device-map", default="auto", help="Transformers device_map; pass '' to disable")
    compliance_behaviour.add_argument("--attn-implementation", default=None, choices=["eager", "sdpa", "flash_attention_2"])
    compliance_behaviour.add_argument("--trust-remote", action="store_true")
    compliance_behaviour.add_argument("--chat-template", action="store_true",
                                      help="Wrap each prompt in the model's chat template before "
                                           "tokenizing (user turn + assistant generation prompt).")
    compliance_behaviour.add_argument("--components", default=None, help="Comma-separated subset: mlp,gate,up,attn,heads,q,k,v")
    compliance_behaviour.add_argument("--prompt-key", default="prompt")
    compliance_behaviour.add_argument("--positive-prompt-key", default=None)
    compliance_behaviour.add_argument("--negative-prompt-key", default=None)
    compliance_behaviour.add_argument("--category-key", default="category")
    compliance_behaviour.add_argument("--bucket-key", default="bucket")
    compliance_behaviour.add_argument("--positive-label", default="positive")
    compliance_behaviour.add_argument("--negative-label", default="negative")
    compliance_behaviour.add_argument("--hf-token", default=None)
    compliance_behaviour.add_argument("--no-truncate", action="store_true")
    compliance_behaviour.add_argument("--no-auth-prompt", action="store_true",
                                      help="Skip the interactive HF/W&B key prompt; fall back to env vars only.")
    compliance_behaviour.add_argument("--persist-login", action="store_true",
                                      help="Cache the pasted keys via huggingface_hub/wandb login.")
    compliance_behaviour.set_defaults(func=_compliance_behaviour_local)

    args = parser.parse_args()
    if getattr(args, "device_map", None) == "":
        args.device_map = None

    # Resolve HF + W&B credentials before anything touches the network / model
    # download. Hidden-input prompt only for keys not already in env / cached
    # login, so split runs in the same shell/pod are silent after the first.
    from qwip_atlas.auth import ensure_credentials
    hf_token, _ = ensure_credentials(
        hf_token=getattr(args, "hf_token", None),
        prompt=not getattr(args, "no_auth_prompt", False),
        persist_login=getattr(args, "persist_login", False),
    )
    if hf_token and not getattr(args, "hf_token", None):
        args.hf_token = hf_token

    args.func(args)


if __name__ == "__main__":
    main()
