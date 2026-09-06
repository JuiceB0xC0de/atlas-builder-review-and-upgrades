"""
Interactive credential bootstrap for the atlas creator.

Mirrors the UX of `wandb` / SFT trainers that pop up a hidden-input prompt for
your API key the first time they need it: the key is read with `getpass` (no
echo), set into the process environment, and (optionally) cached via the
official `huggingface_hub.login` / `wandb.login` so the pod stays logged in.

Why this exists: RunPod's env-var key system is fiddly and easy to fumble. With
this, you can SSH into a fresh pod and just run the atlas — it'll ask for your
HF + W&B keys inline (blocked out), exactly like an SFT run does. If a key is
already in the environment or cached login, it is NOT re-asked, so split runs
in the same shell/pod are silent after the first one.

Design rules:
  - Hidden input via getpass (never echoed, never logged).
  - Only prompts when a TTY is attached, so non-interactive container
    entrypoints (no TTY) fall back to env vars silently instead of hanging.
  - `--no-auth-prompt` / `QWIP_NO_AUTH_PROMPT=1` disables the prompt entirely.
  - An explicit `--hf-token` flag (or pre-set env) always wins — no prompt.
  - `--persist-login` / `QWIP_PERSIST_LOGIN=1` caches the keys via the official
    login calls so the pod stays logged in for later commands/stages. OFF by
    default (keeps your Mac clean — no keys written to disk there).
  - W&B prompt only fires if the `wandb` package is importable; if it isn't,
    there's nothing to log to, so we skip silently.
"""
from __future__ import annotations

import os
import sys
from getpass import getpass


def _have_tty() -> bool:
    return sys.stdin.isatty()


def _env_or_none(*names: str) -> str | None:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return None


def _hf_cached_token() -> str | None:
    """HF token from huggingface_hub's cached login, if any (optional dep)."""
    try:
        from huggingface_hub import HfApi  # local import; optional dep
    except Exception:
        return None
    try:
        return HfApi().token  # respects HF_TOKEN + cached ~/.cache/huggingface/token
    except Exception:
        return None


def _wandb_has_login() -> bool:
    """True if wandb is importable AND already logged in (env or ~/.netrc)."""
    try:
        import wandb  # noqa: F401
    except Exception:
        return False
    if os.environ.get("WANDB_API_KEY"):
        return True
    try:
        api = wandb.Api()  # raises if not logged in
        return bool(api.api_key)
    except Exception:
        return False


def _wandb_installed() -> bool:
    try:
        import wandb  # noqa: F401
        return True
    except Exception:
        return False


def _persist_hf(token: str) -> None:
    try:
        from huggingface_hub import login as hf_login
    except Exception as exc:
        print(f"[auth] huggingface_hub.login unavailable, skipping cache ({exc.__class__.__name__})", file=sys.stderr)
        return
    try:
        hf_login(token=token, add_to_git_credential=False)
        print("[auth] cached HF token via huggingface_hub.login (~/.cache/huggingface/token)", file=sys.stderr)
    except Exception as exc:
        print(f"[auth] HF login cache failed: {exc.__class__.__name__}: {exc}", file=sys.stderr)


def _persist_wandb(key: str) -> None:
    try:
        import wandb
    except Exception:
        return
    try:
        wandb.login(api_key=key)  # writes ~/.netrc
        print("[auth] cached W&B key via wandb.login (~/.netrc)", file=sys.stderr)
    except Exception as exc:
        print(f"[auth] W&B login cache failed: {exc.__class__.__name__}: {exc}", file=sys.stderr)


def ensure_credentials(
    *,
    hf_token: str | None = None,
    prompt: bool = True,
    want_wandb: bool = True,
    persist_login: bool = False,
) -> tuple[str | None, str | None]:
    """
    Resolve HF + W&B credentials, prompting interactively (hidden input) for
    anything missing. Returns (hf_token, wandb_key). Sets HF_TOKEN / WANDB_API_KEY
    in the environment as a side effect so downstream code that reads env vars
    picks them up.

    Resolution order for HF:
        explicit `hf_token` arg > HF_TOKEN env > HUGGING_FACE_HUB_TOKEN env >
        huggingface_hub cached login > interactive prompt (if TTY + prompt).

    Resolution order for W&B:
        WANDB_API_KEY env > existing wandb login > interactive prompt
        (if wandb installed + TTY + prompt).

    `prompt=False` (or `QWIP_NO_AUTH_PROMPT=1`) skips all prompts and just
    reports what's already available — safe for non-interactive runs.

    `persist_login=True` (or `QWIP_PERSIST_LOGIN=1`) caches the resolved keys
    via `huggingface_hub.login` / `wandb.login` so the pod stays logged in for
    later commands and stages. Off by default.
    """
    allow_prompt = prompt and not os.environ.get("QWIP_NO_AUTH_PROMPT")
    do_persist = persist_login or bool(os.environ.get("QWIP_PERSIST_LOGIN"))

    # --- HF token (needed before the model download, so prompt first) ---
    hf = hf_token or _env_or_none("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN") or _hf_cached_token()
    prompted_hf = False
    if not hf and allow_prompt and _have_tty():
        print("[auth] No HuggingFace token found in env / cached login.", file=sys.stderr)
        try:
            hf = getpass("  Paste your HF token (input hidden): ").strip() or None
            prompted_hf = True
        except (EOFError, KeyboardInterrupt):
            print("[auth] no HF token entered; continuing without one.", file=sys.stderr)
            hf = None
    if hf:
        os.environ["HF_TOKEN"] = hf
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", hf)
        if do_persist:
            _persist_hf(hf)

    # --- W&B key (asked right after HF, one after the other) ---
    wandb_key = _env_or_none("WANDB_API_KEY")
    wandb_ok = bool(wandb_key) or _wandb_has_login()
    prompted_wb = False
    if want_wandb and not wandb_ok and _wandb_installed() and allow_prompt and _have_tty():
        print("[auth] No W&B API key found (WANDB_API_KEY env / wandb login).", file=sys.stderr)
        try:
            wandb_key = getpass("  Paste your W&B API key (input hidden, Enter to skip): ").strip() or None
            prompted_wb = True
        except (EOFError, KeyboardInterrupt):
            print("[auth] no W&B key entered; continuing without one.", file=sys.stderr)
            wandb_key = None
    if wandb_key:
        os.environ["WANDB_API_KEY"] = wandb_key
        if do_persist:
            _persist_wandb(wandb_key)

    # Short status line so you can tell at a glance what's armed.
    armed = [k for k, v in (("HF", bool(hf)), ("W&B", bool(wandb_key))) if v]
    print(f"[auth] credentials armed: {', '.join(armed) or 'none'}", file=sys.stderr)
    return hf, wandb_key