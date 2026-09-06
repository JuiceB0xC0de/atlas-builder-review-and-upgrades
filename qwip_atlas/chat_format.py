"""Chat-template wrapping for prompt tokenization.

Instruct models were post-trained inside a chat template. Feeding raw prompt
text captures activations off the model's operating distribution; the
assistant-persona directions are engaged when the official template (and an
assistant generation prompt) is present. Each prompt becomes a single user turn
plus an assistant generation prompt. For gemma-4 that renders as:

    <bos><|turn>user\n{prompt}<turn|>\n<|turn>model\n

Mechanics
---------
* ``tokenizer.apply_chat_template(..., add_generation_prompt=True)`` already
  emits BOS, so when wrapping is on we tokenize with ``add_special_tokens=False``.
* The template *tail* (``<turn|>\\n<|turn>model\\n``) is what last-token pooling
  reads. HF tokenizers truncate on the right by default, so a long prompt used
  to silently lose the tail and the "last token" became an arbitrary mid-prompt
  token. ``encode_prompts`` now truncates the *user content* to the token budget
  left after the template overhead, so both BOS and the tail always survive, and
  sets ``truncation_side="left"`` as a backstop.
* ``assert_generation_tail`` verifies, on real token ids, that every row ends
  with the template's generation-prompt tokens. The census calls it on the
  first batch and refuses to run if the tail is not what pooling will read.

The active state is stored once on the tokenizer (``atlas_chat_template``) at
load time. ``set_chat_template`` raises if the tokenizer rejects the attribute:
a silent no-op here means two runs become incomparable without anyone knowing.
"""
from __future__ import annotations

import hashlib
from typing import Any, Iterable

_FLAG_ATTR = "atlas_chat_template"
_OVERHEAD_ATTR = "atlas_chat_template_overhead"
_TAIL_ATTR = "atlas_chat_template_tail_ids"

# Sentinel that survives tokenization intact and never appears in real prompts.
_PROBE = "QWIPPROBE7"


def set_chat_template(tokenizer, enabled: bool) -> None:
    """Record on the tokenizer whether prompts should be chat-template wrapped.

    Raises instead of swallowing: a tokenizer that cannot carry the flag would
    otherwise silently run in raw mode while the run believes it is templated.
    """
    enabled = bool(enabled)
    if enabled and not getattr(tokenizer, "chat_template", None):
        raise ValueError(
            f"--chat-template requested but {type(tokenizer).__name__} has no chat_template; "
            "pass --no-chat-template for a base model or a tokenizer without a template"
        )
    setattr(tokenizer, _FLAG_ATTR, enabled)
    if getattr(tokenizer, _FLAG_ATTR, None) is not enabled:
        raise RuntimeError(f"{type(tokenizer).__name__} did not retain attribute {_FLAG_ATTR!r}")
    if enabled:
        # Left-truncate as a backstop so the generation prompt is never the part
        # that gets cut; encode_prompts normally pre-truncates content instead.
        try:
            tokenizer.truncation_side = "left"
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"cannot set truncation_side on {type(tokenizer).__name__}: {exc}") from exc
        setattr(tokenizer, _OVERHEAD_ATTR, template_overhead_tokens(tokenizer))
        setattr(tokenizer, _TAIL_ATTR, generation_tail_ids(tokenizer))


def chat_template_enabled(tokenizer, override: bool | None = None) -> bool:
    """Resolve the effective wrapping decision: explicit override wins, else the
    value stamped on the tokenizer at load time, else off."""
    if override is not None:
        return bool(override)
    return bool(getattr(tokenizer, _FLAG_ATTR, False))


def render_user_turn(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def format_prompt_text(tokenizer, prompt: str, chat_template: bool | None = None) -> str:
    """Return the prompt as a user turn + assistant generation prompt, or the
    prompt unchanged when chat templating is off."""
    if not chat_template_enabled(tokenizer, chat_template):
        return prompt
    return render_user_turn(tokenizer, prompt)


def template_sha(tokenizer) -> str | None:
    """sha256 of the raw Jinja chat template, for the run manifest."""
    tpl = getattr(tokenizer, "chat_template", None)
    if not tpl:
        return None
    if isinstance(tpl, dict):  # multi-template tokenizers
        tpl = "\n".join(f"{k}:{v}" for k, v in sorted(tpl.items()))
    return "sha256:" + hashlib.sha256(str(tpl).encode("utf-8")).hexdigest()


def _ids(tokenizer, text: str) -> list[int]:
    out = tokenizer(text, add_special_tokens=False)["input_ids"]
    # Some fake/legacy tokenizers return a plain list, others a BatchEncoding.
    if out and isinstance(out[0], list):
        out = out[0]
    return list(out)


def template_overhead_tokens(tokenizer) -> int:
    """Number of tokens the template adds around the user content."""
    rendered = render_user_turn(tokenizer, _PROBE)
    n_probe = len(_ids(tokenizer, _PROBE))
    return max(0, len(_ids(tokenizer, rendered)) - n_probe)


def generation_tail_ids(tokenizer) -> list[int]:
    """Token ids of the template text that follows the user content.

    These are the tokens every templated row must end with; the last one is
    what last-token pooling reads.
    """
    rendered = render_user_turn(tokenizer, _PROBE)
    cut = rendered.rfind(_PROBE)
    if cut < 0:
        raise RuntimeError("chat template did not include the user content; cannot locate the generation tail")
    tail_text = rendered[cut + len(_PROBE):]
    return _ids(tokenizer, tail_text)


def truncate_content(tokenizer, prompt: str, budget: int) -> str:
    """Trim the user content to at most ``budget`` tokens (keeping its head)."""
    if budget <= 0:
        return ""
    ids = _ids(tokenizer, prompt)
    if len(ids) <= budget:
        return prompt
    return tokenizer.decode(ids[:budget], skip_special_tokens=False)


def encode_prompts(
    tokenizer,
    prompts: Iterable[str],
    *,
    chat_template: bool | None = None,
    **tok_kwargs: Any,
):
    """Tokenize a batch of prompt strings, optionally wrapping each in the chat
    template first.

    When wrapping is active:
      * the template already carries BOS, so ``add_special_tokens`` defaults to
        False (an explicit value in ``tok_kwargs`` is respected);
      * if ``max_length`` is given, the user content is pre-truncated to
        ``max_length - overhead`` tokens so the template tail is never cut.
    """
    prompts = list(prompts)
    if chat_template_enabled(tokenizer, chat_template):
        max_length = tok_kwargs.get("max_length")
        if max_length is not None:
            overhead = getattr(tokenizer, _OVERHEAD_ATTR, None)
            if overhead is None:
                overhead = template_overhead_tokens(tokenizer)
            budget = int(max_length) - int(overhead)
            if budget < 1:
                raise ValueError(
                    f"max_length={max_length} leaves no room for content after the "
                    f"{overhead}-token chat template; raise --max-length"
                )
            prompts = [truncate_content(tokenizer, p, budget) for p in prompts]
        prompts = [render_user_turn(tokenizer, p) for p in prompts]
        tok_kwargs.setdefault("add_special_tokens", False)
    return tokenizer(prompts, **tok_kwargs)


def assert_generation_tail(tokenizer, enc, chat_template: bool | None = None) -> list[int]:
    """Check that every row of a batch encoding ends with the template tail.

    Assumes left padding (real tokens right-aligned), which is how every
    forward-pass stage tokenizes. Returns the tail ids that were verified.
    Raises AssertionError with the offending row rendered when the check fails.
    """
    if not chat_template_enabled(tokenizer, chat_template):
        return []
    tail = getattr(tokenizer, _TAIL_ATTR, None) or generation_tail_ids(tokenizer)
    if not tail:
        raise AssertionError("chat template has an empty generation tail; last-token pooling would read user content")
    input_ids = enc["input_ids"]
    attn = enc["attention_mask"]
    try:
        input_ids = input_ids.tolist()
        attn = attn.tolist()
    except AttributeError:
        pass
    for r, (row, mask) in enumerate(zip(input_ids, attn)):
        real = [t for t, m in zip(row, mask) if m]
        if real[-len(tail):] != list(tail):
            got = tokenizer.decode(real[-len(tail) - 3:])
            want = tokenizer.decode(tail)
            raise AssertionError(
                f"row {r}: rendered prompt does not end with the generation prompt. "
                f"expected tail {want!r}, got {got!r}. Last-token pooling would read the wrong token."
            )
    return list(tail)
