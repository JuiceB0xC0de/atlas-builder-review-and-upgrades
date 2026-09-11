"""Unit tests for qwip_atlas.chat_format.

Pins the correctness-critical behaviours without a real model:
  1. Chat templating OFF: prompts pass through untouched, tokenizer adds BOS (one).
  2. Chat templating ON: each prompt is wrapped via apply_chat_template (which
     already carries BOS), so encode_prompts tokenizes with add_special_tokens=False
     -> exactly one BOS, never two.
  3. set_chat_template raises (no silent no-op) when the tokenizer has no template
     or refuses the attribute.
  4. Long prompts are pre-truncated on the *content* so the generation tail is
     never cut; assert_generation_tail verifies the last real tokens.
"""
import pytest

from qwip_atlas.chat_format import (
    assert_generation_tail,
    chat_template_enabled,
    encode_prompts,
    format_prompt_text,
    generation_tail_ids,
    set_chat_template,
    template_overhead_tokens,
    template_sha,
)


class FakeTokenizer:
    """Whitespace tokenizer with a gemma-style template.

    Vocabulary is built lazily; special markers are single tokens. Emulates HF
    left padding and right truncation defaults."""

    BOS = "<bos>"
    chat_template = "{{ '<bos><|turn>user\\n' + messages[0]['content'] + '<turn|>\\n<|turn>model\\n' }}"

    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.inv: dict[int, str] = {}
        self.last_call = None
        self.padding_side = "left"
        self.truncation_side = "right"

    # -- template ---------------------------------------------------------
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        content = messages[-1]["content"]
        tail = " <turn|> \n <|turn> model \n" if add_generation_prompt else " <turn|> \n"
        return f"{self.BOS} <|turn> user \n {content}{tail}"

    # -- tokenization -------------------------------------------------------
    def _tok(self, text):
        ids = []
        for w in text.split(" "):
            if not w:
                continue
            if w not in self.vocab:
                self.vocab[w] = len(self.vocab) + 3
                self.inv[self.vocab[w]] = w
            ids.append(self.vocab[w])
        return ids

    def __call__(self, prompts, add_special_tokens=True, padding=False, truncation=False,
                 max_length=None, return_tensors=None):
        single = isinstance(prompts, str)
        texts = [prompts] if single else list(prompts)
        rows = []
        for t in texts:
            ids = self._tok(t)
            if add_special_tokens:
                ids = self._tok(self.BOS) + ids
            if truncation and max_length is not None and len(ids) > max_length:
                ids = ids[:max_length] if self.truncation_side == "right" else ids[-max_length:]
            rows.append(ids)
        self.last_call = {"texts": texts, "add_special_tokens": add_special_tokens}
        if single:
            return {"input_ids": rows[0], "attention_mask": [1] * len(rows[0])}
        width = max(len(r) for r in rows)
        input_ids, attn = [], []
        for r in rows:
            pad = width - len(r)
            input_ids.append([0] * pad + r)
            attn.append([0] * pad + [1] * len(r))
        return {"input_ids": input_ids, "attention_mask": attn}

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(self.inv.get(i, "?") for i in ids)


def _n_bos(tok, ids):
    return sum(1 for i in ids if tok.inv.get(i) == tok.BOS)


def test_flag_resolution_default_off():
    tok = FakeTokenizer()
    assert chat_template_enabled(tok) is False


def test_set_and_read_flag():
    tok = FakeTokenizer()
    set_chat_template(tok, True)
    assert chat_template_enabled(tok) is True
    assert chat_template_enabled(tok, override=False) is False  # explicit wins
    assert tok.truncation_side == "left"  # backstop so the tail is never the cut part


def test_set_chat_template_raises_without_template():
    class NoTemplate(FakeTokenizer):
        chat_template = None
    with pytest.raises(ValueError):
        set_chat_template(NoTemplate(), True)
    # off is always allowed
    set_chat_template(NoTemplate(), False)


def test_set_chat_template_raises_when_attribute_rejected():
    class Frozen(FakeTokenizer):
        __slots__ = ()

        def __setattr__(self, k, v):
            if k == "atlas_chat_template":
                return  # silently drop, the failure mode the old try/except hid
            object.__setattr__(self, k, v)
    with pytest.raises(RuntimeError):
        set_chat_template(Frozen(), True)


def test_off_passes_through_with_one_bos():
    tok = FakeTokenizer()
    out = encode_prompts(tok, ["hello", "world"])
    assert tok.last_call["add_special_tokens"] is True
    assert all("<|turn>" not in t for t in tok.last_call["texts"])  # not wrapped
    assert all(_n_bos(tok, r) == 1 for r in out["input_ids"])


def test_on_wraps_and_avoids_double_bos():
    tok = FakeTokenizer()
    set_chat_template(tok, True)
    out = encode_prompts(tok, ["hello", "world"], padding=True)
    assert tok.last_call["add_special_tokens"] is False
    for t in tok.last_call["texts"]:
        assert "<|turn> user" in t and t.endswith("<|turn> model \n")
    for r in out["input_ids"]:
        assert _n_bos(tok, r) == 1  # single BOS, not two


def test_explicit_add_special_tokens_respected_when_on():
    tok = FakeTokenizer()
    set_chat_template(tok, True)
    encode_prompts(tok, ["x"], add_special_tokens=True)
    assert tok.last_call["add_special_tokens"] is True


def test_format_prompt_text_single():
    tok = FakeTokenizer()
    assert format_prompt_text(tok, "hi") == "hi"  # off
    set_chat_template(tok, True)
    assert format_prompt_text(tok, "hi").endswith("hi <turn|> \n <|turn> model \n")


def test_template_sha_and_overhead():
    tok = FakeTokenizer()
    assert template_sha(tok).startswith("sha256:")
    # <bos> <|turn> user \n ... <turn|> \n <|turn> model \n  = 9 tokens of scaffolding
    assert template_overhead_tokens(tok) == 9
    assert tok.decode(generation_tail_ids(tok)) == "<turn|> \n <|turn> model \n"


def test_long_prompt_keeps_generation_tail():
    tok = FakeTokenizer()
    set_chat_template(tok, True)
    long = " ".join(f"w{i}" for i in range(100))
    enc = encode_prompts(tok, [long, "short"], padding=True, truncation=True, max_length=32)
    tail = assert_generation_tail(tok, enc)  # would raise if the tail were cut
    assert tok.decode(tail) == "<turn|> \n <|turn> model \n"
    for row, mask in zip(enc["input_ids"], enc["attention_mask"]):
        real = [t for t, m in zip(row, mask) if m]
        assert len(real) <= 32
        assert _n_bos(tok, real) == 1  # head survived too (content was cut, not the scaffold)


def test_assert_generation_tail_detects_cut_tail():
    tok = FakeTokenizer()
    set_chat_template(tok, True)
    # Bypass encode_prompts: right-truncate a templated prompt so the tail is lost.
    tok.truncation_side = "right"
    rendered = tok.apply_chat_template([{"role": "user", "content": " ".join(["w"] * 50)}],
                                       tokenize=False, add_generation_prompt=True)
    enc = tok([rendered], add_special_tokens=False, padding=True, truncation=True, max_length=20)
    with pytest.raises(AssertionError):
        assert_generation_tail(tok, enc)


def test_max_length_too_small_for_template_raises():
    tok = FakeTokenizer()
    set_chat_template(tok, True)
    with pytest.raises(ValueError):
        encode_prompts(tok, ["hello"], truncation=True, max_length=5)
