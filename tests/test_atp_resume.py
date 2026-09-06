"""CPU-only test for the Stage 3 AtP per-pair checkpoint/resume logic.

Simulates Rick's exact failure mode: the run crashes (e.g. CUDA OOM or a kill)
part-way through the AtP pair loop, and a re-run must resume from the last
completed pair instead of restarting from pair 0.

No GPU, no real LLM: a tiny stub model + tokenizer drive `_capture_atp_gradients`.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import torch
import torch.nn.functional as F

from qwip_atlas.sub_zero_surgery import _capture_atp_gradients


# --------------------------------------------------------------------------- #
# Stubs
# --------------------------------------------------------------------------- #
class _Out:
    def __init__(self, logits):
        self.logits = logits


class StubLayer(torch.nn.Module):
    """A layer whose mlp exposes gate/up/down projections, matching
    `_get_projection_map`'s Llama MLP aliases."""

    def __init__(self, hidden):
        super().__init__()
        self.mlp = torch.nn.Module()
        self.mlp.gate_proj = torch.nn.Linear(hidden, hidden, bias=False)
        self.mlp.up_proj = torch.nn.Linear(hidden, hidden, bias=False)
        self.mlp.down_proj = torch.nn.Linear(hidden, hidden, bias=False)


class StubModel(torch.nn.Module):
    def __init__(self, vocab=16, hidden=8):
        super().__init__()
        self.config = type("Cfg", (), {"_name_or_path": "stub-model"})()
        self.embed = torch.nn.Embedding(vocab, hidden)
        self.layer = StubLayer(hidden)
        self.unembed = torch.nn.Linear(hidden, vocab, bias=False)
        # forward-call counter so tests can assert how many pairs actually ran.
        self.calls = 0
        # if set, raise this exception on the Nth forward call (1-indexed).
        self._raise_on_call: int | None = None
        self._raise_exc: BaseException | None = None

    def forward(self, input_ids, use_cache=False, **kw):
        self.calls += 1
        if self._raise_on_call is not None and self.calls == self._raise_on_call:
            raise self._raise_exc
        h = self.embed(input_ids)
        g = self.layer.mlp.gate_proj(h)
        u = self.layer.mlp.up_proj(h)
        d = self.layer.mlp.down_proj(F.gelu(g) * u)
        logits = self.unembed(h + d)
        return _Out(logits)


class StubTokenizer:
    """Fixed 4-token output regardless of input prompt."""

    def __call__(self, prompt, return_tensors="pt", truncation=True, max_length=None):
        ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _make_inputs(hidden=8, rank=4, n_act=5):
    torch.manual_seed(0)
    proj_svd = {
        0: {
            "gate_proj": (
                torch.randn(hidden, rank),
                torch.randn(rank),
                torch.randn(rank, hidden),  # vh: [rank, in]
            ),
            "up_proj": (
                torch.randn(hidden, rank),
                torch.randn(rank),
                torch.randn(rank, hidden),
            ),
            "down_proj": (
                torch.randn(hidden, rank),
                torch.randn(rank),
                torch.randn(rank, hidden),
            ),
        }
    }
    corp_acts = {0: {p: torch.randn(n_act, hidden) for p in proj_svd[0]}}
    auth_acts = {0: {p: torch.randn(n_act, hidden) for p in proj_svd[0]}}
    return proj_svd, corp_acts, auth_acts


def _run(model, ckpt_dir, fingerprint, corp, auth, **kw):
    proj_svd, corp_acts, auth_acts = _make_inputs()
    return _capture_atp_gradients(
        model, StubTokenizer(), corp, auth, [model.layer],
        corp_acts, auth_acts, proj_svd, max_length=4,
        ckpt_dir=Path(ckpt_dir), fingerprint=fingerprint,
        ckpt_name="stage3_atp", **kw,
    )


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_full_run_writes_checkpoint():
    tmp = Path("/tmp/atp_resume_full")
    if tmp.exists():
        shutil.rmtree(tmp)
    m = StubModel()
    out = _run(m, tmp, "fp1", ["a", "b"], ["a", "b"])  # n_pairs=2
    assert set(out[0]) == {"gate_proj", "up_proj", "down_proj"}
    ckpt = tmp / "stage3_atp.pt"
    assert ckpt.exists(), "per-pair checkpoint not written"
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert blob["fingerprint"] == "fp1"
    assert blob["done"] == 2, f"expected done=2, got {blob['done']}"
    assert m.calls == 2
    print("PASS: full run writes done=2 checkpoint")


def test_resume_skips_completed_pairs():
    """Crash-after-2 scenario: a done=2 ckpt exists, real run has 3 pairs.
    Re-run must execute ONLY pair index 2 (one forward call), not 0+1+2."""
    tmp = Path("/tmp/atp_resume_partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    # First: complete 2 pairs, leaving a done=2 checkpoint.
    m1 = StubModel()
    _run(m1, tmp, "fpX", ["a", "b"], ["a", "b"])
    assert (tmp / "stage3_atp.pt").exists()
    # Now resume with a 3-pair corpus: only pair index 2 should run.
    m2 = StubModel()
    out = _run(m2, tmp, "fpX", ["a", "b", "c"], ["a", "b", "c"])
    assert m2.calls == 1, f"resume should run 1 pair, ran {m2.calls}"
    blob = torch.load(tmp / "stage3_atp.pt", weights_only=False)
    assert blob["done"] == 3, f"expected done=3 after resume, got {blob['done']}"
    # all three projection scores present and finite
    for p in ("gate_proj", "up_proj", "down_proj"):
        assert torch.isfinite(out[0][p]).all(), f"{p} has non-finite values"
    print("PASS: resume ran only the missing pair (1 forward call), done=3")


def test_fingerprint_mismatch_starts_fresh():
    tmp = Path("/tmp/atp_resume_fp")
    if tmp.exists():
        shutil.rmtree(tmp)
    m1 = StubModel()
    _run(m1, tmp, "fp1", ["a", "b"], ["a", "b"])
    # Different fingerprint -> must ignore the old ckpt and recompute all pairs.
    m2 = StubModel()
    _run(m2, tmp, "fp2", ["a", "b"], ["a", "b"])
    assert m2.calls == 2, f"fingerprint mismatch should recompute 2 pairs, ran {m2.calls}"
    blob = torch.load(tmp / "stage3_atp.pt", weights_only=False)
    assert blob["fingerprint"] == "fp2"
    print("PASS: fingerprint mismatch recomputes from scratch")


def test_oom_on_one_pair_is_skipped_not_fatal():
    tmp = Path("/tmp/atp_resume_oom")
    if tmp.exists():
        shutil.rmtree(tmp)
    m = StubModel()
    # Raise a CUDA OOM on the 3rd forward call (the 3rd pair).
    m._raise_on_call = 3
    m._raise_exc = torch.cuda.OutOfMemoryError("simulated OOM")
    out = _run(m, tmp, "fp1", ["a", "b", "c"], ["a", "b", "c"])  # n_pairs=3
    # pairs 0 and 1 computed; pair 2 (3rd call) skipped, not fatal.
    assert m.calls == 3, f"expected 3 attempts, got {m.calls}"
    for p in ("gate_proj", "up_proj", "down_proj"):
        assert torch.isfinite(out[0][p]).all(), f"{p} non-finite after OOM skip"
    blob = torch.load(tmp / "stage3_atp.pt", weights_only=False)
    assert blob["done"] == 3, f"OOM-skipped pair should still advance done, got {blob['done']}"
    print("PASS: OOM on one pair is skipped, stage completes, done advances")


if __name__ == "__main__":
    test_full_run_writes_checkpoint()
    test_resume_skips_completed_pairs()
    test_fingerprint_mismatch_starts_fresh()
    test_oom_on_one_pair_is_skipped_not_fatal()
    print("\nALL ATP RESUME TESTS PASSED")