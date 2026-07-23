"""Encode-once perf-path parity tests.

`Lemmatizer.lemmatize` encodes each DISTINCT sentence ONCE and pools every token's span from
the shared hidden state (instead of re-encoding the sentence per token). The user-visible
LABELS must be preserved: predicted lemma + POS identical, retrieval score within 1e-5. The
reference is a one-item-per-call loop (each token encoded alone) — the strictest batch
composition to compare against.

`source` is deliberately NOT asserted equal, and sentences here are content-word-heavy: when
two bank vectors are within the batched-matmul noise floor (~1e-7) rebatching can pick the
other near-tied neighbour. That flips provenance (and, for a bank with duplicate entries, the
lemma) — inherent float non-associativity, not an encode-once bug.

Skipped automatically when the weights aren't cached (no network in CI).
"""
import pytest


@pytest.fixture(scope="module")
def lz():
    try:
        import os
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import shoshan
        return shoshan.Lemmatizer.from_pretrained(device="cpu", verbose=False)
    except Exception as e:                       # weights not cached / offline
        pytest.skip(f"weights unavailable: {e}")


def _items(sents):
    items = []
    for s in sents:
        pos = 0
        for w in s.split(" "):
            st = s.index(w, pos)
            items.append({"form": w, "sentence": s, "start": st})
            pos = st + len(w)
    return items


def _assert_parity(once, ref, *, tol=1e-5):
    assert len(once) == len(ref)
    for a, b in zip(once, ref):
        assert a["lemma"] == b["lemma"], (a, b)
        assert a["pos"] == b["pos"], (a, b)
        assert abs(a["score"] - b["score"]) < tol, (a["form"], a["score"], b["score"])


_DOC = ["הילדים הלכו לבית הספר הגדול בבוקר מוקדם",
        "המורה כתבה מילים חדשות וקשות על הלוח",
        "החוקרים גילו ממצאים חשובים בערים העתיקות"]


def test_encode_once_parity_multi_sentence(lz):
    """Batched multi-sentence doc (sentences deduped, encoded once) matches the per-token
    encode token-for-token."""
    items = _items(_DOC)
    once = lz.lemmatize(items)                     # encode-once
    ref = [lz.lemmatize([it])[0] for it in items]  # per-token reference
    _assert_parity(once, ref)


def test_batch_size_invariance(lz):
    """The `batch` lever (distinct sentences per forward — the throughput knob) must not
    change labels: batch=3 vs batch=256 give the same lemma + POS, scores within 1e-5."""
    items = _items(_DOC)
    _assert_parity(lz.lemmatize(items, batch=3), lz.lemmatize(items, batch=256))


def test_on_device_retrieval_matches_numpy(lz):
    """The on-device (torch) retrieval path must match the numpy path label-for-label. On an
    accelerator the bank lives on-device (self.L_t) and q @ L.T + argmax run there; here we
    force that branch with a CPU torch tensor so it's validated without a GPU."""
    import torch
    items = _items(_DOC)
    numpy_out = lz.lemmatize(items)                # numpy retrieval (L_t is None on CPU)
    assert lz.L_t is None
    lz.L_t = torch.from_numpy(lz.L)                # force the on-device branch (CPU tensor)
    lz._cand_dev_cache = {}
    try:
        torch_out = lz.lemmatize(items)
    finally:
        lz.L_t = None
        lz._cand_dev_cache = {}
    _assert_parity(torch_out, numpy_out)
