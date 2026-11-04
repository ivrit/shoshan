"""Sentences longer than the encoder's input window.

The encoder reads at most `JointEncoder.max_len` subword tokens of a sentence (512 for the
BERT backbone). A word that lies past that window has no hidden states, so no prediction
exists for it. Earlier releases capped the window at 160 and silently pooled the
sentence-start vector for anything past it; the POS head read that vector as PART, and
with `blank_function_words=True` every such word was dropped as a stopword. These tests
pin the fix: the window is the encoder's own limit, and a word past it is reported as
source="truncated" with its surface form as the lemma, never blanked.
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


def _past_window_items():
    # > 512 subwords: the last word lies past the encoder window, the first does not.
    long_sent = " ".join(["מילה"] * 600) + " סוף"
    return [{"form": "מילה", "sentence": long_sent, "start": 0},
            {"form": "סוף", "sentence": long_sent, "start": long_sent.rindex("סוף")}]


def test_default_window_is_the_encoder_limit(lz):
    """A sentence of ~300 subwords (over the former cap of 160) is encoded whole."""
    import torch
    sent = " ".join(["הממשלה החליטה להקצות תקציב נוסף"] * 60)
    n = len(lz.enc.tok(sent)["input_ids"])
    assert 160 < n <= lz.enc.max_len, n
    with torch.no_grad():
        hidden, offsets = lz.enc.encode_sentences([sent])
        _, trunc = lz.enc.pool_spans(hidden, offsets, [0], [(0, 6)], return_truncated=True)
    assert hidden.shape[1] == n
    assert trunc == [False]


def test_past_window_token_is_reported_as_truncated(lz):
    out = lz.lemmatize(_past_window_items())
    assert out[0]["source"] != "truncated"
    assert out[0]["lemma"] == "מילה"
    assert out[1]["source"] == "truncated"
    assert out[1]["lemma"] == "סוף"
    assert out[1]["pos"] == "X"
    assert out[1]["score"] == 0.0


def test_past_window_token_is_never_blanked_as_function_word(lz):
    """The reported defect: PART is a function POS, so the word was dropped as a stopword."""
    old = lz.blank_function_words
    lz.blank_function_words = True
    try:
        out = lz.lemmatize(_past_window_items())
    finally:
        lz.blank_function_words = old
    assert out[1]["source"] == "truncated"
    assert out[1]["lemma"] == "סוף"


def test_past_window_parity_encode_once_vs_per_token(lz):
    """Encode-once and per-token encoding truncate the shared sentence the same way."""
    items = _past_window_items()
    once = lz.lemmatize(items)
    ref = [lz.lemmatize([it])[0] for it in items]
    for a, b in zip(once, ref):
        assert (a["lemma"], a["pos"], a["source"]) == (b["lemma"], b["pos"], b["source"])
    assert once[1]["source"] == "truncated"


def test_lemmatize_text_keeps_truncated_words_indexable(lz):
    """Through the document API the word stays in `tokens` and in the indexable stream,
    carried by its surface form, instead of vanishing."""
    text = " ".join(["מילה"] * 600) + " סוף"
    doc = lz.lemmatize_text(text)
    last = doc["tokens"][-1]
    assert last["token"] == "סוף"
    assert last["source"] == "truncated"
    assert last["lemma"] == "סוף"
    assert any(e["token"] == "סוף" for e in doc["es_tokens"])
