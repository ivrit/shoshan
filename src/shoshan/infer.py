#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lemmatize Hebrew word-forms in context.

Shoshan retrieves the lemma from a fixed bank, and when the top retrieval is
morphologically implausible for the surface form (the coverage gate), it
transduces the lemma with a learned, form-relative edit script. Every output is
either a real bank entry or a bounded edit of the input word, so the model can
never emit a free-form string.

Runtime is offline once the weights are cached: the encoder is loaded from a
local directory and the lemma bank from disk. The query is pooled over the
target form's subword tokens (located by character offsets), the same encoding
used at training time.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from .model_joint import JointEncoder, UPOS
from .lemma_bank import LemmaBank
from .edit_script import apply_script, coverage
from .hub import DEFAULT_REPO, download_weights


class Lemmatizer:
    """A loaded Shoshan model: encoder + lemma bank + coverage-gated router."""

    def __init__(self, model_dir: Union[str, Path], bank_dir: Union[str, Path],
                 device: str = "cpu", use_router: bool = True,
                 cov_thresh: float = 0.60, min_sim: float = 0.0,
                 use_pos_filter: bool = True):
        self.enc = JointEncoder.load(model_dir, device=device)
        self.bank = LemmaBank.load(bank_dir)
        if self.bank.embeddings is None:
            raise RuntimeError(
                f"No precomputed embeddings in {bank_dir}. The bank must ship with "
                f"lemmas.npy (the encoded lemma vectors).")
        self.L = self.bank.embeddings  # [n, dim], L2-normalized
        self.device = device
        self.use_router = use_router
        self.cov_thresh = cov_thresh
        self.min_sim = min_sim
        self.use_pos_filter = use_pos_filter

    @classmethod
    def from_pretrained(cls, repo: str = DEFAULT_REPO, device: str = "cpu",
                        revision: Optional[str] = None, **kwargs) -> "Lemmatizer":
        """Download the weights from the Hub (cached) and load the model."""
        root = download_weights(repo, revision=revision)
        return cls(root / "model", root / "bank", device=device, **kwargs)

    @staticmethod
    def _span(form: str, sentence: str):
        st = sentence.find(form)
        return (st, st + len(form)) if st >= 0 else (0, len(form))

    def lemmatize(self, items: List[Dict[str, str]], batch: int = 256) -> List[Dict]:
        """Lemmatize a list of dicts.

        Each item needs a ``form`` and (ideally) a ``sentence``; an optional
        ``pos`` restricts retrieval to lemmas seen with that part of speech.
        Each result adds ``lemma``, ``pos`` (predicted), ``score`` (retrieval
        cosine), and ``source`` ("retrieved" or "transduced").
        """
        out: List[Dict] = []
        for i in range(0, len(items), batch):
            chunk = items[i:i + batch]
            sents = [str(it.get("sentence") or it["form"]) for it in chunk]
            spans = [self._span(str(it["form"]), s) for it, s in zip(chunk, sents)]
            with torch.no_grad():
                q, pos_logits, edit_logits = self.enc.encode_query(sents, spans)
            q = q.cpu().numpy()
            pos_pred = pos_logits.argmax(1).tolist()
            epred = edit_logits.argmax(1).tolist() if edit_logits is not None else None
            sims = q @ self.L.T
            for k, it in enumerate(chunk):
                form = str(it["form"])
                cand = (self.bank.candidate_ids(it.get("pos", ""))
                        if self.use_pos_filter else None)
                if cand is not None:
                    j = int(cand[int(np.argmax(sims[k][cand]))])
                else:
                    j = int(np.argmax(sims[k]))
                ret_lemma, ret_sim = self.bank.lemmas[j], float(sims[k][j])
                lemma, source = ret_lemma, "retrieved"
                if self.use_router and epred is not None:
                    trust = (coverage(form, ret_lemma) >= self.cov_thresh
                             and ret_sim >= self.min_sim)
                    if not trust:
                        lemma = apply_script(form, self.enc.scripts[epred[k]])
                        source = "transduced"
                out.append({**it, "lemma": lemma, "pos": UPOS[pos_pred[k]],
                            "score": ret_sim, "source": source})
        return out

    def lemma(self, form: str, sentence: Optional[str] = None) -> str:
        """Return just the lemma string for one form in context."""
        return self.lemmatize([{"form": form, "sentence": sentence or form}])[0]["lemma"]

    def annotate(self, sentence: str) -> List[Dict]:
        """Tokenize `sentence` and lemmatize every word token in context."""
        from .text import tokenize
        items = [{"form": f, "sentence": sentence} for f in tokenize(sentence)]
        return self.lemmatize(items)
