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
import pandas as pd
import torch

from .model_joint import JointEncoder, UPOS
from .lemma_bank import LemmaBank
from .edit_script import apply_script, coverage
from .suppletive import SuppletiveGate
from .hub import DEFAULT_REPO, download_weights

# Closed-class parts of speech. For information retrieval these are stopwords, and
# they are also where the edit-script fallback is least reliable (it has no real
# lemma to copy toward). With blank_function_words=True they return an empty lemma.
FUNCTION_POS = {"ADP", "AUX", "CCONJ", "SCONJ", "DET", "PRON", "PART", "INTJ"}


class Lemmatizer:
    """A loaded Shoshan model: encoder + lemma bank + coverage-gated router."""

    def __init__(self, model_dir: Union[str, Path], bank_dir: Union[str, Path],
                 device: str = "cpu", use_router: bool = True,
                 cov_thresh: float = 0.60, min_sim: float = 0.0,
                 use_pos_filter: bool = True, blank_function_words: bool = False,
                 log_misses: bool = False, suppletives_path: Union[str, Path, None] = None):
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
        self.blank_function_words = blank_function_words
        # When True, flag tokens where retrieval is not trusted (likely OOV / bank
        # miss). The frequency-sorted log (write_miss_log) is a curation worklist:
        # the word-forms most worth annotating or adding to the lexicon.
        self.log_misses = log_misses
        self.miss_log: List[Dict] = []
        # Curated (surface, POS) -> lemma lookup for suppletive forms whose lemma shares
        # too few characters with the surface (היא->הוא, נשים->איש) for the coverage gate
        # to trust the correct retrieval. Optional; ships in the model dir.
        sup = Path(suppletives_path) if suppletives_path else Path(model_dir) / "suppletives.csv"
        self.suppletive_gate = SuppletiveGate(sup) if Path(sup).exists() else None

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
        cosine), and ``source``: "retrieved" (from the bank), "suppletive" (a
        curated suppletive-lexicon lookup, score 1.0), "transduced" (the edit-script
        fallback), or "function" (a closed-class word blanked because
        ``blank_function_words`` is on).
        """
        from .text import normalize_text
        out: List[Dict] = []
        for i in range(0, len(items), batch):
            chunk = items[i:i + batch]
            # normalize input the same way training forms are normalized (quote/prime
            # variants -> ASCII); length-preserving, so spans stay valid.
            sents = [normalize_text(str(it.get("sentence") or it["form"])) for it in chunk]
            forms = [normalize_text(str(it["form"])) for it in chunk]
            spans = [self._span(f, s) for f, s in zip(forms, sents)]
            with torch.no_grad():
                q, pos_logits, edit_logits = self.enc.encode_query(sents, spans)
            q = q.cpu().numpy()
            pos_pred = pos_logits.argmax(1).tolist()
            epred = edit_logits.argmax(1).tolist() if edit_logits is not None else None
            sims = q @ self.L.T
            for k, it in enumerate(chunk):
                form = forms[k]
                cand = (self.bank.candidate_ids(it.get("pos", ""))
                        if self.use_pos_filter else None)
                if cand is not None:
                    j = int(cand[int(np.argmax(sims[k][cand]))])
                else:
                    j = int(np.argmax(sims[k]))
                ret_lemma, ret_sim = self.bank.lemmas[j], float(sims[k][j])
                pos = UPOS[pos_pred[k]]
                # curated suppletive lookup (surface+POS -> lemma), keyed on predicted POS
                # so homographs are split (accusative את does not match the pronoun entry).
                sup = (self.suppletive_gate.lemma(form, pos)
                       if self.use_router and self.suppletive_gate is not None else None)
                score = ret_sim
                if self.blank_function_words and pos in FUNCTION_POS:
                    lemma, source = "", "function"          # IR stopword blanking wins
                elif sup is not None:
                    lemma, source, score = sup, "suppletive", 1.0   # curated-dict lookup
                else:
                    lemma, source = ret_lemma, "retrieved"
                    if self.use_router and epred is not None:
                        trust = (coverage(form, ret_lemma) >= self.cov_thresh
                                 and ret_sim >= self.min_sim)
                        if not trust:
                            lemma = apply_script(form, self.enc.scripts[epred[k]])
                            source = "transduced"
                    if self.log_misses:
                        self._record_miss(form, pos, lemma, ret_lemma, ret_sim)
                out.append({**it, "form": form, "lemma": lemma, "pos": pos,
                            "score": score, "source": source})
        return out

    def _record_miss(self, form: str, pos: str, lemma: str,
                     ret_lemma: str, ret_sim: float) -> None:
        """Flag a token where retrieval is NOT trusted: the bank's best lemma is
        morphologically implausible for the form (coverage below threshold) or its
        similarity is below the floor. These are the likely-OOV / bank-miss tokens
        worth annotating or adding to the lexicon. A base form that is its own lemma
        (ספר->ספר, coverage 1.0) is correctly not flagged even though lemma==form."""
        cov = coverage(form, ret_lemma)
        cov_low, sim_low = cov < self.cov_thresh, ret_sim < self.min_sim
        if not (cov_low or sim_low):
            return
        reasons = [r for r, on in (("coverage_low", cov_low), ("sim_low", sim_low)) if on]
        if lemma == form:  # the transducer also gave up and copied the surface form
            reasons.append("copy_fallback")
        self.miss_log.append({
            "wordform": form, "predicted_pos": pos, "predicted_lemma": lemma,
            "retrieved_lemma": ret_lemma, "coverage": round(cov, 4),
            "sim": round(ret_sim, 4), "reason": "+".join(reasons)})

    def write_miss_log(self, path: Union[str, Path]) -> int:
        """Aggregate misses by (wordform, predicted_pos) and write a frequency-sorted
        CSV: the prioritized worklist of forms to curate. Returns the row count."""
        cols = ["wordform", "predicted_pos", "count", "predicted_lemma",
                "retrieved_lemma", "mean_coverage", "mean_sim", "reason"]
        if not self.miss_log:
            pd.DataFrame(columns=cols).to_csv(path, index=False, encoding="utf-8")
            return 0
        df = pd.DataFrame(self.miss_log)
        agg = (df.groupby(["wordform", "predicted_pos"], sort=False)
                 .agg(count=("wordform", "size"),
                      predicted_lemma=("predicted_lemma", "first"),
                      retrieved_lemma=("retrieved_lemma", "first"),
                      mean_coverage=("coverage", "mean"),
                      mean_sim=("sim", "mean"),
                      reason=("reason", lambda s: s.mode().iat[0]))
                 .reset_index()
                 .sort_values("count", ascending=False))
        agg[cols].to_csv(path, index=False, encoding="utf-8")
        return len(agg)

    def lemma(self, form: str, sentence: Optional[str] = None) -> str:
        """Return just the lemma string for one form in context."""
        return self.lemmatize([{"form": form, "sentence": sentence or form}])[0]["lemma"]

    def annotate(self, sentence: str) -> List[Dict]:
        """Tokenize `sentence` and lemmatize every word token in context."""
        from .text import tokenize, normalize_text
        sentence = normalize_text(sentence)   # so quote variants don't split acronyms
        items = [{"form": f, "sentence": sentence} for f in tokenize(sentence)]
        return self.lemmatize(items)
