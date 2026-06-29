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
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch

from .model_joint import JointEncoder, UPOS
from .lemma_bank import LemmaBank
from .edit_script import apply_script, coverage
from .suppletive import SuppletiveGate
from .doc_text import split_sentences, tokenize as doc_tokenize
from .hub import DEFAULT_REPO, download_weights

_PKG_DATA = Path(__file__).parent / "data"

# Closed-class parts of speech. For information retrieval these are stopwords, and
# they are also where the edit-script fallback is least reliable (it has no real
# lemma to copy toward). With blank_function_words=True they return an empty lemma.
FUNCTION_POS = {"ADP", "AUX", "CCONJ", "SCONJ", "DET", "PRON", "PART", "INTJ"}

# Word tokens (letters/digits) are lemmatized + indexed; standalone punctuation is not.
_HAS_WORDCHAR = re.compile(r"[0-9A-Za-z֐-׿]")
_HEB_LETTER = re.compile(r"[֐-׿]")


def _is_valid_lemma(s: str) -> bool:
    """A real word lemma: >=2 chars, has a Hebrew letter, no digit."""
    return len(s) >= 2 and bool(_HEB_LETTER.search(s)) and not any(c.isdigit() for c in s)


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
        # membership set over bank lemmas: a token is only a real dictionary GAP worth
        # annotating if the lemma we predicted for it is not already in the bank.
        self._bank_lemma_set = frozenset(self.bank.lemmas)
        # Suppletive gate: (surface, POS) -> lemma for forms the coverage gate can't
        # trust (היא->הוא, זאת->זה, etc.). Prefer model_dir copy (may be more recent);
        # fall back to the package-bundled CSV.
        sup = Path(suppletives_path) if suppletives_path else Path(model_dir) / "suppletives.csv"
        if not sup.exists():
            sup = _PKG_DATA / "suppletives.csv"
        self.suppletive_gate = SuppletiveGate(sup) if sup.exists() else None
        # Acronym gate: set of known Hebrew acronym surface forms (Wiktionary, CC BY-SA).
        # When a form is a known acronym, skip the coverage/transduction check entirely —
        # the retrieval result stands and the token is labeled source="acronym".
        acr_path = Path(model_dir) / "wiktionary_acronyms.csv"
        if not acr_path.exists():
            acr_path = _PKG_DATA / "wiktionary_acronyms.csv"
        self._acronym_set: frozenset = frozenset()
        if acr_path.exists():
            import csv as _csv
            with open(acr_path, encoding="utf-8-sig", newline="") as fh:
                self._acronym_set = frozenset(
                    r["surface"].strip() for r in _csv.DictReader(fh) if r.get("surface", "").strip()
                )

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
        curated suppletive-lexicon lookup, score 1.0), "acronym" (a known Hebrew
        acronym, retrieval accepted without the coverage check), "transduced" (the
        edit-script fallback), or "function" (a closed-class word blanked because
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
                elif form in self._acronym_set:
                    lemma, source = form, "acronym"         # known acronym: lemma = surface form
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
        """Flag a token where retrieval is NOT trusted AND the result is a genuine,
        novel dictionary gap worth annotating (open-class, not already in the bank)."""
        cov = coverage(form, ret_lemma)
        cov_low, sim_low = cov < self.cov_thresh, ret_sim < self.min_sim
        if not (cov_low or sim_low):
            return
        if not self._worth_annotating(lemma, pos):
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

    # ---- document API: lemmatize_text (string / file / folder -> doc dict) ----
    _LEMMA_NOISE = str.maketrans("", "", "־- \t")
    _CLOSED_POS = frozenset({"ADP", "PRON", "DET", "CCONJ", "SCONJ", "AUX", "PART"})

    def _worth_annotating(self, lemma: str, pos: str = "") -> bool:
        """A predicted lemma is worth annotating only if it is a real, NOVEL, open-class
        dictionary gap: not closed-class, not already in the bank, and still a valid lemma
        once clitic maqaf/hyphen noise is stripped."""
        if pos in self._CLOSED_POS:
            return False
        if lemma in self._bank_lemma_set:
            return False
        return _is_valid_lemma(lemma.translate(self._LEMMA_NOISE))

    def _lemmatize_doc(self, text: str) -> Dict:
        """Lemmatize a document string into the doc-dict shape. Tokenization runs on the
        ORIGINAL text (doc_text) so token offsets index `text` verbatim (round-trip);
        lemmatize() normalizes internally."""
        from .text import collapse_gender_slash
        toks = [t for t in doc_tokenize(text) if _HAS_WORDCHAR.search(t.text)]
        sents = {s.id: s for s in split_sentences(text)}
        items = [{"form": collapse_gender_slash(t.text),
                  "sentence": (sents[t.sent_id].text if t.sent_id in sents else t.text)}
                 for t in toks]
        preds = self.lemmatize(items) if items else []

        tokens, es_tokens = [], []
        unknown: Dict[str, Dict] = {}
        pos_i = 0
        for t, p in zip(toks, preds):
            lemma = p["lemma"]
            tokens.append({"token": t.text, "start": t.start, "end": t.end,
                           "lemma": lemma, "pos": p["pos"], "source": p["source"],
                           "score": p["score"], "sent_id": t.sent_id})
            # blanked function words (source="function", lemma="") stay in tokens for
            # provenance but are not indexed.
            if lemma:
                es_tokens.append({"token": lemma, "start_offset": t.start,
                                  "end_offset": t.end, "position": pos_i, "type": "lemma"})
                pos_i += 1
            # `unknown` = real, novel dictionary gaps: the transduced fallback fired and the
            # predicted lemma is a genuine open-class gap.
            if p["source"] == "transduced" and self._worth_annotating(lemma, p["pos"]):
                u = unknown.get(t.text)
                if u is None:
                    unknown[t.text] = {"token": t.text, "lemma": lemma, "pos": p["pos"], "count": 1}
                else:
                    u["count"] += 1
        return {"text": text, "tokens": tokens,
                "analyzed_text": " ".join(tk["lemma"] for tk in tokens if tk["lemma"]),
                "es_tokens": es_tokens, "unknown": list(unknown.values())}

    def lemmatize_text(self, source, *, files_glob: str = "*.txt",
                       recursive: bool = True, verbose: bool = True):
        """Document/file/folder main call: lemmatize raw text with absolute character
        offsets + a pseudo-Elasticsearch `_analyze` token stream.

        `source` is a raw string, a file path, or a folder path:
          - a `Path` or a `str` naming an existing path is a path; anything else is RAW TEXT;
          - file -> one doc dict (with a `path` key);
          - dir  -> `{relative_path: doc dict}` for every text file (`files_glob`, recursive);
          - raw text -> one doc dict.

        Each doc dict: `text` (echoed input), `tokens` (offsets + lemma + pos + source +
        score + sent_id), `analyzed_text` (space-joined lemmas), `es_tokens` (ES-style
        stream), `unknown` (out-of-bank transduced tokens worth annotating)."""
        is_path = isinstance(source, Path) or (isinstance(source, str) and os.path.exists(source))
        if not is_path:
            return self._lemmatize_doc(str(source))
        p = Path(source)
        if p.is_dir():
            globber = p.rglob if recursive else p.glob
            files = [f for f in sorted(globber(files_glob)) if f.is_file()]
            out = {}
            for i, f in enumerate(files, 1):
                if verbose:
                    print(f"[shoshan] lemmatizing {i}/{len(files)}: {f}", flush=True)
                doc = self._lemmatize_doc(f.read_text(encoding="utf-8"))
                doc["path"] = str(f)
                out[str(f.relative_to(p))] = doc
            return dict(sorted(out.items()))
        doc = self._lemmatize_doc(p.read_text(encoding="utf-8"))
        doc["path"] = str(p)
        return doc
