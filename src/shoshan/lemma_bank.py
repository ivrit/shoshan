#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The lemma retrieval dictionary ("bank").

The bank is the heart of the no-hallucination guarantee: predictions are always
one of these lemmas. It is a plain, auditable artifact:
  - lemmas.csv : lemma, pos_tags (pipe-separated UPOS seen with this lemma), source
  - lemmas.npy : float32 [n_lemmas, dim] L2-normalized embeddings (built per encoder)

Extending the system = adding rows to lemmas.csv and re-encoding. No re-training.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Optional, Set
import numpy as np
import pandas as pd


class LemmaBank:
    def __init__(self, lemmas: List[str], pos_by_lemma: Optional[Dict[str, Set[str]]] = None,
                 source_by_lemma: Optional[Dict[str, str]] = None):
        # stable, de-duplicated order
        seen, ordered = set(), []
        for lm in lemmas:
            if lm and lm not in seen:
                seen.add(lm)
                ordered.append(lm)
        self.lemmas: List[str] = ordered
        self.index: Dict[str, int] = {lm: i for i, lm in enumerate(self.lemmas)}
        self.pos_by_lemma: Dict[str, Set[str]] = {k: set(v) for k, v in (pos_by_lemma or {}).items()}
        self.source_by_lemma: Dict[str, str] = dict(source_by_lemma or {})
        self.embeddings: Optional[np.ndarray] = None  # [n, dim], L2-normalized

    def __len__(self) -> int:
        return len(self.lemmas)

    # ---- candidate filtering for homographs -------------------------------
    def candidate_ids(self, pos: Optional[str] = None) -> Optional[np.ndarray]:
        """Indices of lemmas compatible with `pos`.

        Returns None (= search all) when no POS is given or POS info is absent.
        Lemmas with unknown POS are always kept (never filtered out wrongly).
        """
        if not pos or not self.pos_by_lemma:
            return None
        ids = [i for i, lm in enumerate(self.lemmas)
               if (lm not in self.pos_by_lemma) or (pos in self.pos_by_lemma[lm])]
        if not ids or len(ids) == len(self.lemmas):
            return None
        return np.asarray(ids, dtype=np.int64)

    # ---- persistence -------------------------------------------------------
    def save(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        rows = [{
            "lemma": lm,
            "pos_tags": "|".join(sorted(self.pos_by_lemma.get(lm, set()))),
            "source": self.source_by_lemma.get(lm, ""),
        } for lm in self.lemmas]
        pd.DataFrame(rows).to_csv(out / "lemmas.csv", index=False, encoding="utf-8")
        if self.embeddings is not None:
            np.save(out / "lemmas.npy", self.embeddings.astype(np.float32))
        (out / "bank_meta.json").write_text(json.dumps({
            "n_lemmas": len(self.lemmas),
            "has_pos": bool(self.pos_by_lemma),
            "has_embeddings": self.embeddings is not None,
            "embedding_dim": int(self.embeddings.shape[1]) if self.embeddings is not None else None,
        }, ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, in_dir: str | Path) -> "LemmaBank":
        d = Path(in_dir)
        df = pd.read_csv(d / "lemmas.csv").fillna("")
        pos_by = {r["lemma"]: set(t for t in str(r["pos_tags"]).split("|") if t)
                  for _, r in df.iterrows()}
        src_by = {r["lemma"]: r.get("source", "") for _, r in df.iterrows()}
        bank = cls(df["lemma"].tolist(), pos_by, src_by)
        npy = d / "lemmas.npy"
        if npy.exists():
            bank.embeddings = np.load(npy)
        return bank

    # ---- embedding ---------------------------------------------------------
    def encode(self, model, batch: int = 256, device: str = "cpu") -> np.ndarray:
        """Encode all lemmas with a SentenceTransformer; cache + return."""
        emb = model.encode(self.lemmas, batch_size=batch, convert_to_numpy=True,
                           normalize_embeddings=True, device=device,
                           show_progress_bar=True)
        self.embeddings = np.asarray(emb, dtype=np.float32)
        return self.embeddings


def bank_from_processed(csv_paths: List[str | Path],
                        extra_lemma_files: Optional[List[str | Path]] = None) -> LemmaBank:
    """Build a bank from processed train/dev/test CSVs (+ optional lexicon lists).

    Treebank lemmas carry their observed UPOS tags (for POS filtering); external
    lexicon lemmas are added without POS (kept as universal candidates).
    """
    pos_by: Dict[str, Set[str]] = {}
    source_by: Dict[str, str] = {}
    order: List[str] = []

    for p in csv_paths:
        df = pd.read_csv(p).fillna("")
        lcol = "lemma" if "lemma" in df.columns else df.columns[1]
        pcol = next((c for c in ("pos", "upos", "POS", "UPOS") if c in df.columns), None)
        for _, r in df.iterrows():
            lm = str(r[lcol]).strip()
            if not lm:
                continue
            if lm not in pos_by:
                pos_by[lm] = set()
                source_by[lm] = "treebank"
                order.append(lm)
            if pcol and str(r[pcol]).strip():
                pos_by[lm].add(str(r[pcol]).strip())

    for f in (extra_lemma_files or []):
        for line in Path(f).read_text(encoding="utf-8").splitlines():
            lm = line.strip()
            if not lm or lm.startswith("[") or len(lm) <= 1:
                continue
            if lm not in pos_by:
                pos_by[lm] = set()
                source_by[lm] = "lexicon"
                order.append(lm)

    return LemmaBank(order, pos_by, source_by)
