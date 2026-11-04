#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Joint encoder: lemma retrieval + POS head + edit-script head (see
docs/POS_DESIGN.md, docs/EDIT_SCRIPT_DESIGN.md).

One shared transformer encoder, three heads off the span-pooled token vector q:
  - lemma : cosine(q, lemma embeddings) over the bank
  - POS   : Linear(q) over 15 UPOS  (= dot-product vs 15 anchors)
  - edit  : Linear(q) over N learned, form-relative edit scripts (OOV-safe)

Span-pooling locates the target token by CHAR OFFSETS (no [FORM] markers in the
model input) and mean-pools its subword vectors (BLINK-style mention pooling).
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

UPOS = ["NOUN", "PROPN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP", "NUM",
        "AUX", "CCONJ", "SCONJ", "PART", "INTJ", "X"]
POS2ID = {p: i for i, p in enumerate(UPOS)}


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)


class JointEncoder(nn.Module):
    def __init__(self, backbone: str = "dicta-il/dictabert",
                 num_pos: int = len(UPOS), scripts: Optional[List[str]] = None):
        super().__init__()
        self.backbone_name = backbone
        self.tok = AutoTokenizer.from_pretrained(backbone)
        self.enc = AutoModel.from_pretrained(backbone)
        d = self.enc.config.hidden_size
        self.pos_head = nn.Linear(d, num_pos)
        self.scripts: List[str] = list(scripts) if scripts else []
        self.edit_head = nn.Linear(d, len(self.scripts)) if self.scripts else None

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def max_len(self) -> int:
        """Default input window of `encode_sentences`, in subword tokens.

        This is the encoder's own position limit (512 for the BERT backbone), not a smaller
        cap. Anything past the window is cut off, and a token that was cut off has no
        hidden states at all: `pool_spans` can only hand back the sentence-start vector for
        it, which none of the heads were trained to read (the POS head tends to call it
        PART, and `blank_function_words` would then drop the word as a stopword). An earlier
        default of 160 kept batches cheap and covered every training sentence (the longest
        was 127 subwords), but silently truncated longer real-world sentences. Tokens that
        fall past even this window are flagged by `pool_spans(return_truncated=True)` and
        reported by `Lemmatizer.lemmatize` as source="truncated"."""
        return int(getattr(self.enc.config, "max_position_embeddings", 512))

    def encode_lemma(self, lemmas: List[str], max_len: int = 32) -> torch.Tensor:
        b = self.tok(lemmas, padding=True, truncation=True, max_length=max_len,
                     return_tensors="pt").to(self.device)
        out = self.enc(**b).last_hidden_state
        return F.normalize(masked_mean(out, b["attention_mask"]), dim=-1)

    def encode_sentences(self, sentences: List[str], max_len: Optional[int] = None):
        """Run the encoder ONCE over each sentence (the expensive forward pass).

        Returns (hidden, offsets): `hidden` is last_hidden_state [B, T, D]; `offsets`
        is the list of B per-token (char_start, char_end) rows. Pooling is factored out
        into `pool_spans`, so a sentence encoded once here can serve many token spans —
        the encode-once path (`Lemmatizer.lemmatize`) encodes each DISTINCT sentence once
        and pools every token's span from the shared hidden state. `max_len` defaults to
        the encoder's position limit (see `max_len`)."""
        if max_len is None:
            max_len = self.max_len
        b = self.tok(sentences, padding=True, truncation=True, max_length=max_len,
                     return_offsets_mapping=True, return_tensors="pt")
        offsets = b.pop("offset_mapping").tolist()
        b = {k: v.to(self.device) for k, v in b.items()}
        return self.enc(**b).last_hidden_state, offsets

    def pool_spans(self, hidden: torch.Tensor, offsets, rows: List[int],
                   spans: List[Tuple[int, int]], return_truncated: bool = False):
        """Mean-pool each (row, span) from a shared encode → q [len(spans), D], L2-norm.

        `rows[k]` selects the sentence row in `hidden` for span `spans[k]=(s,e)`; many
        spans may share one row (encode-once). Reproduces `encode_query`'s span mask
        exactly: a subword token is included when its char span overlaps [s, e)
        (`cs < e and ce > s`). When the form was truncated out of the window (mask empty),
        position 0 ([CLS]) is pooled instead and the span is flagged: with
        `return_truncated=True` the result is `(q, truncated)`, `truncated[k]` True for
        exactly those spans. Callers must not read the heads on a flagged row as a
        prediction (see `max_len`). Spans that share a hidden row are pooled with a single
        [K, T]·[T, D] matmul (the shared row is never duplicated across spans)."""
        T, D = hidden.shape[1], hidden.shape[2]
        q = hidden.new_zeros(len(spans), D)
        truncated = [False] * len(spans)
        groups: Dict[int, List[int]] = {}
        for k, r in enumerate(rows):
            groups.setdefault(r, []).append(k)
        for r, ks in groups.items():
            off_r = offsets[r]
            m = hidden.new_zeros(len(ks), T)
            for j, k in enumerate(ks):
                s, e = spans[k]
                for t in range(T):
                    cs, ce = off_r[t]
                    if cs == ce:
                        continue
                    if cs < e and ce > s:
                        m[j, t] = 1.0
                if m[j].sum() == 0:
                    m[j, 0] = 1.0
                    truncated[k] = True
            pooled = (m @ hidden[r]) / m.sum(1, keepdim=True).clamp(min=1e-9)
            q[ks] = F.normalize(pooled, dim=-1)
        return (q, truncated) if return_truncated else q

    def encode_query(self, sentences: List[str], spans: List[Tuple[int, int]],
                     max_len: Optional[int] = None):
        """Returns (q, pos_logits, edit_logits). edit_logits is None if no edit head.

        1:1 sentences↔spans (one span per sentence). The runtime document path dedups
        sentences and calls encode_sentences + pool_spans directly (encode-once)."""
        hidden, offsets = self.encode_sentences(sentences, max_len)
        q = self.pool_spans(hidden, offsets, list(range(len(sentences))), spans)
        pos_logits = self.pos_head(q)
        edit_logits = self.edit_head(q) if self.edit_head is not None else None
        return q, pos_logits, edit_logits

    # ---- persistence -------------------------------------------------------
    def save(self, out_dir: str | Path):
        d = Path(out_dir); d.mkdir(parents=True, exist_ok=True)
        self.enc.save_pretrained(d / "encoder")
        self.tok.save_pretrained(d / "encoder")
        np.save(d / "pos_anchors.npy", self.pos_head.weight.detach().cpu().numpy().astype(np.float32))
        np.save(d / "pos_bias.npy", self.pos_head.bias.detach().cpu().numpy().astype(np.float32))
        (d / "pos_labels.json").write_text(json.dumps(UPOS, ensure_ascii=False), encoding="utf-8")
        if self.edit_head is not None:
            np.save(d / "edit_weight.npy", self.edit_head.weight.detach().cpu().numpy().astype(np.float32))
            np.save(d / "edit_bias.npy", self.edit_head.bias.detach().cpu().numpy().astype(np.float32))
            (d / "scripts.json").write_text(json.dumps(self.scripts, ensure_ascii=False), encoding="utf-8")
        (d / "joint_meta.json").write_text(json.dumps(
            {"backbone": self.backbone_name, "num_pos": len(UPOS),
             "num_scripts": len(self.scripts)}, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, in_dir: str | Path, device: str = "cpu") -> "JointEncoder":
        d = Path(in_dir)
        meta = json.loads((d / "joint_meta.json").read_text(encoding="utf-8"))
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.backbone_name = meta["backbone"]
        obj.tok = AutoTokenizer.from_pretrained(str(d / "encoder"))
        obj.enc = AutoModel.from_pretrained(str(d / "encoder"))
        dim = obj.enc.config.hidden_size
        obj.pos_head = nn.Linear(dim, meta["num_pos"])
        obj.pos_head.weight.data = torch.tensor(np.load(d / "pos_anchors.npy"))
        obj.pos_head.bias.data = torch.tensor(np.load(d / "pos_bias.npy"))
        if (d / "scripts.json").exists():
            obj.scripts = json.loads((d / "scripts.json").read_text(encoding="utf-8"))
            obj.edit_head = nn.Linear(dim, len(obj.scripts))
            obj.edit_head.weight.data = torch.tensor(np.load(d / "edit_weight.npy"))
            obj.edit_head.bias.data = torch.tensor(np.load(d / "edit_bias.npy"))
        else:
            obj.scripts = []
            obj.edit_head = None
        return obj.to(device).eval()
