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
from typing import List, Tuple, Optional

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

    def encode_lemma(self, lemmas: List[str], max_len: int = 32) -> torch.Tensor:
        b = self.tok(lemmas, padding=True, truncation=True, max_length=max_len,
                     return_tensors="pt").to(self.device)
        out = self.enc(**b).last_hidden_state
        return F.normalize(masked_mean(out, b["attention_mask"]), dim=-1)

    def encode_query(self, sentences: List[str], spans: List[Tuple[int, int]],
                     max_len: int = 160):
        """Returns (q, pos_logits, edit_logits). edit_logits is None if no edit head."""
        b = self.tok(sentences, padding=True, truncation=True, max_length=max_len,
                     return_offsets_mapping=True, return_tensors="pt")
        offsets = b.pop("offset_mapping")
        b = {k: v.to(self.device) for k, v in b.items()}
        out = self.enc(**b).last_hidden_state
        B, T, _ = out.shape
        span_mask = torch.zeros(B, T, device=self.device)
        off = offsets.tolist()
        for i, (s, e) in enumerate(spans):
            for t in range(T):
                cs, ce = off[i][t]
                if cs == ce:
                    continue
                if cs < e and ce > s:
                    span_mask[i, t] = 1.0
            if span_mask[i].sum() == 0:
                span_mask[i, 0] = 1.0
        q = F.normalize(masked_mean(out, span_mask), dim=-1)
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
        (d / "pos_labels.json").write_text(json.dumps(UPOS, ensure_ascii=False))
        if self.edit_head is not None:
            np.save(d / "edit_weight.npy", self.edit_head.weight.detach().cpu().numpy().astype(np.float32))
            np.save(d / "edit_bias.npy", self.edit_head.bias.detach().cpu().numpy().astype(np.float32))
            (d / "scripts.json").write_text(json.dumps(self.scripts, ensure_ascii=False))
        (d / "joint_meta.json").write_text(json.dumps(
            {"backbone": self.backbone_name, "num_pos": len(UPOS),
             "num_scripts": len(self.scripts)}, ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, in_dir: str | Path, device: str = "cpu") -> "JointEncoder":
        d = Path(in_dir)
        meta = json.loads((d / "joint_meta.json").read_text())
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
            obj.scripts = json.loads((d / "scripts.json").read_text())
            obj.edit_head = nn.Linear(dim, len(obj.scripts))
            obj.edit_head.weight.data = torch.tensor(np.load(d / "edit_weight.npy"))
            obj.edit_head.bias.data = torch.tensor(np.load(d / "edit_bias.npy"))
        else:
            obj.scripts = []
            obj.edit_head = None
        return obj.to(device).eval()
