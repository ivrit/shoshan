#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re-normalize + clean + re-encode the lemma bank.

The shipped bank carries niqqud-vocalized lemma variants (אֶל / אַל / אל) that DictaBERT's
tokenizer strips to the same subwords — so they encode to the same vector and retrieval
among them is an arbitrary tie. It also carries non-Hebrew junk lemmas (foreign words that
leaked from the treebank). This script folds every lemma to its canonical undotted key with
`normalize_lemma`, DROPS keys that aren't valid Hebrew lemmas (`is_valid_lemma`), MERGES rows
that collide after folding, re-encodes with the current encoder, and writes a fresh bank.

Merge semantics (LemmaBank de-dups by FIRST occurrence and DROPS later duplicates' POS/source,
so the merge is done HERE, before constructing it):
  - pos_tags : UNION of the |-separated UPOS sets of all colliding source rows.
  - source   : UNION of the +-joined source tokens, sorted, re-joined with '+'.
Order is first-seen order of the normalized lemma (stable, auditable). `model/` (encoder +
heads) is untouched — only the bank changes.

  python scripts/rebuild_bank.py --out build/bank_clean          # uses cached HF weights
  python scripts/rebuild_bank.py --in-bank <dir> --model <dir> --out <dir> --device cpu
"""
import argparse
import os
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from shoshan.normalize import normalize_lemma, is_valid_lemma  # noqa: E402


def _split_set(s: str, sep: str):
    return [t for t in str(s).split(sep) if t]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-bank", default=None, help="dir with the input lemmas.csv (default: cached HF bank)")
    ap.add_argument("--model", default=None, help="JointEncoder dir (default: cached HF model)")
    ap.add_argument("--out", required=True, help="output bank dir")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--repo", default="HebArabNlpProject/shoshan", help="HF weights repo for defaults")
    args = ap.parse_args()

    # Resolve defaults from the cached weights snapshot (offline-friendly).
    if args.in_bank is None or args.model is None:
        from shoshan.hub import download_weights
        root = download_weights(args.repo)
        in_bank = Path(args.in_bank) if args.in_bank else root / "bank"
        model_dir = Path(args.model) if args.model else root / "model"
    else:
        in_bank, model_dir = Path(args.in_bank), Path(args.model)

    df = pd.read_csv(in_bank / "lemmas.csv", encoding="utf-8").fillna("")
    n_in = len(df)

    # Fold to the canonical undotted key; drop invalid keys; merge colliders (union POS/source),
    # preserving first-seen order of the normalized lemma.
    pos_by: "OrderedDict[str, set]" = OrderedDict()
    src_by: dict = {}
    members: dict = {}          # normalized lemma -> count of source rows that map to it
    n_dropped = 0
    dropped_samples = []

    for raw_lemma, raw_pos, raw_src in zip(df.lemma.astype(str),
                                           df.pos_tags.astype(str),
                                           df.source.astype(str)):
        key = normalize_lemma(raw_lemma)
        if not is_valid_lemma(key):        # non-Hebrew junk, single-letter, digits
            n_dropped += 1
            if len(dropped_samples) < 15:
                dropped_samples.append(raw_lemma)
            continue
        if key not in pos_by:
            pos_by[key] = set(); src_by[key] = set(); members[key] = 0
        pos_by[key].update(_split_set(raw_pos, "|"))
        src_by[key].update(_split_set(raw_src, "+"))
        members[key] += 1

    lemmas = list(pos_by.keys())
    n_out = len(lemmas)
    n_groups = sum(1 for c in members.values() if c > 1)     # keys built from >=2 kept rows
    n_merged_away = (n_in - n_dropped) - n_out               # kept rows that collapsed into a collision

    # Re-encode with OUR encoder (encode_lemma = masked-mean + L2), batch 256. CPU ≈ 11 min.
    import torch
    from shoshan.lemma_bank import LemmaBank
    from shoshan.model_joint import JointEncoder

    enc = JointEncoder.load(model_dir, device=args.device); enc.eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(lemmas), 256):
            embs.append(enc.encode_lemma(lemmas[i:i + 256], max_len=32).cpu().numpy())
    embeddings = np.vstack(embs).astype(np.float32)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    bank = LemmaBank(lemmas, pos_by_lemma=pos_by,
                     source_by_lemma={lm: "+".join(sorted(src_by[lm])) for lm in lemmas})
    bank.embeddings = embeddings
    bank.save(out)

    print(f"input rows:        {n_in:,}")
    print(f"dropped (invalid): {n_dropped:,}   e.g. {dropped_samples[:8]}")
    print(f"collision groups:  {n_groups:,}")
    print(f"rows merged away:  {n_merged_away:,}")
    print(f"output rows:       {n_out:,}")
    print(f"embedding shape:   {embeddings.shape}")
    print(f"wrote {out}/lemmas.csv + lemmas.npy + bank_meta.json")


if __name__ == "__main__":
    main()
