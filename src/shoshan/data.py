#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Load the Shoshan lemmatization data on demand from the Hugging Face Hub.

The CSVs are not shipped in the package; they are pulled from the dataset repo the
first time you ask for a split and cached locally after that (no manual download,
no install-time fetch). Columns: form, lemma, pos, sentence, source, sent_id.

    from shoshan import data
    df = data.load("ood")     # -> pandas DataFrame
    data.splits()             # -> list of split names
"""

from __future__ import annotations

DATASET_REPO = "noamor/shoshan-data"

_FILES = {
    "train": "train.csv", "dev": "dev.csv", "test": "test.csv",
    "ood": "ood.csv", "oov": "oov.csv",
    "ood_bagatz": "ood_Bagatz.csv", "ood_geektime": "ood_GeekTime.csv",
    "ood_dicta": "ood_Dicta.csv",
}


def splits():
    """Names accepted by load()."""
    return sorted(_FILES)


def load(split: str = "ood", repo: str = DATASET_REPO):
    """Download (cached) and return one split as a pandas DataFrame.

    `split` is one of splits(), or any ".csv" filename in the dataset repo.
    """
    import pandas as pd
    from huggingface_hub import hf_hub_download
    key = split.lower()
    fname = _FILES.get(key, split if split.endswith(".csv") else f"{split}.csv")
    path = hf_hub_download(repo_id=repo, filename=fname, repo_type="dataset")
    return pd.read_csv(path)
