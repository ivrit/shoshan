#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fetch the Shoshan weights from the Hugging Face Hub.

The model repo holds two folders: ``model/`` (the encoder, the POS and edit-script
heads, and the script inventory) and ``bank/`` (the pre-encoded lemma bank). The
first call downloads a snapshot to the local Hugging Face cache and returns its
path; later calls reuse the cache and make no network requests.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

DEFAULT_REPO = "noamor/shoshan"


def download_weights(repo: str = DEFAULT_REPO, revision: Optional[str] = None) -> Path:
    """Download (or reuse the cached) weights and return the snapshot directory.

    The returned path contains ``model/`` and ``bank/`` subfolders.
    """
    from huggingface_hub import snapshot_download
    local = snapshot_download(repo_id=repo, revision=revision)
    return Path(local)
