#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shoshan — a zero-hallucination Hebrew lemmatizer (retrieve, then transduce)."""

from .infer import Lemmatizer
from .hub import DEFAULT_REPO, download_weights
from .runtime import detect, configure
from . import data

__version__ = "0.2.0"
__all__ = ["Lemmatizer", "DEFAULT_REPO", "download_weights", "detect", "configure",
           "data", "__version__"]
