---
title: Shoshan
emoji: 🌷
colorFrom: pink
colorTo: purple
sdk: gradio
sdk_version: 5.49.1
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
models:
- noamor/shoshan
---

# Shoshan — Hebrew lemmatizer

Paste Hebrew text and get each word's lemma (dictionary form). Shoshan retrieves
the lemma from a fixed bank and, for unknown words, derives it by editing the
word itself — so it never invents a word.

Code: <https://github.com/ivrit/shoshan> · Weights: <https://huggingface.co/noamor/shoshan>
