# Changelog

## 0.1.0 — 2026-10-29

Initial release.

- Character-level Transformer encoder-decoder (3,410,234 parameters),
  implemented and trained from scratch in PyTorch — no pretrained
  weights, no fine-tuning.
- Direction: Roman Urdu → Urdu script (word level; multi-word input is
  transliterated word by word).
- Trained only on the Dakshina dataset's Urdu lexicon training split
  (106,260 pairs, CC BY-SA 4.0); model selected by dev-split CER;
  headline metrics measured once on the held-out test split and
  reproduced from the packaged weights by `examples/verify_metrics.py`.
- Packaged with the exact training script, data-preparation script,
  evaluation script, vocabularies, and the prepared split files so the
  whole pipeline can be rerun end to end.
