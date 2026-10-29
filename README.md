---
license: mit
pipeline_tag: translation
language: ur
tags:
  - transliteration
  - urdu
  - roman-urdu
  - seq2seq
  - transformer
  - dakshina
  - pytorch
  - safetensors
---

# Nuqta-Nano — Roman Urdu → Urdu script transliteration

**A tiny character-level Transformer (3,410,234 parameters), designed and trained completely from scratch to transliterate Roman Urdu into Urdu script — trained only on the openly licensed Dakshina dataset, and evaluated on words it has provably never seen.**

Roman Urdu is how millions of people actually type Urdu; proper Urdu script is what publishing, education, and formal communication need. Nuqta-Nano converts the first into the second, one word at a time:

```python
from nuqta_nano import Transliterator

t = Transliterator.from_dir(".")          # this repo
t.transliterate("mohabbat")               # -> محبت
t.transliterate("yeh kitab meri hai")     # -> یہ کتاب میری ہائی  (last word wrong — see below)
```

## Headline metrics — reproduced, not just claimed

Measured **once** on Dakshina's held-out **test** split (10,327 distinct roman inputs; the split's Urdu word types *and lemmata* are disjoint from training — verified overlap: **0**), under a multi-reference protocol (a prediction is exact if it matches any attested spelling for that input; CER is against the closest attested spelling):

| Metric | Test (held out) | Dev (selection split) |
|---|---|---|
| Corpus CER ↓ | **0.2020** | 0.1970 |
| Exact-match accuracy ↑ | **0.3606** | 0.3756 |
| Mean per-word CER ↓ | 0.2194 | 0.2121 |

Every number was re-derived from the shipped `model.safetensors` by `examples/verify_metrics.py` and matched exactly — run it yourself. Context for reading them: the average test word is ~5 letters, so CER 0.20 is roughly **one wrong letter per word** on words the model has never seen in any form; and the mapping is genuinely many-to-many (9,554 training inputs attest more than one correct spelling).

A secondary, in-context check — transliterating the romanized test *sentences* word by word — scores token exact-match **0.6352** / token CER **0.1699** over 51,698 tokens. That split is *not* held out by word type (running text is mostly frequent words), which is why it is reported second, not first.

## What the model actually outputs (verbatim, unedited)

**First 12 test inputs in file order** (the test file is alphabetized, so these are all spellings of one word family — آئین "constitution". Disclosed because it makes this block unrepresentative *and* genuinely hard: vowel-onset ambiguity):

| input | model | attested |
|---|---|---|
| aaieen | آئیں | آئین |
| aaoin | آوئیں | آئین |
| aayean | آئیاں | آئین |
| aayeen | آئیں | آئین |
| aayin | آئیں | آئین |
| aaeeni | آئینی | آئینی |
| aaini | آئنی | آئینی |
| aayinee | آئنی | آئینی |
| aayini | آئنی | آئینی |
| aayiny | آئینی | آئینی |
| ainey | اینے | آئینی |
| aini | اینی | آئینی |

**Worst 8 failures in the whole test set** (selected by CER, labeled as worst-case):

| input | model | attested |
|---|---|---|
| haa | حا | ھ |
| he | ہے | ھ |
| hey | ہے | ھ |
| monedou | مونیدو | عمر |
| mukeish | مکیش | نل |
| tay | تے | ٹ |
| tey | تے | ٹ |
| amiry | امیری | ہون |

**Demo probes** (fixed common words, fixed before evaluation): mohabbat→محبت, shukriya→شکریہ, pakistan→پاکستان, zindagi→زندگی, khubsurat→خوبصورت, duniya→دنیا, mushkil→مشکل (unseen in training, correct), waqt→وقت, kitab→کتاب, mohalla→محلہ, taqreeb→تقریب — and the failures: mehman→محمان (should be مہمان), sukhan→سوکھن (should be سخن), and in the sentence demo, hai→ہائی (should be ہے). Full output in `example_output.txt`.

## Systematic failure modes (measured over the full test set)

These come from optimal-alignment analysis of every test prediction (`assets/eval_results.json`), not from impressions:

- **ں vs ن (top substitution: ن predicted as ں, 238 times).** Word-final nasalization is barely marked in romanization; the model over-applies ں and guesses elsewhere.
- **Retroflexes ٹ/ڈ/ڑ vs dentals ت/د/ر.** ٹ→ت 175, ت→ٹ 184, ڈ→د 141. Roman `t`/`d` do not say which one; the model is right only when context gives it away.
- **The ہ / ح / ھ / ع / ا cluster.** ہ→ح 150, ہ→ا 147, ع→ا 167, ا deleted 1,008 times (the largest single error class), ھ deleted 75 times — and the three worst failures in the test set are the single letter **ھ** itself, which the model never produces alone.
- **Diacritics are effectively not predicted.** Only 483 test references contain vowel marks (zabar/zer/pesh/shadda); exact match on them is **2.7%**. Romanization carries almost no signal for them, and the model has learned to omit them — matching how Urdu is usually written, but wrong against these references.
- **Short words are the hardest.** Mean CER by input length: 1–3 letters **0.42**, 4–6: 0.26, 7–9: 0.17, 10+: 0.13. A 1-character reference scored against a 2-character prediction is CER 2.0 — several worst cases are exactly that.

## Architecture & training

- Standard Transformer encoder–decoder, character level on both sides: 4 + 4 layers, d_model 192, 4 heads, d_ff 512, dropout 0.1, learned positional embeddings. **3,410,234 parameters.** No pretrained anything.
- Vocabularies are the training split's characters: 26 roman letters (+4 specials), 54 Urdu characters (+4 specials). Greedy decoding, fully deterministic.
- Training data: Dakshina Urdu **lexicon train split only** — 106,260 (romanization → Urdu word) pairs. Dev split used only to select the checkpoint (best dev CER, epoch 12 of 12); test split touched once, by the evaluation script.
- Recipe: AdamW (0.9/0.98), peak LR 1e-3 with warmup + cosine, batch 512, cross-entropy, seed 20261029, CPU-only (2 threads). Exact scripts: `scripts/train.py`, `scripts/resume_train.py`, `scripts/prepare_data.py`.
- **Incident, disclosed:** the first training run was killed by a VM reboot after epoch 4 (best checkpoint survived intact: dev CER 0.2231). Training was resumed from those weights with a fresh optimizer and a restarted cosine schedule (peak 7e-4, epochs 5–12). Both runs' logs are in the repo (`training_log.txt`, `training_log_resumed.txt`); total training compute ≈ 2.1 hours on a shared 2-core VM. The final weights are the resumed run's epoch 12.

## Data & licensing

- Sole source: **Dakshina dataset v1.0** (Google Research; Roark et al., LREC 2020), **CC BY-SA 4.0**. Full provenance — exact files, sizes, preparation steps, verified split hygiene — is in `DATA.md`. Nothing was scraped; no other pairs were used.
- Code and weights packaging: MIT (`LICENSE`). The packaged data splits remain CC BY-SA 4.0 with Dakshina attribution; weights were trained exclusively on that data and carry the attribution note in `DATA.md`.

## Limitations (stated, not hidden)

- **Word-level only.** Sentences are handled by transliterating each word independently — no context crosses word boundaries, so homographs that context would disambiguate are guessed from the word alone.
- **Unseen-word CER is 0.20.** This is a tiny educational/showcase model trained on ~106k word pairs, not a production transliterator. Dakshina's own paper notes human round-trip transliteration itself carries 3.5–8.5% character error at sentence level — the mapping is lossy — but this model's errors are well above that floor.
- **Informal Roman Urdu will degrade it.** The training spellings are annotator-attested forms from Wikipedia vocabulary. Chat-style spellings that drop vowels aggressively (e.g. "mujhy" written "mjy") are out of distribution.
- **English loanwords and named entities** written in ad-hoc romanization are unreliable (see `monedou` → مونیدو for عمر in the worst-8).
- One training seed; no seed-variance estimate. No beam search (greedy only, by design, for determinism).

## Verify it yourself

```bash
pip install -e ".[dev]"
PYTHONPATH=src python examples/verify_metrics.py   # re-derives every headline number
PYTHONPATH=src python -m pytest -q                  # 8 tests, incl. metric reproduction on a fixed subset
PYTHONPATH=src python examples/transliterate_demo.py
```

## Citation

If you use the data, cite Dakshina:

```bibtex
@inproceedings{roark-etal-2020-processing,
  title = "Processing {South} {Asian} Languages Written in the {Latin} Script: the {Dakshina} Dataset",
  author = "Roark, Brian and Wolf-Sonkin, Lawrence and Kirov, Christo and Mielke, Sabrina J. and Johny, Cibu and Demir{\c{s}}ahin, I{\c{s}}in and Hall, Keith",
  booktitle = "Proceedings of The 12th Language Resources and Evaluation Conference (LREC)",
  year = "2020",
  pages = "2413--2423"
}
```
