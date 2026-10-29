# DATA.md — corpus card for Nuqta-Nano

## Source (the only source)

**Dakshina dataset v1.0** — Roark et al., *Processing South Asian
Languages Written in the Latin Script: the Dakshina Dataset*, LREC 2020.

- Project page / official source: https://github.com/google-research-datasets/dakshina
- Download used: `dakshina_dataset_v1.0.tar` (2,008,340,480 bytes) from
  `https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar`,
  fetched 2026-10-29 (Asia/Karachi). The tarball is not redistributed
  here; it is freely downloadable from that URL.
- **License: CC BY-SA 4.0** (dataset README, "License" section). The
  underlying text is derived from Wikipedia; the romanizations and
  lexicon annotations were produced by human annotators for the
  dataset. Attribution: this card, the README citation, and the
  packaged split files' provenance notes.

No other corpus was used. Nothing was scraped; no pair in the package
comes from anywhere but the files below.

## Exact files used (all inside the tarball under `dakshina_dataset_v1.0/ur/`)

| File | Bytes | Lines | Used for |
|---|---|---|---|
| `lexicons/ur.translit.sampled.train.tsv` | 2,185,405 | 106,260 | **training** (inverted) |
| `lexicons/ur.translit.sampled.dev.tsv` | 213,745 | 10,424 | model selection (dev CER) |
| `lexicons/ur.translit.sampled.test.tsv` | 216,098 | 10,517 | **final held-out evaluation only** |
| `romanized/ur.romanized.rejoined.test.native.txt` | 723,547 | 4,880 | secondary sentence-level check |
| `romanized/ur.romanized.rejoined.test.roman.txt` | 487,641 | 4,880 | secondary sentence-level check |

Dakshina lexicon rows are `(Urdu word, romanization, attestation
count)`. The lexicons are already partitioned into train/dev/test
**by Urdu word type and by lemma** (dataset README), i.e. no Urdu word
— and no lemma of one — appears in two splits.

## Preparation (`scripts/prepare_data.py`, rerunnable)

1. Invert each lexicon row to `(romanization, Urdu word)` — the
   model's direction — lowercase the romanization (the source is
   already all-lowercase; verified: the train romanizations use
   exactly the 26 letters a–z), NFC-normalize both sides, drop the
   attestation counts (every unique pair is used exactly once).
2. Write `data/train.tsv`, `data/dev.tsv`, `data/test.tsv`
   (roman TAB Urdu) and `data/sentences_test.tsv` (native TAB roman).
3. Build character vocabularies from the **train split only**:
   source = 26 letters (+4 specials = 30); target = 54 Urdu-script
   characters (+4 specials = 58). Dev/test contain **no character**
   outside these vocabularies (verified in `assets/data_stats.json`).

## Verified split hygiene (measured, not assumed)

- Urdu word-type overlap: train∩dev = 0, train∩test = 0, dev∩test = 0.
- The held-out guarantee is therefore on **unseen words**, which is
  the honest version of a transliteration eval: the model cannot have
  memorized any test word's spelling.
- 1,906 roman strings do recur between train and test — attached to
  *different* Urdu words (romanization is many-to-many). This is
  intrinsic ambiguity of the task, disclosed here rather than hidden;
  it cannot leak a test word's spelling, because the spelling is what
  is being predicted.

## Ambiguity (a property of the data, reported for honesty)

- Train: 93,621 distinct roman inputs for 25,000 Urdu word types;
  9,554 roman inputs attest more than one Urdu spelling (max 28).
- Evaluation therefore uses a multi-reference protocol: a prediction
  is exact if it matches **any** attested Urdu form for that input,
  and CER is measured against the closest attested form. See
  `src/nuqta_nano/metrics.py`.

## What is NOT in the data

- Word-level pairs only. The model never sees sentences in training;
  the romanized sentence files are used solely for the secondary
  word-by-word check reported in the README.
- Dakshina notes that Urdu romanization in the wild is inconsistent
  (short vowels often dropped, e.g. "km" for کم). The lexicon teaches
  the annotators' attested spellings; informal spellings far from
  those conventions will degrade, and the README says so.

## Licensing summary

- Code in this package: MIT (see `LICENSE`).
- Data (packaged splits, derived from Dakshina): **CC BY-SA 4.0**,
  attribution above; share-alike applies to redistributions of the
  data files.
- Weights: trained exclusively on the CC BY-SA data above. They are
  distributed with this attribution; users redistributing the weights
  should preserve the Dakshina attribution. (This mirrors the caveat
  recorded for Sukhan-Nano's weights.)
