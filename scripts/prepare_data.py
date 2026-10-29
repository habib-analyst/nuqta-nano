"""Prepare Nuqta-Nano's packaged data from the Dakshina v1.0 Urdu files.

Input: the extracted dataset in the build work directory
(~/workspace/earning/models/nuqta-nano-work/dakshina_dataset_v1.0/ur/),
downloaded as the official tarball
https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar
from https://github.com/google-research-datasets/dakshina (CC BY-SA 4.0).

Output (in the package):
  data/train.tsv / data/dev.tsv / data/test.tsv   roman<TAB>urdu lexicon pairs
  data/sentences_test.tsv                          native<TAB>roman sentence pairs
  vocab.json                                       char vocabularies (train split only)
  assets/data_stats.json                           every count printed below

The Dakshina lexicon files are (urdu, romanization, attestation_count)
rows, already partitioned into train/dev/test by Urdu word type AND
lemma (see the dataset README), so dev/test Urdu words were never seen
in training. This script only inverts the columns, NFC-normalizes, drops
the attestation counts (each unique pair is used once), and reports.
"""

from __future__ import annotations

import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

WORK = Path.home() / "workspace/earning/models/nuqta-nano-work/dakshina_dataset_v1.0/ur"
PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / "src"))

from nuqta_nano.vocab import VocabPair, nfc  # noqa: E402


def load_lexicon(path: Path):
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            urdu, roman = nfc(parts[0]), nfc(parts[1]).lower()
            pairs.append((roman, urdu))
    return pairs


def main() -> None:
    stats = {}
    all_pairs = {}
    for split in ("train", "dev", "test"):
        pairs = load_lexicon(WORK / f"lexicons/ur.translit.sampled.{split}.tsv")
        all_pairs[split] = pairs
        out = PKG / f"data/{split}.tsv"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(f"{r}\t{u}\n" for r, u in pairs), encoding="utf-8")
        refs = {}
        for r, u in pairs:
            refs.setdefault(r, set()).add(u)
        stats[split] = {
            "pairs": len(pairs),
            "urdu_word_types": len({u for _, u in pairs}),
            "roman_inputs": len(refs),
            "roman_inputs_with_multiple_refs": sum(1 for v in refs.values() if len(v) > 1),
            "max_refs_for_one_input": max(len(v) for v in refs.values()),
        }

    # Split hygiene, verified rather than assumed:
    urdu_types = {s: {u for _, u in all_pairs[s]} for s in all_pairs}
    stats["urdu_type_overlap"] = {
        "train_dev": len(urdu_types["train"] & urdu_types["dev"]),
        "train_test": len(urdu_types["train"] & urdu_types["test"]),
        "dev_test": len(urdu_types["dev"] & urdu_types["test"]),
    }
    roman_inputs = {s: {r for r, _ in all_pairs[s]} for s in all_pairs}
    stats["roman_input_overlap"] = {
        "train_test": len(roman_inputs["train"] & roman_inputs["test"]),
        "note": "roman strings may recur across splits attached to different Urdu words; "
                "the held-out guarantee is on Urdu word types + lemmata (Dakshina's partition)",
    }

    # Romanized sentence test pairs (for the secondary, in-context eval).
    nat = (WORK / "romanized/ur.romanized.rejoined.test.native.txt").read_text(encoding="utf-8").splitlines()
    rom = (WORK / "romanized/ur.romanized.rejoined.test.roman.txt").read_text(encoding="utf-8").splitlines()
    assert len(nat) == len(rom), (len(nat), len(rom))
    sent_path = PKG / "data/sentences_test.tsv"
    sent_path.write_text(
        "".join(f"{nfc(n)}\t{nfc(r)}\n" for n, r in zip(nat, rom)), encoding="utf-8"
    )
    stats["sentences_test"] = {"sentence_pairs": len(nat)}

    # Vocabularies from the TRAIN split only.
    vocabs = VocabPair.build(
        [r for r, _ in all_pairs["train"]], [u for _, u in all_pairs["train"]]
    )
    vocabs.save(PKG / "vocab.json")
    stats["vocab"] = {
        "src_chars": len(vocabs.src.chars), "src_size": len(vocabs.src),
        "tgt_chars": len(vocabs.tgt.chars), "tgt_size": len(vocabs.tgt),
        "tgt_charset": "".join(vocabs.tgt.chars),
    }
    # Characters in dev/test targets or sources never seen in train (encoded as UNK):
    for split in ("dev", "test"):
        stats[split]["unseen_src_chars"] = sorted(
            {c for r, _ in all_pairs[split] for c in r} - set(vocabs.src.chars)
        )
        stats[split]["unseen_tgt_chars"] = sorted(
            {c for _, u in all_pairs[split] for c in u} - set(vocabs.tgt.chars)
        )

    (PKG / "assets").mkdir(exist_ok=True)
    (PKG / "assets/data_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
