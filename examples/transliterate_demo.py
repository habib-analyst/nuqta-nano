"""Transliteration demo on the packaged weights.

A fixed probe list of common Roman Urdu words. For each word the demo
prints the model's output and, truthfully labeled:
  - whether this exact roman string was a training input (seen), and
  - the attested Urdu form(s) if the word appears in the held-out test
    split's references.
Nothing is cherry-picked: the list was fixed before evaluation and the
outputs printed are whatever the model produces.
"""

from __future__ import annotations

import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / "src"))
sys.path.insert(0, str(PKG / "scripts"))

from evaluate import load_pairs  # noqa: E402
from nuqta_nano.infer import Transliterator  # noqa: E402
from nuqta_nano.metrics import group_refs  # noqa: E402

PROBES = [
    "mohabbat", "shukriya", "pakistan", "zindagi", "khubsurat", "mehman",
    "duniya", "mushkil", "waqt", "insaan", "kitab", "mohalla",
    "taqreeb", "chandni", "sukhan",
]


def main() -> None:
    trans = Transliterator.from_dir(PKG)
    train_inputs = {r for r, _ in load_pairs("train")}
    test_refs = group_refs(load_pairs("test"))
    for w in PROBES:
        out = trans.transliterate(w)
        seen = "seen in training" if w in train_inputs else "UNSEEN in training"
        attested = test_refs.get(w)
        note = f" | test references: {' / '.join(attested)}" if attested else ""
        print(f"{w:>10}  ->  {out}   [{seen}]{note}")
    print()
    print(trans.transliterate("yeh kitab meri hai"))


if __name__ == "__main__":
    main()
