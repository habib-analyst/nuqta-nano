"""Nuqta-Nano test suite.

Pins the things that must not silently change: the metric definitions,
the vocabulary contract, the packaged model's parameter count, the
reproduced headline metrics (fixed 500-input subset), and one pinned
transliteration so a weights swap fails loudly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / "src"))
sys.path.insert(0, str(PKG / "scripts"))

from nuqta_nano import (  # noqa: E402
    Transliterator, alignment_ops, edit_distance, group_refs, score_predictions,
)
from nuqta_nano.vocab import VocabPair  # noqa: E402

# Filled from the evaluated model's actual output (see REPORT.md):
PINNED = {"mohabbat": "محبت"}


def test_edit_distance_known_values():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("", "") == 0
    assert edit_distance("", "abc") == 3
    assert edit_distance("abc", "") == 3
    assert edit_distance("محبت", "محبت") == 0
    assert edit_distance("کتاب", "کتب") == 1


def test_alignment_ops_consistent_with_distance():
    for pred, ref in [("abc", "axc"), ("", "ab"), ("abcd", "ac"), ("محبت", "محبت")]:
        subs, dels, ins = alignment_ops(pred, ref)
        assert len(subs) + len(dels) + len(ins) == edit_distance(pred, ref)


def test_group_refs_and_multi_ref_scoring():
    refs = group_refs([("a", "x"), ("a", "y"), ("a", "x"), ("b", "z")])
    assert refs["a"] == ["x", "y"] and list(refs) == ["a", "b"]
    out = score_predictions({"a": "y", "b": "w"}, refs)
    assert out["n_inputs"] == 2
    assert out["exact_match"] == 0.5          # 'y' matches via the second reference
    assert out["corpus_cer"] == 0.5           # 1 edit over 2 reference chars


def test_vocab_round_trip():
    vocabs = VocabPair.load(PKG / "vocab.json")
    lines = (PKG / "data/train.tsv").read_text(encoding="utf-8").splitlines()[:200]
    for line in lines:
        roman, urdu = line.split("\t")
        assert vocabs.src.decode(vocabs.src.encode(roman)) == roman
        assert vocabs.tgt.decode(vocabs.tgt.encode(urdu)) == urdu
    assert len(vocabs.src) == 30 and len(vocabs.tgt) == 58


def _transliterator() -> Transliterator:
    return Transliterator.from_dir(PKG)


def test_model_loads_and_param_count():
    cfg = json.loads((PKG / "config.json").read_text(encoding="utf-8"))
    trans = _transliterator()
    n = trans.model.count_parameters()
    assert n == cfg["parameters"]
    assert 2_000_000 <= n <= 8_000_000


def test_deterministic_transliteration():
    trans = _transliterator()
    first = trans.transliterate("mohabbat")
    second = trans.transliterate("mohabbat")
    assert first == second and first != ""
    tgt_chars = set(trans.vocabs.tgt.chars)
    assert set(first) <= tgt_chars


def test_pinned_transliteration():
    trans = _transliterator()
    for roman, expected in PINNED.items():
        assert trans.transliterate(roman) == expected


def test_metric_reproduction_subset500():
    from evaluate import load_pairs, predict_all

    recorded = json.loads((PKG / "assets/eval_results.json").read_text(encoding="utf-8"))
    trans = _transliterator()
    refs = group_refs(load_pairs("test"))
    subset = list(refs.keys())[:500]
    preds = predict_all(trans, subset)
    scores = score_predictions(preds, {k: refs[k] for k in subset})
    assert scores["corpus_cer"] == recorded["test_subset500"]["corpus_cer"]
    assert scores["exact_match"] == recorded["test_subset500"]["exact_match"]
