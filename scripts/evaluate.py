"""Full honest evaluation of the packaged Nuqta-Nano weights.

Loads model.safetensors from the package root and computes, on data the
model never trained on:

  * test lexicon (primary): corpus CER + exact-match under the
    multi-reference protocol, on Dakshina's held-out test split
    (Urdu word types + lemmata disjoint from training);
  * dev lexicon: same metrics (the model-selection split);
  * a fixed 500-input test subset record, used by the test suite for
    exact metric reproduction;
  * verbatim samples: the first 12 test inputs in file order (not
    cherry-picked) and the 8 worst-CER failures (labeled as such);
  * a character-level error analysis over optimal alignments —
    which Urdu letters get substituted, dropped, or hallucinated;
  * a secondary in-context check: word-by-word transliteration of the
    romanized test sentences, scored per token against the native
    sentences where token counts align.

Writes assets/eval_results.json. Every number in README.md comes from
this file's output.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / "src"))

from nuqta_nano.infer import Transliterator  # noqa: E402
from nuqta_nano.metrics import alignment_ops, edit_distance, group_refs, score_predictions  # noqa: E402

BATCH = 512


def load_pairs(split: str):
    rows = []
    for line in (PKG / f"data/{split}.tsv").read_text(encoding="utf-8").splitlines():
        r, u = line.split("\t")
        rows.append((r, u))
    return rows


def predict_all(trans: Transliterator, inputs: list[str]) -> dict:
    preds = {}
    for i in range(0, len(inputs), BATCH):
        chunk = inputs[i : i + BATCH]
        for w, out in zip(chunk, trans.transliterate_words(chunk)):
            preds[w] = out
    return preds


def per_input_cer(pred: str, targets: list[str]) -> float:
    best = min(targets, key=lambda ref: edit_distance(pred, ref))
    return edit_distance(pred, best) / max(1, len(best))


def main() -> None:
    trans = Transliterator.from_dir(PKG)
    results = {"model": "nuqta-nano", "version": "0.1.0"}

    split_scores = {}
    test_preds = test_refs = None
    for split in ("dev", "test"):
        pairs = load_pairs(split)
        refs = group_refs(pairs)
        preds = predict_all(trans, list(refs.keys()))
        scores = score_predictions(preds, refs)
        split_scores[split] = scores
        print(f"{split}: CER {scores['corpus_cer']:.4f}  exact {scores['exact_match']:.4f}  "
              f"mean CER {scores['mean_cer']:.4f}  n={scores['n_inputs']}")
        if split == "test":
            test_preds, test_refs = preds, refs
    results["dev"] = split_scores["dev"]
    results["test"] = split_scores["test"]

    # Fixed subset for the test suite's exact-reproduction check.
    subset_inputs = list(test_refs.keys())[:500]
    sub_refs = {k: test_refs[k] for k in subset_inputs}
    results["test_subset500"] = score_predictions(
        {k: test_preds[k] for k in subset_inputs}, sub_refs
    )

    # Verbatim samples — first 12 in file order (not cherry-picked).
    first12 = list(test_refs.keys())[:12]
    results["samples_first12"] = [
        {"input": w, "prediction": test_preds[w], "references": test_refs[w]} for w in first12
    ]
    # Worst failures — selected BY the metric, labeled as worst-case.
    ranked = sorted(
        test_refs.keys(), key=lambda w: (-per_input_cer(test_preds[w], test_refs[w]), w)
    )
    worst = [w for w in ranked if per_input_cer(test_preds[w], test_refs[w]) > 0][:8]
    results["samples_worst8"] = [
        {"input": w, "prediction": test_preds[w], "references": test_refs[w],
         "cer": round(per_input_cer(test_preds[w], test_refs[w]), 3)} for w in worst
    ]

    # Character-level error analysis over closest-reference alignments.
    subs, dels, ins = Counter(), Counter(), Counter()
    diacritic_exact = [0, 0]  # [correct, total] for refs containing vowel diacritics
    diacritics = set("ًٌٍَُِّْٰ")
    len_buckets: dict[str, list[float]] = {}
    for w, targets in test_refs.items():
        pred = test_preds[w]
        ref = min(targets, key=lambda r: edit_distance(pred, r))
        s, d, i2 = alignment_ops(pred, ref)
        subs.update(s); dels.update(d); ins.update(i2)
        has_diac = any(c in diacritics for c in ref)
        if has_diac:
            diacritic_exact[1] += 1
            diacritic_exact[0] += int(pred in targets)
        bucket = "1-3" if len(w) <= 3 else "4-6" if len(w) <= 6 else "7-9" if len(w) <= 9 else "10+"
        len_buckets.setdefault(bucket, []).append(per_input_cer(pred, targets))
    results["error_analysis"] = {
        "top_substitutions_ref_to_pred": [[f"{a}->{b}", c] for (a, b), c in subs.most_common(10)],
        "top_deleted_ref_chars": [[a, c] for a, c in dels.most_common(10)],
        "top_inserted_pred_chars": [[a, c] for a, c in ins.most_common(10)],
        "refs_with_diacritics": {"n": diacritic_exact[1],
                                 "exact_match": diacritic_exact[0] / max(1, diacritic_exact[1])},
        "mean_cer_by_input_length": {k: round(sum(v) / len(v), 4) for k, v in sorted(len_buckets.items())},
    }

    # Secondary: word-by-word over the romanized test sentences.
    sent_rows = []
    for line in (PKG / "data/sentences_test.tsv").read_text(encoding="utf-8").splitlines():
        nat, rom = line.split("\t")
        sent_rows.append((nat, rom))
    aligned_pairs = 0
    positions = []  # (native_token, roman_token) to score
    for nat, rom in sent_rows:
        ntoks, rtoks = nat.split(), rom.split()
        if len(ntoks) != len(rtoks):
            continue
        aligned_pairs += 1
        for ntok, rtok in zip(ntoks, rtoks):
            if rtok.isalpha() and rtok.isascii():
                positions.append((ntok, rtok))
    unique_tokens = sorted({rtok for _, rtok in positions})
    token_preds = predict_all(trans, unique_tokens)
    scored_tokens = exact_tokens = 0
    tok_edits = tok_ref_chars = 0
    for ntok, rtok in positions:
        pred = token_preds[rtok]
        scored_tokens += 1
        exact_tokens += int(pred == ntok)
        tok_edits += edit_distance(pred, ntok)
        tok_ref_chars += len(ntok)
    results["sentences_test_word_by_word"] = {
        "sentence_pairs": len(sent_rows),
        "token_count_aligned_pairs": aligned_pairs,
        "scored_tokens": scored_tokens,
        "token_exact_match": exact_tokens / max(1, scored_tokens),
        "token_corpus_cer": tok_edits / max(1, tok_ref_chars),
        "note": "secondary check: words transliterated in isolation, scored against the "
                "native sentence's token at the same position; only roman tokens that "
                "are pure a-z are scored",
    }
    print(f"sentences: token exact {results['sentences_test_word_by_word']['token_exact_match']:.4f} "
          f"CER {results['sentences_test_word_by_word']['token_corpus_cer']:.4f} "
          f"over {scored_tokens} tokens")

    (PKG / "assets/eval_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print("wrote assets/eval_results.json")


if __name__ == "__main__":
    main()
