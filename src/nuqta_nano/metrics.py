"""Levenshtein edit distance and transliteration metrics.

One implementation, used identically by training-time model selection
(dev split), the packaged evaluation, the verify script, and the tests,
so every reported number is reproduced by construction.

Multi-reference protocol: the Dakshina lexicon attests several Urdu
spellings for some romanizations (and several romanizations for most
Urdu words). For a given roman input, the reference set is every Urdu
form attested for it in the split. A prediction is an exact match if it
equals any reference; its CER uses the closest reference (minimum edit
distance), which is the standard multi-reference CER.
"""

from __future__ import annotations

from collections import OrderedDict


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance in characters (unit costs)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[len(b)]


def group_refs(pairs) -> "OrderedDict[str, list[str]]":
    """Group (source, target) pairs into source -> [targets], file order."""
    refs: "OrderedDict[str, list[str]]" = OrderedDict()
    for src, tgt in pairs:
        bucket = refs.setdefault(src, [])
        if tgt not in bucket:
            bucket.append(tgt)
    return refs


def alignment_ops(pred: str, ref: str):
    """One optimal alignment as (subs, dels, ins) character lists.

    subs: (ref_char, pred_char) pairs; dels: ref chars with no
    counterpart; ins: pred chars with no counterpart. Used for the
    error analysis: which Urdu letters the model drops, adds, confuses.
    """
    n, m = len(ref), len(pred)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + (ref[i - 1] != pred[j - 1]),
            )
    subs, dels, ins = [], [], []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (ref[i - 1] != pred[j - 1]):
            if ref[i - 1] != pred[j - 1]:
                subs.append((ref[i - 1], pred[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            dels.append(ref[i - 1])
            i -= 1
        else:
            ins.append(pred[j - 1])
            j -= 1
    return subs, dels, ins


def score_predictions(predictions: dict, refs: "OrderedDict[str, list[str]]") -> dict:
    """Score predictions against multi-reference targets.

    Returns corpus CER (total edits / total reference characters, using
    each input's closest reference), mean per-input CER, and exact-match
    accuracy over distinct inputs.
    """
    total_edits = 0
    total_ref_chars = 0
    exact = 0
    per_input_cer = []
    for src, targets in refs.items():
        pred = predictions.get(src, "")
        best_edits = None
        best_len = None
        for ref in targets:
            d = edit_distance(pred, ref)
            if best_edits is None or d < best_edits:
                best_edits = d
                best_len = len(ref)
        total_edits += best_edits
        total_ref_chars += best_len
        per_input_cer.append(best_edits / best_len if best_len else 0.0)
        if pred in targets:
            exact += 1
    n = len(refs)
    return {
        "n_inputs": n,
        "corpus_cer": total_edits / total_ref_chars if total_ref_chars else 0.0,
        "mean_cer": sum(per_input_cer) / n if n else 0.0,
        "exact_match": exact / n if n else 0.0,
        "total_edits": total_edits,
        "total_ref_chars": total_ref_chars,
    }
