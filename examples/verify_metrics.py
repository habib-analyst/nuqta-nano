"""Reproduce the headline test metrics from the packaged weights.

Loads model.safetensors (nothing else), re-scores the full held-out
test split and the fixed 500-input subset, and compares against
assets/eval_results.json. Greedy decoding is deterministic, so the
numbers must match exactly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / "src"))
sys.path.insert(0, str(PKG / "scripts"))

from evaluate import load_pairs, predict_all  # noqa: E402
from nuqta_nano.infer import Transliterator  # noqa: E402
from nuqta_nano.metrics import group_refs, score_predictions  # noqa: E402


def main() -> None:
    recorded = json.loads((PKG / "assets/eval_results.json").read_text(encoding="utf-8"))
    trans = Transliterator.from_dir(PKG)
    refs = group_refs(load_pairs("test"))
    preds = predict_all(trans, list(refs.keys()))
    scores = score_predictions(preds, refs)

    ok = True
    for field in ("corpus_cer", "mean_cer", "exact_match", "n_inputs"):
        match = scores[field] == recorded["test"][field]
        ok &= match
        print(f"test {field}: reproduced {scores[field]} vs recorded {recorded['test'][field]} "
              f"{'OK' if match else 'MISMATCH'}")
    subset_inputs = list(refs.keys())[:500]
    sub = score_predictions({k: preds[k] for k in subset_inputs},
                            {k: refs[k] for k in subset_inputs})
    for field in ("corpus_cer", "exact_match"):
        match = sub[field] == recorded["test_subset500"][field]
        ok &= match
        print(f"subset500 {field}: reproduced {sub[field]} vs recorded "
              f"{recorded['test_subset500'][field]} {'OK' if match else 'MISMATCH'}")
    print("ALL HEADLINE METRICS REPRODUCED" if ok else "REPRODUCTION FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
