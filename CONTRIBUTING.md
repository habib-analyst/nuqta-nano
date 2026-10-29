# Contributing

Issues and pull requests are welcome on the GitHub repository.

Ground rules for this artifact:

- **No unverifiable numbers.** Any metric in the README must be
  reproducible from the packaged weights with
  `examples/verify_metrics.py`. If you change the weights, rerun
  `scripts/evaluate.py` and commit the regenerated
  `assets/eval_results.json` with the change.
- **Data licensing is a hard gate.** Training data comes only from the
  Dakshina dataset (CC BY-SA 4.0) as documented in `DATA.md`. Do not
  add scraped or unclearly-licensed pairs.
- **The test split stays held out.** Model selection uses the dev
  split only; the test split is for final measurement.
- Run the test suite (`python -m pytest`) before submitting — it pins
  the headline metrics and the model's parameter count, so a silent
  weights or code change fails loudly.
