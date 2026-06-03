# Synthetic generality experiment

A small, self-contained experiment showing that the elite-selection bias
studied in the paper is a general property of noisy ranking, not a Tetris
artifact.

It optimizes a known concave objective `f(theta) = -||theta||^2` with the
cross-entropy method, where each evaluation is corrupted by **heavy-tailed,
heteroscedastic** noise (noise scale grows for weak candidates, so they
occasionally post lucky high scores). Two elite rules are compared:

- **training-score selection**: pick the top-K candidates by the noisy
  training-score mean;
- **robust-validation selection**: re-rank a shortlisted pool on an
  independent noise stream by `val_mean - lambda * val_std`.

Because the true optimum is known (`f* = 0`), the script reports the true
regret of the finally selected point versus the per-candidate evaluation
budget. Robust-validation selection is far closer to the optimum, especially
at small budgets, reproducing the bias and its correction observed for
Tetris-CE.

## Requirements

Python 3 with `numpy`.

## Run

```bash
python synth_ce.py --runs 300 --out synth_results.json
```

This reproduces `synth_results.json` (the data behind the synthetic-task
figure in the paper). Results are averaged over `--runs` paired runs and are
deterministic given the seeds in the script.
