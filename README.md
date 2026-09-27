# Quantathon — Circuit Runtime Predictor

Predicts how long a quantum circuit will take to simulate, given the circuit
(QASM 2/3) and a simulation threshold (`16`, `64`, `512`). Runtimes span five
orders of magnitude — milliseconds to multi-hour timeouts — so the model
predicts `log10(seconds)` and is scored on a capped log-error metric.

**Held-out test score: 92.46%** (repeated grouped CV on the training set:
high-80s to low-90s%), well above a naive per-threshold median baseline of
~57%.

## How it works

1. **Parse** — a fast, pure-Python QASM parser (`model.py`) reads each
   circuit once and extracts ~50 features across nine families: circuit
   size/depth, gate mix, two-qubit locality, a worst-case bond-dimension cost
   tracker, exact Clifford-skeleton entanglement entropy (via a stabilizer
   tableau evolved alongside the parse), interaction-graph stats, sampling
   cost, and measurement/mid-circuit-measurement counts.
2. **Train** — a LightGBM model (`train.py`) is trained on `log10(seconds)`,
   evaluated with **repeated grouped cross-validation** (a circuit's runs at
   different thresholds never split across train/test, so there's no
   leakage) and validated on a genuinely held-out 20% of circuits.
3. **Predict** — `run.py` loads the trained model and scores a directory of
   circuits at each threshold, producing `submission.csv`.

## Project structure

| File | Purpose |
|---|---|
| `model.py` | Circuit parser + feature extraction + `RuntimeModel` (used by `run.py`) |
| `train.py` | Builds features, runs repeated grouped CV, trains & saves the model |
| `split.py` | Splits `runtime-data.csv` into `train_labels.csv` / `test_labels.csv`, grouped and stratified by circuit |
| `plots.py` | Generates all analysis/presentation figures into `plots/` |
| `run.py` | **Submission harness — do not edit.** Scores a circuit directory with the trained model and writes `submission.csv` |

## Setup

```
pip install numpy lightgbm scikit-learn matplotlib optuna zstandard
```
(`optuna` only needed for `--tune`; `zstandard` only needed if your circuits are `.zst`-compressed.)

## Workflow

### 1. Split off a held-out test set (once)
```
python split.py
```
Writes `train_labels.csv` (80%) and `test_labels.csv` (20%), split **by
circuit** (never by run) and stratified by timeout status + runtime quartile
so both sides look like the full dataset.

### 2. Evaluate on the held-out split
```
python train.py --circuits training_circuits --labels train_labels.csv --test test_labels.csv
```
Prints repeated grouped CV score (on the 80%) and a `TEST SPLIT` score (on
the unseen 20%) — the number that best estimates real performance.

Useful add-ons:
- `--tune --trials 40` — Optuna hyperparameter search (saves `model_tuned_params.json`)
- `--compare` — compares LightGBM / blend / 2-stage / ridge / random forest / MLP / XGBoost-AFT (`model_comparison.csv`)
- `--ablate` — drops each feature group in turn to see what's actually helping (`ablation.csv`)
- `--learning-curve` — score vs. fraction of training circuits used (`learning_curve.csv`)
- `--final KIND` — which model to save: `lgb` (default), `blend`, `lgb_2stage`, `blend_2stage`

### 3. Generate plots
```
python plots.py
```
Reads whatever output files exist (`cv_predictions.csv`, `test_predictions.csv`,
`feature_importance.csv`, `ablation.csv`, `learning_curve.csv`,
`model_comparison.csv`) and writes labeled PNGs to `plots/`.

### 4. Train the final submission model on ALL labeled data
```
python train.py --circuits training_circuits --labels runtime-data.csv --final <best kind from --compare>
```
No `--test` this time — the submitted model should use 100% of the labeled
circuits, not just the 80% split. This overwrites `model_lgb.txt` (and
`model_ridge.json` / `model_clf.txt` / `model_config.json` as needed), which
is what `run.py` actually loads.

### 5. Score the real submission circuits and produce `submission.csv`
```
python run.py --team "Your Team Name" --circuits <organizer-provided circuits dir>
```
`--circuits` defaults to `circuits/` — point it at wherever the actual
submission set lives (likely **not** `training_circuits/`). DM the resulting
`submission.csv` to the organizers.

## Notes / known issues

- **`features_cache.json`** caches parsed features per circuit and is
  versioned (`CACHE_VERSION` in `train.py`); bump it whenever `model.py`'s
  parser changes so stale features get thrown away automatically. Pass
  `--no-cache` to force a full re-parse.
- **π character bug (`model.py`, `_param_vals`)**: the line meant to replace
  a literal Unicode `π` with its numeric value currently matches the wrong
  character (`"Ï"`, U+00CF, not `"π"`, U+03C0) — likely leftover mojibake
  from an encoding mismatch. Any circuit using the literal `π` symbol (rather
  than the ASCII word `pi`) will have that gate's parameter silently treated
  as symbolic and conservatively counted as non-Clifford. Fix and re-run with
  `--no-cache` before the final submission if any circuits use the symbol.
- `run.py` is the harness the organizers provide — don't edit it; all model
  changes belong in `model.py`.
