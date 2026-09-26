#!/usr/bin/env python3
"""
train.py  --  build features, cross-validate, train and save the model.

Usage (from inside quantathon-harness/):
  python train.py                      # uses circuits/ and runtime-data.csv
  python train.py --no-cache           # re-parse every circuit
  python train.py --ablate             # also measure each feature group's value
  python train.py --labels train_labels.csv --test test_labels.csv
                                       # train on the train split, score on the test split

Outputs:
  features_cache.json   parsed features per circuit (so re-runs are fast)
  model_lgb.txt         trained LightGBM model, loaded by model.py
  cv_predictions.csv    out-of-fold predictions for error analysis / slides
  feature_importance.csv, ablation.csv   inputs for plots.py
"""
import argparse
import csv
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

from run import read_qasm                  # the harness's zstd reader
from model import (parse_qasm, feature_vector, FEATURE_NAMES, FEATURE_GROUPS,
                   CAP_S, MODEL_FILE)

L_METRIC = 2.0
CACHE_VERSION = "v2-entropy"


def comp_score(pred_s, actual_s, is_timeout):
    """Exact competition metric, vectorised."""
    pred = np.asarray(pred_s, float).copy()
    act = np.asarray(actual_s, float)
    pred[is_timeout] = np.minimum(pred[is_timeout], CAP_S)
    pred = np.maximum(pred, 1e-9)
    return np.maximum(0.0, 1.0 - np.abs(np.log10(pred / act)) / L_METRIC)


def load_labels(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(line.replace("\r\n", "\n") for line in f):
            to = r["status"].strip().lower() == "timeout"
            d = r["duration_s"].strip()
            if not to and not d:
                continue
            act = CAP_S if to else float(d)
            rows.append((r["filename"].strip(), int(float(r["threshold"])), act, to))
    return rows


def featurize_all(circuit_dir, cache_path, use_cache):
    cache = {}
    if use_cache and os.path.exists(cache_path):
        cache = json.load(open(cache_path))
        if cache.get("__version__") != CACHE_VERSION:
            cache = {}
    cache["__version__"] = CACHE_VERSION
    files = sorted(p for p in Path(circuit_dir).rglob("*")
                   if p.suffix in (".qasm", ".zst") and p.is_file())
    slow = []
    for i, p in enumerate(files, 1):
        name = p.name[:-4] if p.suffix == ".zst" else p.name
        if name in cache:
            continue
        txt = read_qasm(p)
        t0 = time.perf_counter()
        cache[name] = parse_qasm(txt)
        dt = time.perf_counter() - t0
        if dt > 5:
            slow.append((name, round(dt, 1), len(txt) // 1_000_000))
        if i % 25 == 0:
            print(f"  parsed {i}/{len(files)}")
            json.dump(cache, open(cache_path, "w"))
    json.dump(cache, open(cache_path, "w"))
    if slow:
        print("Slow parses (name, s, MB):", slow[:10])
    return cache


def lgb_params(n_rows):
    return dict(
        objective="l1",               # metric is linear in |log error| -> L1
        learning_rate=0.03,
        num_leaves=15,
        min_data_in_leaf=max(2, min(15, n_rows // 40)),
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=1.0,
        verbose=-1,
        seed=0,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuits", default="circuits")
    ap.add_argument("--labels", default="runtime-data.csv")
    ap.add_argument("--rounds", type=int, default=600)
    ap.add_argument("--test", default=None,
                    help="labels CSV of held-out circuits to score the final model on")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--ablate", action="store_true",
                    help="retrain with each feature group removed and report the CV drop")
    args = ap.parse_args()

    feats = featurize_all(args.circuits, "features_cache.json", not args.no_cache)
    labels = [r for r in load_labels(args.labels) if r[0] in feats]
    print(f"{len(labels)} labeled runs over {len({r[0] for r in labels})} circuits")

    X = np.array([feature_vector(feats[f], t) for f, t, _, _ in labels])
    act = np.array([a for _, _, a, _ in labels])
    to = np.array([x for _, _, _, x in labels])
    y = np.log10(np.minimum(act, CAP_S))          # target: log10 seconds, capped
    groups = np.array([f for f, _, _, _ in labels])
    params = lgb_params(len(y))

    # ---------- grouped CV: a circuit never appears in both train and test ----------
    n_groups = len(set(groups))
    k = min(5, n_groups)

    def cv_score(cols):
        oof = np.full(len(y), np.nan)
        for tr, te in GroupKFold(n_splits=k).split(X, y, groups):
            m = lgb.train(params, lgb.Dataset(X[tr][:, cols], y[tr]), args.rounds)
            oof[te] = m.predict(X[te][:, cols])
        p = np.clip(10 ** oof, 0.01, CAP_S)
        return p, comp_score(p, act, to)

    all_cols = list(range(X.shape[1]))
    if k >= 2:
        pred_s, s = cv_score(all_cols)
        print(f"\nGrouped {k}-fold CV competition score: {s.mean():.2%}")
        for thr in sorted(set(t for _, t, _, _ in labels)):
            msk = np.array([t == thr for _, t, _, _ in labels])
            print(f"   threshold {thr:>4}: {s[msk].mean():.2%}  (n={msk.sum()})")
        print(f"   timeout rows    : {s[to].mean():.2%}  (n={to.sum()})" if to.any() else "")

        # baseline to beat: median log-runtime per threshold (from train folds)
        base = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=k).split(X, y, groups):
            med = defaultdict(list)
            for i in tr:
                med[labels[i][1]].append(y[i])
            for i in te:
                base[i] = np.median(med.get(labels[i][1], [np.median(y[tr])]))
        print(f"   baseline (median per threshold): "
              f"{comp_score(10 ** base, act, to).mean():.2%}")

        with open("cv_predictions.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["filename", "threshold", "actual_s", "timeout", "pred_s", "score"])
            for (fn, t, a, x), p, sc in zip(labels, pred_s, s):
                w.writerow([fn, t, a, int(x), round(float(p), 4), round(float(sc), 4)])
        print("Out-of-fold predictions -> cv_predictions.csv")

        if args.ablate:
            full = s.mean()
            print("\nAblation (CV score when a feature group is REMOVED):")
            abl_rows = [("all features", full)]
            for gname, names in FEATURE_GROUPS.items():
                if gname in ("setting",):
                    continue
                drop = {FEATURE_NAMES.index(n) for n in names}
                cols = [i for i in all_cols if i not in drop]
                _, sa = cv_score(cols)
                print(f"   without {gname:<13}: {sa.mean():.2%}   (change {sa.mean() - full:+.2%})")
                abl_rows.append((gname, sa.mean()))
            with open("ablation.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["removed_group", "cv_score"])
                w.writerows(abl_rows)
    else:
        print("Too few circuits for CV; training on everything.")

    # ---------- final model on all data ----------
    final = lgb.train(params, lgb.Dataset(X, y), args.rounds)
    final.save_model(MODEL_FILE)

    # ---------- held-out test split (circuits never used in training) ----------
    if args.test:
        test = [r for r in load_labels(args.test) if r[0] in feats]
        overlap = {r[0] for r in test} & set(groups)
        if overlap:
            print(f"WARNING: {len(overlap)} test circuits also in training labels!")
        if test:
            Xt = np.array([feature_vector(feats[f], t) for f, t, _, _ in test])
            at = np.array([a for _, _, a, _ in test])
            tt = np.array([x for _, _, _, x in test])
            pt = np.clip(10 ** final.predict(Xt), 0.01, CAP_S)
            st = comp_score(pt, at, tt)
            print(f"\nTEST SPLIT score ({len({r[0] for r in test})} unseen circuits, "
                  f"{len(test)} runs): {st.mean():.2%}")
            with open("test_predictions.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["filename", "threshold", "actual_s", "timeout", "pred_s", "score"])
                for (fn, t, a, x), p, sc in zip(test, pt, st):
                    w.writerow([fn, t, a, int(x), round(float(p), 4), round(float(sc), 4)])
            print("Test predictions -> test_predictions.csv")
            print("NOTE: for the final submission, retrain on ALL labels: python train.py")
        else:
            print("No test circuits found in circuits/ (check filenames).")
    imp = final.feature_importance("gain")
    order = np.argsort(-imp)
    with open("feature_importance.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature", "gain"])
        for i in order:
            w.writerow([FEATURE_NAMES[i], float(imp[i])])
    print(f"\nSaved {MODEL_FILE}. Top features by gain:")
    for i in order[:12]:
        print(f"   {FEATURE_NAMES[i]:<18} {imp[i]:.1f}")


if __name__ == "__main__":
    main()