#!/usr/bin/env python3
"""
train.py  --  build features, evaluate with REPEATED grouped CV, train & save.

Typical use (from inside quantathon-harness/):
  python train.py --labels train_labels.csv --test test_labels.csv   # evaluate on the 20% split
  python train.py --compare --ablate --learning-curve               # full analysis on all data
  python train.py --final blend                                     # final model for submission

Flags
  --repeats N        repeated grouped CV: N different random groupings (default 5)
  --compare          compare LightGBM / 2-stage / XGBoost-AFT / ridge / random forest / MLP / blend
  --ablate           remove each feature group in turn (paired, same folds) and report the drop
  --learning-curve   score vs fraction of training circuits used
  --final KIND       what to save for submission: lgb (default), blend, lgb_2stage, blend_2stage
  --test FILE        labels of held-out circuits: scored with the SAVED model via model.py
  --no-cache         re-parse all circuits (after changing the parser)

Outputs
  features_cache.json, model_lgb.txt (+ model_ridge.json, model_clf.txt, model_config.json)
  cv_predictions.csv, test_predictions.csv, feature_importance.csv,
  model_comparison.csv, ablation.csv, learning_curve.csv   (read by plots.py)
"""
import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

from run import read_qasm
import model as M
from model import parse_qasm, feature_vector, FEATURE_NAMES, FEATURE_GROUPS, CAP_S

# FIX: bumped from "v3-graph-sampling" so the old cached features (made by
# the buggy parser) are thrown away and every circuit is re-parsed.
CACHE_VERSION = "v4-parser-fixes"
LOG_CAP = np.log10(CAP_S)


# ----------------------------------------------------------------- metric
def comp_score(pred_s, actual_s, is_timeout):
    """Exact competition metric (same as score.py), vectorised."""
    pred = np.asarray(pred_s, float).copy()
    act = np.asarray(actual_s, float)
    pred[is_timeout] = np.minimum(pred[is_timeout], CAP_S)
    pred = np.maximum(pred, 1e-9)
    return np.maximum(0.0, 1.0 - np.abs(np.log10(pred / act)) / 2.0)


def to_seconds(log_pred):
    return np.clip(10 ** np.asarray(log_pred, float), 0.01, CAP_S)


# ----------------------------------------------------------------- data
def load_labels(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(l.replace("\r\n", "\n") for l in f):
            to = r["status"].strip().lower() == "timeout"
            d = r["duration_s"].strip()
            if not to and not d:
                continue
            rows.append((r["filename"].strip(), int(float(r["threshold"])),
                         CAP_S if to else float(d), to))
    return rows


def featurize_all(circuit_dir, cache_path, use_cache):
    cache = {}
    if use_cache and os.path.exists(cache_path):
        cache = json.load(open(cache_path))
        if cache.get("__version__") != CACHE_VERSION:
            print("Feature cache is from an older parser version -> re-parsing.")
            cache = {}
    cache["__version__"] = CACHE_VERSION
    files = sorted(p for p in Path(circuit_dir).rglob("*")
                   if p.suffix in (".qasm", ".zst") and p.is_file())
    slow, t_all = [], time.perf_counter()
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
            print(f"  parsed {i}/{len(files)}  ({time.perf_counter() - t_all:.0f}s)")
            json.dump(cache, open(cache_path, "w"))
    json.dump(cache, open(cache_path, "w"))
    if slow:
        print("Slow parses (name, seconds, MB):", slow[:10])
    return cache


# ----------------------------------------------------------------- models
def lgb_params(n_rows, **kw):
    p = dict(objective="l1", learning_rate=0.03, num_leaves=15,
             min_data_in_leaf=max(2, min(15, n_rows // 40)),
             feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
             lambda_l2=1.0, verbose=-1, seed=0)
    p.update(kw)
    return p


def fit_lgb(X, y, rounds, seed=0):
    return lgb.train(lgb_params(len(y), seed=seed), lgb.Dataset(X, y), rounds)


def fit_clf(X, to, rounds):
    p = dict(objective="binary", learning_rate=0.05, num_leaves=7,
             min_data_in_leaf=max(2, min(10, len(to) // 60)),
             feature_fraction=0.8, verbose=-1, seed=0)
    return lgb.train(p, lgb.Dataset(X, to.astype(float)), max(100, rounds // 3))


def fit_ridge(X, y, alpha=3.0):
    from sklearn.linear_model import Ridge
    mean = X.mean(0)
    scale = X.std(0)
    scale[scale < 1e-9] = 1.0
    r = Ridge(alpha=alpha).fit((X - mean) / scale, y)
    return {"mean": mean.tolist(), "scale": scale.tolist(),
            "coef": r.coef_.tolist(), "intercept": float(r.intercept_)}


def ridge_predict(r, X):
    return ((X - np.array(r["mean"])) / np.array(r["scale"])) @ np.array(r["coef"]) + r["intercept"]


BLEND_W = 0.75   # weight on LightGBM in the blend (checked in --compare)


def fit_predict(kind, Xtr, ytr, totr, act_tr, Xte, rounds):
    """Train model `kind` on the training fold, return log10-second predictions."""
    if kind == "lgb":
        return fit_lgb(Xtr, ytr, rounds).predict(Xte)
    if kind == "lgb_seeds":                      # average of 5 seeds (variance reduction)
        return np.mean([fit_lgb(Xtr, ytr, rounds, s).predict(Xte) for s in range(5)], 0)
    if kind in ("lgb_2stage", "blend_2stage"):
        base = fit_predict("lgb" if kind == "lgb_2stage" else "blend",
                           Xtr, ytr, totr, act_tr, Xte, rounds)
        if totr.sum() < 3:
            return base
        p = fit_clf(Xtr, totr, rounds).predict(Xte)
        return np.where(p > 0.5, LOG_CAP, base)
    if kind == "ridge":
        return ridge_predict(fit_ridge(Xtr, ytr), Xte)
    if kind == "blend":
        return (BLEND_W * fit_lgb(Xtr, ytr, rounds).predict(Xte)
                + (1 - BLEND_W) * ridge_predict(fit_ridge(Xtr, ytr), Xte))
    if kind == "rf":
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                   max_features=0.5, n_jobs=-1, random_state=0)
        return rf.fit(Xtr, ytr).predict(Xte)
    if kind == "mlp":
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        m = make_pipeline(StandardScaler(),
                          MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-2,
                                       max_iter=3000, early_stopping=True,
                                       random_state=0))
        return m.fit(Xtr, ytr).predict(Xte)
    if kind == "xgb_aft":
        # Censored regression: a timeout means "true runtime >= cap, unknown".
        import xgboost as xgb
        lo = act_tr.copy()
        hi = np.where(totr, np.inf, act_tr)
        d = xgb.DMatrix(Xtr)
        d.set_float_info("label_lower_bound", lo)
        d.set_float_info("label_upper_bound", hi)
        p = dict(objective="survival:aft", eval_metric="aft-nloglik",
                 aft_loss_distribution="normal", aft_loss_distribution_scale=1.0,
                 learning_rate=0.03, max_depth=4, subsample=0.8,
                 colsample_bytree=0.8, verbosity=0, seed=0)
        b = xgb.train(p, d, rounds)
        return np.log10(np.maximum(b.predict(xgb.DMatrix(Xte)), 1e-3))
    raise ValueError(kind)


# ----------------------------------------------------------------- repeated CV
def shuffled_groups(groups, seed):
    """GroupKFold is deterministic; relabelling circuits randomly gives a
    different (still grouped) fold assignment for every repeat."""
    uniq = np.unique(groups)
    perm = dict(zip(uniq, np.random.default_rng(seed).permutation(len(uniq))))
    return np.array([perm[g] for g in groups])


def repeated_cv(kinds, X, y, to, act, groups, cols, repeats, rounds,
                keep_oof=None, train_frac=1.0):
    """Returns {kind: array of per-repeat mean scores}. Every kind sees exactly
    the same folds in each repeat, so comparisons between kinds are PAIRED.
    train_frac < 1 subsamples training circuits (for the learning curve)."""
    k = min(5, len(np.unique(groups)))
    res = {kd: [] for kd in kinds}
    oof = np.full(len(y), np.nan) if keep_oof else None
    for rep in range(repeats):
        g = shuffled_groups(groups, rep)
        preds = {kd: np.full(len(y), np.nan) for kd in kinds}
        for fold, (tr, te) in enumerate(GroupKFold(n_splits=k).split(X, y, g)):
            if train_frac < 1.0:
                ug = np.unique(g[tr])
                keep = set(np.random.default_rng(1000 * rep + fold).choice(
                    ug, max(2, int(len(ug) * train_frac)), replace=False))
                tr = np.array([i for i in tr if g[i] in keep])
            for kd in kinds:
                preds[kd][te] = fit_predict(kd, X[tr][:, cols], y[tr], to[tr],
                                            act[tr], X[te][:, cols], rounds)
        for kd in kinds:
            res[kd].append(comp_score(to_seconds(preds[kd]), act, to).mean())
        if keep_oof and rep == 0:
            oof[:] = preds[keep_oof]
    return {kd: np.array(v) for kd, v in res.items()}, oof


def fmt(v):
    return f"{v.mean():.2%} ± {v.std():.2%}"


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuits", default="circuits")
    ap.add_argument("--labels", default="runtime-data.csv")
    ap.add_argument("--test", default=None)
    ap.add_argument("--rounds", type=int, default=600)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--learning-curve", action="store_true")
    ap.add_argument("--final", default="lgb",
                    choices=["lgb", "blend", "lgb_2stage", "blend_2stage"])
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    feats = featurize_all(args.circuits, "features_cache.json", not args.no_cache)
    labels = [r for r in load_labels(args.labels) if r[0] in feats]
    n_circ = len({r[0] for r in labels})
    print(f"\n{len(labels)} labeled runs over {n_circ} circuits, {len(FEATURE_NAMES)} features")

    X = np.array([feature_vector(feats[f], t) for f, t, _, _ in labels])
    act = np.array([a for _, _, a, _ in labels])
    to = np.array([x for _, _, _, x in labels])
    y = np.log10(np.minimum(act, CAP_S))
    groups = np.array([f for f, _, _, _ in labels])
    all_cols = list(range(X.shape[1]))
    can_cv = n_circ >= 5

    if can_cv:
        # ---------- 1. repeated grouped CV of the chosen final model ----------
        res, oof = repeated_cv([args.final], X, y, to, act, groups, all_cols,
                               args.repeats, args.rounds, keep_oof=args.final)
        print(f"\nRepeated grouped CV ({args.repeats}x5 folds), model '{args.final}': "
              f"{fmt(res[args.final])}")
        pred_s = to_seconds(oof)
        s = comp_score(pred_s, act, to)
        for thr in sorted(set(t for _, t, _, _ in labels)):
            m = np.array([t == thr for _, t, _, _ in labels])
            print(f"   threshold {thr:>4}: {s[m].mean():.2%}  (n={m.sum()})")
        if to.any():
            print(f"   timeout rows   : {s[to].mean():.2%}  (n={to.sum()})")
        with open("cv_predictions.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["filename", "threshold", "actual_s", "timeout", "pred_s", "score"])
            for (fn, t, a, x), p, sc in zip(labels, pred_s, s):
                w.writerow([fn, t, a, int(x), round(float(p), 4), round(float(sc), 4)])

        # ---------- 2. model comparison ----------
        if args.compare:
            kinds = ["lgb", "lgb_seeds", "lgb_2stage", "ridge", "blend",
                     "blend_2stage", "rf", "mlp"]
            try:
                import xgboost  # noqa: F401
                kinds.insert(3, "xgb_aft")
            except ImportError:
                print("(xgboost not installed -> skipping the censored AFT model)")
            print(f"\nModel comparison (same folds for every model, {args.repeats} repeats):")
            res, _ = repeated_cv(kinds, X, y, to, act, groups, all_cols,
                                 args.repeats, args.rounds)
            base_rows = [r for r in labels]
            with open("model_comparison.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["model", "cv_mean", "cv_std", "diff_vs_lgb", "diff_std"])
                for kd in sorted(kinds, key=lambda k: -res[k].mean()):
                    d = res[kd] - res["lgb"]
                    print(f"   {kd:<13} {fmt(res[kd])}   vs lgb {d.mean():+.2%} ± {d.std():.2%}")
                    w.writerow([kd, res[kd].mean(), res[kd].std(), d.mean(), d.std()])
            # naive baseline for reference: median log runtime per threshold
            med = {t: np.median([yy for (f_, tt, _, _), yy in zip(base_rows, y) if tt == t])
                   for t in set(t for _, t, _, _ in labels)}
            bl = comp_score(to_seconds([med[t] for _, t, _, _ in labels]), act, to).mean()
            print(f"   (naive baseline, median per threshold, in-sample: {bl:.2%})")

        # ---------- 3. ablation (paired) ----------
        if args.ablate:
            print(f"\nAblation: remove one feature group, same folds ({args.repeats} repeats):")
            full, _ = repeated_cv(["lgb"], X, y, to, act, groups, all_cols,
                                  args.repeats, args.rounds)
            full = full["lgb"]
            rows = [("all features", full.mean(), full.std(), 0.0, 0.0)]
            for gname, names in FEATURE_GROUPS.items():
                if gname == "setting":
                    continue
                drop = {FEATURE_NAMES.index(n) for n in names}
                cols = [i for i in all_cols if i not in drop]
                r, _ = repeated_cv(["lgb"], X, y, to, act, groups, cols,
                                   args.repeats, args.rounds)
                d = r["lgb"] - full
                flag = "  <- clearly helps" if d.mean() + 2 * d.std() < 0 else ""
                print(f"   without {gname:<13} {fmt(r['lgb'])}   change {d.mean():+.2%} ± {d.std():.2%}{flag}")
                rows.append((gname, r["lgb"].mean(), r["lgb"].std(), d.mean(), d.std()))
            with open("ablation.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["removed_group", "cv_mean", "cv_std", "delta", "delta_std"])
                w.writerows(rows)

        # ---------- 4. learning curve ----------
        if args.learning_curve:
            print("\nLearning curve (fraction of training circuits used in each fold):")
            with open("learning_curve.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["train_frac", "n_train_circuits", "cv_mean", "cv_std"])
                for frac in (0.2, 0.4, 0.6, 0.8, 1.0):
                    r, _ = repeated_cv(["lgb"], X, y, to, act, groups, all_cols,
                                       args.repeats, args.rounds, train_frac=frac)
                    ntr = int(n_circ * 0.8 * frac)
                    print(f"   {frac:>4.0%} (~{ntr} circuits): {fmt(r['lgb'])}")
                    w.writerow([frac, ntr, r["lgb"].mean(), r["lgb"].std()])
    else:
        print("Too few circuits for cross-validation; training on everything.")

    # ---------- 5. final model on all given labels ----------
    for p in (M.RIDGE_FILE, M.CLF_FILE):
        if os.path.exists(p):
            os.remove(p)
    final = fit_lgb(X, y, args.rounds)
    final.save_model(M.MODEL_FILE)
    cfg = {"blend_w": 1.0, "use_clf": False, "kind": args.final}
    if "blend" in args.final:
        json.dump(fit_ridge(X, y), open(M.RIDGE_FILE, "w"))
        cfg["blend_w"] = BLEND_W
    if "2stage" in args.final and to.sum() >= 3:
        fit_clf(X, to, args.rounds).save_model(M.CLF_FILE)
        cfg["use_clf"] = True
    json.dump(cfg, open(M.CONFIG_FILE, "w"))
    imp = final.feature_importance("gain")
    order = np.argsort(-imp)
    with open("feature_importance.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature", "gain"])
        for i in order:
            w.writerow([FEATURE_NAMES[i], float(imp[i])])
    print(f"\nSaved final model '{args.final}' ({M.MODEL_FILE}). Top features by gain:")
    for i in order[:12]:
        print(f"   {FEATURE_NAMES[i]:<22} {imp[i]:.1f}")

    # ---------- 6. held-out test split, scored through model.py itself ----------
    if args.test:
        test = [r for r in load_labels(args.test) if r[0] in feats]
        overlap = {r[0] for r in test} & set(groups)
        if overlap:
            print(f"WARNING: {len(overlap)} test circuits are also in the training labels!")
        if test:
            rm = M.RuntimeModel()                      # exactly what run.py will use
            pt = np.array([rm.predict(feats[f], t) for f, t, _, _ in test])
            at = np.array([a for _, _, a, _ in test])
            tt = np.array([x for _, _, _, x in test])
            st = comp_score(pt, at, tt)
            print(f"\nTEST SPLIT ({len({r[0] for r in test})} unseen circuits, "
                  f"{len(test)} runs): {st.mean():.2%}")
            with open("test_predictions.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["filename", "threshold", "actual_s", "timeout", "pred_s", "score"])
                for (fn, t, a, x), p, sc in zip(test, pt, st):
                    w.writerow([fn, t, a, int(x), round(float(p), 4), round(float(sc), 4)])
            print("NOTE: for the submission, retrain on ALL labels (no --test, "
                  "--labels runtime-data.csv).")


if __name__ == "__main__":
    main()