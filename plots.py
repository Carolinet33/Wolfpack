#!/usr/bin/env python3
"""
plots.py  --  presentation / analysis figures.  Run AFTER train.py.

  python plots.py            # writes PNGs into plots/

Each figure is produced only if its input file exists, so you can run this
at any stage:
  runtime-data.csv       -> 01-04  (data exploration, needs labels only)
  features_cache.json    -> 05-07  (features vs runtime)
  cv_predictions.csv     -> 08-10  (model accuracy + error analysis, CV)
  test_predictions.csv   -> test_08-10 (same, on the held-out 20% split)
  feature_importance.csv -> 11
  ablation.csv           -> 12
  learning_curve.csv     -> 13
  model_comparison.csv   -> 14
"""
import csv
import json
import math
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
from scipy.optimize import curve_fit

CAP = 14400.0
OUT = "plots"
THR_COLORS = {16: "#1f77b4", 64: "#ff7f0e", 512: "#d62728"}
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=150)
    plt.close(fig)
    print("  wrote", os.path.join(OUT, name))


def load_labels(path="runtime-data.csv"):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(l.replace("\r\n", "\n") for l in f):
            to = r["status"].strip().lower() == "timeout"
            d = r["duration_s"].strip()
            if not to and not d:
                continue
            rows.append(dict(f=r["filename"].strip(), t=int(float(r["threshold"])),
                             y=CAP if to else float(d), to=to))
    return rows


# ---------------------------------------------------------------- labels only
def label_plots(rows):
    by = defaultdict(dict)
    for r in rows:
        by[r["f"]][r["t"]] = r
    thrs = [16, 64, 512]

    # 01: runtime vs threshold, one line per circuit ("flat" vs "exploding")
    fig, ax = plt.subplots(figsize=(7, 5))
    for f, d in by.items():
        ts = [t for t in thrs if t in d]
        if len(ts) < 2:
            continue
        ys = [d[t]["y"] for t in ts]
        ratio = ys[-1] / ys[0]
        col = "#d62728" if ratio > 10 else ("#2ca02c" if ratio < 0.5 else "#7f7f7f")
        ax.plot(ts, ys, color=col, alpha=0.25, lw=0.8)
    ax.axhline(CAP, color="k", ls="--", lw=1, label="4 h cap")
    ax.set(xscale="log", yscale="log", xticks=thrs, xticklabels=thrs,
           xlabel="threshold", ylabel="runtime (s)",
           title="Runtime vs threshold, one line per circuit")
    ax.plot([], [], color="#d62728", label=">10x slower at max threshold")
    ax.plot([], [], color="#7f7f7f", label="roughly flat")
    ax.plot([], [], color="#2ca02c", label="FASTER at higher threshold")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    save(fig, "01_runtime_vs_threshold.png")

    # 02: runtime distribution per threshold (log scale) + timeouts
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.logspace(-1, math.log10(CAP) + 0.2, 40)
    for t in thrs:
        ys = [r["y"] for r in rows if r["t"] == t and not r["to"]]
        n_to = sum(1 for r in rows if r["t"] == t and r["to"])
        ax.hist(ys, bins=bins, histtype="step", lw=1.5, color=THR_COLORS[t],
                label=f"thr {t}  (n={len(ys)}, timeouts={n_to})")
    ax.set(xscale="log", xlabel="runtime (s)", ylabel="# runs",
           title="Runtimes span 5 orders of magnitude -> log-scale target")
    ax.legend(fontsize=8)
    save(fig, "02_runtime_distribution.png")

    # 03: slowdown ratio histograms (monotonicity check)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (a, b) in zip(axs, [(16, 64), (64, 512)]):
        r = [math.log10(d[b]["y"] / d[a]["y"]) for d in by.values()
             if a in d and b in d and not d[a]["to"] and not d[b]["to"]]
        ax.hist(r, bins=40, color="#555")
        ax.axvline(0, color="r", lw=1)
        neg = sum(1 for x in r if x < -0.3)
        ax.set(xlabel=f"log10( t({b}) / t({a}) )", ylabel="# circuits",
               title=f"{a} -> {b}: {neg} circuits >2x FASTER")
    fig.suptitle("Runtime is NOT always monotonic in threshold")
    save(fig, "03_threshold_ratio.png")

    # 04: status per threshold
    fig, ax = plt.subplots(figsize=(5, 3.5))
    succ = [sum(1 for r in rows if r["t"] == t and not r["to"]) for t in thrs]
    tos = [sum(1 for r in rows if r["t"] == t and r["to"]) for t in thrs]
    x = np.arange(3)
    ax.bar(x, succ, label="success", color="#2ca02c")
    ax.bar(x, tos, bottom=succ, label="timeout", color="#d62728")
    for i, n in enumerate(tos):
        ax.text(i, succ[i] + tos[i] + 5, str(n), ha="center", fontsize=9)
    ax.set(xticks=x, xticklabels=thrs, xlabel="threshold", ylabel="# runs",
           title="Labeled runs and timeouts")
    ax.legend(fontsize=8)
    save(fig, "04_status_by_threshold.png")


# ------------------------------------------------------------ features vs y
def feature_plots(rows, feats):
    import model
    pts = [(model.feature_dict(feats[r["f"]], r["t"]), r) for r in rows if r["f"] in feats]
    if len(pts) < 3:
        print("  (skip 05-07: need features for more circuits)")
        return

    def scatter(fname, xkey, xlabel, title):
        fig, ax = plt.subplots(figsize=(7, 5))
        for t in (16, 64, 512):
            xs = [d[xkey] for d, r in pts if r["t"] == t]
            ys = [r["y"] for d, r in pts if r["t"] == t]
            mk = ["x" if r["to"] else "o" for d, r in pts if r["t"] == t]
            ax.scatter(xs, ys, s=12, alpha=0.6, color=THR_COLORS[t], label=f"thr {t}")
        ax.axhline(CAP, color="k", ls="--", lw=1)
        ax.set(yscale="log", xlabel=xlabel, ylabel="runtime (s)", title=title)
        ax.legend(fontsize=8)
        save(fig, fname)

    scatter("05_entropy_vs_runtime.png", "S_peak_max",
            "peak entanglement entropy (Clifford skeleton, bits)",
            "Entanglement entropy vs runtime")
    scatter("06_bondcost_vs_runtime.png", "lcost2",
            "log10 estimated cost (bond tracker, capped at threshold)",
            "Physics cost estimate vs runtime")
    scatter("07_midmeas_vs_runtime.png", "log_meas_mid",
            "log10(1 + # mid-circuit measurements)",
            "Mid-circuit measurement vs runtime")


# ----------------------------------------------------------- model accuracy
def cv_plots(path="cv_predictions.csv", prefix="", label="Grouped CV"):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(dict(f=r["filename"], t=int(r["threshold"]), y=float(r["actual_s"]),
                             to=r["timeout"] == "1", p=float(r["pred_s"]), s=float(r["score"])))
    if len(rows) < 3:
        print("  (skip 08-10: too few CV rows)")
        return

    # 08: predicted vs actual with 2x / 10x bands
    fig, ax = plt.subplots(figsize=(6, 6))
    lo, hi = 0.05, CAP * 2
    g = np.array([lo, hi])
    ax.fill_between(g, g / 2, g * 2, color="g", alpha=0.12, label="within 2x")
    ax.plot(g, g / 10, "r:", lw=1); ax.plot(g, g * 10, "r:", lw=1, label="10x (score 0)")
    ax.plot(g, g, "k-", lw=0.8)
    for t in (16, 64, 512):
        sel = [r for r in rows if r["t"] == t]
        ax.scatter([r["y"] for r in sel], [r["p"] for r in sel], s=10, alpha=0.6,
                   color=THR_COLORS[t], label=f"thr {t}")
    ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi),
           xlabel="actual runtime (s)", ylabel="predicted (s, out-of-fold)",
           title=f"{label}: mean score {np.mean([r['s'] for r in rows]):.1%}")
    ax.legend(fontsize=8, loc="upper left")
    save(fig, prefix + "08_pred_vs_actual.png")

    # 09: score distribution + per-threshold mean
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].hist([r["s"] for r in rows], bins=25, color="#555")
    axs[0].set(xlabel="per-run score", ylabel="# runs", title="Score distribution")
    ts = [16, 64, 512]
    means = [np.mean([r["s"] for r in rows if r["t"] == t] or [0]) for t in ts]
    axs[1].bar([str(t) for t in ts], means, color=[THR_COLORS[t] for t in ts])
    for i, m in enumerate(means):
        axs[1].text(i, m + 0.01, f"{m:.1%}", ha="center")
    axs[1].set(ylim=(0, 1), xlabel="threshold", ylabel="mean score", title="Score by threshold")
    save(fig, prefix + "09_scores.png")

    # 10: worst misses (for error analysis slides) -- also written as CSV
    worst = sorted(rows, key=lambda r: r["s"])[:15]
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [f"{r['f'][:8]} @{r['t']}{' (TO)' if r['to'] else ''}" for r in worst]
    err = [math.log10(r["p"] / r["y"]) for r in worst]
    ax.barh(labels[::-1], err[::-1], color=["#d62728" if e > 0 else "#1f77b4" for e in err[::-1]])
    ax.axvline(0, color="k", lw=0.8)
    ax.set(xlabel="log10(pred / actual)   (>0 over-predict, <0 under-predict)",
           title="15 worst predictions")
    save(fig, prefix + "10_worst_misses.png")
    with open(os.path.join(OUT, prefix + "worst_misses.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "threshold", "actual_s", "pred_s", "timeout", "score"])
        for r in worst:
            w.writerow([r["f"], r["t"], r["y"], r["p"], int(r["to"]), r["s"]])


def importance_plot():
    rows = list(csv.DictReader(open("feature_importance.csv")))[:20]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh([r["feature"] for r in rows][::-1], [float(r["gain"]) for r in rows][::-1], color="#555")
    ax.set(xlabel="LightGBM gain", title="Top 20 features")
    save(fig, "11_feature_importance.png")


def ablation_plot():
    rows = list(csv.DictReader(open("ablation.csv")))
    full = float(rows[0]["cv_mean"])
    rest = sorted(rows[1:], key=lambda r: float(r["delta"]))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    d = [float(r["delta"]) * 100 for r in rest][::-1]
    e = [float(r["delta_std"]) * 100 for r in rest][::-1]
    ax.barh([r["removed_group"] for r in rest][::-1], d, xerr=e, capsize=3,
            color=["#d62728" if v < 0 else "#2ca02c" for v in d])
    ax.axvline(0, color="k", lw=0.8)
    ax.set(xlabel="change in CV score when group is REMOVED (points, ± over repeats)",
           title=f"Ablation (full model: {full:.1%}); red = group helps")
    save(fig, "12_ablation.png")


def learning_curve_plot():
    rows = list(csv.DictReader(open("learning_curve.csv")))
    x = np.array([int(r["n_train_circuits"]) for r in rows])
    m = np.array([float(r["cv_mean"]) * 100 for r in rows])
    e = np.array([float(r["cv_std"]) * 100 for r in rows])

    # --- Fit a logarithmic curve: y = a + b*log(x)
    def log_curve(x, a, b):
        return a + b * np.log(x)

    popt, _ = curve_fit(log_curve, x, m, maxfev=10000)

    # --- Extrapolate out to more data
    x_ext = np.linspace(min(x), 1000, 300)
    y_ext = log_curve(x_ext, *popt)

    fig, ax = plt.subplots(figsize=(6, 4))

    # Original points
    ax.errorbar(x, m, yerr=e, marker="o", capsize=3, label="CV score")

    # Fitted curve
    ax.plot(x_ext, y_ext, color="red", lw=2, label="Logarithmic fit")

    ax.set(
        xlabel="# training circuits",
        ylabel="CV score (%)",
        title="Learning curve with extrapolation"
    )
    ax.grid(alpha=0.3)
    ax.legend()

    save(fig, "13_learning_curve_fit.png")


def model_comparison_plot():
    rows = list(csv.DictReader(open("model_comparison.csv")))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    names = [r["model"] for r in rows][::-1]
    m = [float(r["cv_mean"]) * 100 for r in rows][::-1]
    e = [float(r["cv_std"]) * 100 for r in rows][::-1]
    ax.barh(names, m, xerr=e, capsize=3, color="#555")
    for i, v in enumerate(m):
        ax.text(v + 0.3, i, f"{v:.1f}", va="center", fontsize=8)
    lo = max(0, min(m) - 10)
    ax.set(xlim=(lo, 100), xlabel="repeated grouped CV score (%)",
           title="Model comparison (same folds for every model)")
    save(fig, "14_model_comparison.png")


if __name__ == "__main__":
    rows = load_labels()
    print("Data exploration plots:")
    label_plots(rows)
    if os.path.exists("features_cache.json"):
        feats = json.load(open("features_cache.json"))
        feats.pop("__version__", None)
        print("Feature plots:")
        feature_plots(rows, feats)
    if os.path.exists("cv_predictions.csv"):
        print("Model plots (cross-validation, out-of-fold):")
        cv_plots()
    if os.path.exists("test_predictions.csv"):
        print("Model plots (held-out 20% test split):")
        cv_plots("test_predictions.csv", prefix="test_", label="Test split (unseen circuits)")
    if os.path.exists("feature_importance.csv"):
        importance_plot()
    if os.path.exists("ablation.csv"):
        ablation_plot()
    if os.path.exists("learning_curve.csv"):
        learning_curve_plot()
    if os.path.exists("model_comparison.csv"):
        model_comparison_plot()
