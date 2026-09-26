#!/usr/bin/env python3
"""
plots.py  --  presentation / analysis figures.  Run AFTER train.py.

  python plots.py            # writes PNGs into plots/

Each figure is produced only if its input file exists, so you can run this
at any stage:
  runtime-data.csv        -> 01-04  (data exploration, needs labels only)
  features_cache.json     -> 05-07  (features vs runtime)
  cv_predictions.csv      -> 08-10  (model accuracy + error analysis, CV / train side)
  feature_importance.csv  -> 11
  ablation.csv            -> 12
  test_predictions.csv    -> 13-15  (model accuracy on the TRUE held-out test split)
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


# ----------------------------------------------------------- shared: pred-vs-actual style plots
def _load_pred_csv(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(dict(f=r["filename"], t=int(r["threshold"]), y=float(r["actual_s"]),
                             to=r["timeout"] == "1", p=float(r["pred_s"]), s=float(r["score"])))
    return rows


def _pred_vs_actual_plots(rows, prefix, label, title_suffix):
    """Shared plotting logic for both cv_predictions.csv and test_predictions.csv."""
    # pred vs actual with 2x / 10x bands
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
           xlabel="actual runtime (s)", ylabel=f"predicted (s, {label})",
           title=f"{title_suffix}: mean score {np.mean([r['s'] for r in rows]):.1%}")
    ax.legend(fontsize=8, loc="upper left")
    save(fig, f"{prefix}_pred_vs_actual.png")

    # score distribution + per-threshold mean
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].hist([r["s"] for r in rows], bins=25, color="#555")
    axs[0].set(xlabel="per-run score", ylabel="# runs", title="Score distribution")
    ts = [16, 64, 512]
    means = [np.mean([r["s"] for r in rows if r["t"] == t] or [0]) for t in ts]
    axs[1].bar([str(t) for t in ts], means, color=[THR_COLORS[t] for t in ts])
    for i, m in enumerate(means):
        axs[1].text(i, m + 0.01, f"{m:.1%}", ha="center")
    axs[1].set(ylim=(0, 1), xlabel="threshold", ylabel="mean score", title="Score by threshold")
    save(fig, f"{prefix}_scores.png")

    # worst misses (for error analysis slides) -- also written as CSV
    worst = sorted(rows, key=lambda r: r["s"])[:15]
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [f"{r['f'][:8]} @{r['t']}{' (TO)' if r['to'] else ''}" for r in worst]
    err = [math.log10(r["p"] / r["y"]) for r in worst]
    ax.barh(labels[::-1], err[::-1], color=["#d62728" if e > 0 else "#1f77b4" for e in err[::-1]])
    ax.axvline(0, color="k", lw=0.8)
    ax.set(xlabel="log10(pred / actual)   (>0 over-predict, <0 under-predict)",
           title="15 worst predictions")
    save(fig, f"{prefix}_worst_misses.png")
    with open(os.path.join(OUT, f"{prefix}_worst_misses.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "threshold", "actual_s", "pred_s", "timeout", "score"])
        for r in worst:
            w.writerow([r["f"], r["t"], r["y"], r["p"], int(r["to"]), r["s"]])


# ----------------------------------------------------------- model accuracy (train-side CV)
def cv_plots():
    if not os.path.exists("cv_predictions.csv"):
        print("  (skip 08-10: cv_predictions.csv not found)")
        return
    rows = _load_pred_csv("cv_predictions.csv")
    if len(rows) < 3:
        print("  (skip 08-10: too few CV rows)")
        return
    _pred_vs_actual_plots(rows, "08_10", "out-of-fold",
                          "Grouped CV (train side)")
    # rename to match original numbering (08, 09, 10) for backwards compatibility
    for old, new in [("08_10_pred_vs_actual.png", "08_pred_vs_actual.png"),
                     ("08_10_scores.png", "09_scores.png"),
                     ("08_10_worst_misses.png", "10_worst_misses.png"),
                     ("08_10_worst_misses.csv", "worst_misses.csv")]:
        op, np_ = os.path.join(OUT, old), os.path.join(OUT, new)
        if os.path.exists(op):
            os.replace(op, np_)


# ------------------------------------------------------ model accuracy (true held-out test)
def test_plots():
    """
    Mirrors cv_plots(), but reads test_predictions.csv -- the circuits from
    split.py's test_labels.csv that the final model never trained on. This is
    the closest thing to a real competition score you can get before submitting.
    Only produced if you ran: python train.py --labels train_labels.csv --test test_labels.csv
    """
    rows = _load_pred_csv("test_predictions.csv")
    if len(rows) < 3:
        print("  (skip 13-15: too few test rows)")
        return
    _pred_vs_actual_plots(rows, "13_15", "held-out test",
                          "TRUE held-out test split")
    for old, new in [("13_15_pred_vs_actual.png", "13_pred_vs_actual.png"),
                     ("13_15_scores.png", "14_scores.png"),
                     ("13_15_worst_misses.png", "15_worst_misses.png"),
                     ("13_15_worst_misses.csv", "test_worst_misses.csv")]:
        op, np_ = os.path.join(OUT, old), os.path.join(OUT, new)
        if os.path.exists(op):
            os.replace(op, np_)


def importance_plot():
    rows = list(csv.DictReader(open("feature_importance.csv")))[:20]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh([r["feature"] for r in rows][::-1], [float(r["gain"]) for r in rows][::-1], color="#555")
    ax.set(xlabel="LightGBM gain", title="Top 20 features")
    save(fig, "11_feature_importance.png")


def ablation_plot():
    rows = list(csv.DictReader(open("ablation.csv")))
    full = float(rows[0]["cv_score"])
    rest = sorted(rows[1:], key=lambda r: float(r["cv_score"]))
    fig, ax = plt.subplots(figsize=(7, 4))
    drops = [(float(r["cv_score"]) - full) * 100 for r in rest]
    ax.barh([r["removed_group"] for r in rest][::-1], drops[::-1],
            color=["#d62728" if d < 0 else "#2ca02c" for d in drops[::-1]])
    ax.axvline(0, color="k", lw=0.8)
    ax.set(xlabel="change in CV score when group is removed (points)",
           title=f"Ablation (full model: {full:.1%})")
    save(fig, "12_ablation.png")


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
        print("Model plots (CV, train side):")
        cv_plots()
    if os.path.exists("test_predictions.csv"):
        print("Model plots (true held-out test):")
        test_plots()
    if os.path.exists("feature_importance.csv"):
        importance_plot()
    if os.path.exists("ablation.csv"):
        ablation_plot()