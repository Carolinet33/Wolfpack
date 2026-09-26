#!/usr/bin/env python3
"""
split.py  --  split runtime-data.csv into train / test BY CIRCUIT.

  python split.py                  # 80/20 split, seed 42
  python split.py --test-frac 0.25 --seed 7

Why by circuit: each circuit appears at up to 3 thresholds. If circuit X at
threshold 16 were in train and X at 512 in test, the model would have already
"seen" X and the test score would be inflated. The real hold-out contains
circuits the model has never seen, so all rows of a circuit go to the same side.

Why stratified: runtimes span 5 orders of magnitude and only ~33 runs time out.
A purely random split could put most slow circuits or most timeouts on one
side. We bin circuits by (has a timeout?, typical log-runtime) and split each
bin 80/20, so both sides look like the full dataset.

Outputs: train_labels.csv, test_labels.csv (same columns as runtime-data.csv)
"""
import argparse
import csv
import math
import random
from collections import defaultdict

CAP = 14400.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="runtime-data.csv")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.labels, newline="") as f:
        reader = csv.DictReader(l.replace("\r\n", "\n") for l in f)
        fields = reader.fieldnames
        rows = list(reader)

    by_circ = defaultdict(list)
    for r in rows:
        by_circ[r["filename"].strip()].append(r)

    # circuit-level stratum: timeout flag + quartile of median log10 runtime
    info = {}
    for fn, rs in by_circ.items():
        to = any(r["status"].strip().lower() == "timeout" for r in rs)
        ys = sorted(CAP if r["status"].strip().lower() == "timeout"
                    else float(r["duration_s"] or CAP) for r in rs)
        info[fn] = (to, math.log10(max(ys[len(ys) // 2], 1e-3)))
    meds = sorted(v[1] for v in info.values())
    q = [meds[int(len(meds) * k / 4)] for k in (1, 2, 3)]
    strata = defaultdict(list)
    for fn, (to, m) in info.items():
        strata["timeout" if to else f"q{sum(m > x for x in q)}"].append(fn)

    rng = random.Random(args.seed)
    test = set()
    for name, fns in sorted(strata.items()):
        fns = sorted(fns)
        rng.shuffle(fns)
        n_test = max(1, round(len(fns) * args.test_frac)) if len(fns) > 1 else 0
        test.update(fns[:n_test])

    for path, keep in (("train_labels.csv", lambda f: f not in test),
                       ("test_labels.csv", lambda f: f in test)):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                if keep(r["filename"].strip()):
                    w.writerow(r)

    def summary(sel):
        rs = [r for fn in sel for r in by_circ[fn]]
        n_to = sum(r["status"].strip().lower() == "timeout" for r in rs)
        ys = sorted(float(r["duration_s"]) for r in rs if r["duration_s"].strip())
        return f"{len(sel):>4} circuits, {len(rs):>5} runs, {n_to:>3} timeouts, median {ys[len(ys)//2]:.2f} s"

    train = set(by_circ) - test
    print("train:", summary(train))
    print("test :", summary(test))
    print("strata:", {k: len(v) for k, v in sorted(strata.items())})
    print("Wrote train_labels.csv and test_labels.csv")


if __name__ == "__main__":
    main()
