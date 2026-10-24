"""Train and evaluate the classical HOG + colour-histogram PASS/FAIL baseline.

Protocol (no information from the test split is used to build the model):

1. Extract features for train / valid / test (cached to ``runs/baseline_features.npz``).
2. Fit a logistic regression on *train* for each regularisation strength C.
3. Pick C with the best F1 on *valid*. Cross-validation inside train is not
   used because each source image appears three times there (augmentation),
   so folds would leak.
4. Report test-split metrics with 95% bootstrap confidence intervals and the
   per-image latency of the full pipeline (feature extraction + prediction).

Outputs ``results/baseline/metrics.json`` and ``results/baseline/test_predictions.csv``.

Usage::

    python scripts/baseline.py --data data/pork-rasher-v4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from porkqc.baseline import extract_features, features_for_paths, make_classifier
from porkqc.metrics import binary_metrics, bootstrap_ci
from porkqc.timing import benchmark

C_GRID = [0.0001, 0.001, 0.01, 0.1, 1.0]


def load_split(root: Path, split: str, cache: dict) -> tuple[list[Path], np.ndarray, np.ndarray]:
    """Image paths, feature matrix and tray labels (True = defective) for a split."""
    paths = sorted((root / split / "images").glob("*.jpg"))
    labels = np.array([
        bool((root / split / "labels" / f"{p.stem}.txt").read_text().strip()) for p in paths
    ])
    if split not in cache:
        print(f"extracting features for {split} ({len(paths)} images)...")
        cache[split] = features_for_paths(paths)
    return paths, cache[split], labels


def main() -> None:
    """Run the baseline protocol and write the results."""
    ap = argparse.ArgumentParser(description="HOG + colour histogram baseline")
    ap.add_argument("--data", type=Path, default=Path("data/pork-rasher-v4"))
    ap.add_argument("--cache", type=Path, default=Path("runs/baseline_features.npz"))
    ap.add_argument("--out", type=Path, default=Path("results/baseline"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cache = dict(np.load(args.cache)) if args.cache.exists() else {}
    tr_p, x_tr, y_tr = load_split(args.data, "train", cache)
    _, x_va, y_va = load_split(args.data, "valid", cache)
    te_p, x_te, y_te = load_split(args.data, "test", cache)
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.cache, **cache)

    # Model selection on the validation split.
    sweep = []
    for c in C_GRID:
        clf = make_classifier(c).fit(x_tr, y_tr)
        m = binary_metrics(y_va, clf.predict(x_va).astype(bool))
        sweep.append({"C": c, **{k: round(v, 4) for k, v in m.items()}})
        print(f"C={c:<7} valid F1={m['f1']:.3f} recall={m['recall']:.3f} "
              f"false-reject={m['false_reject_rate']:.3f}")
    best_c = max(sweep, key=lambda r: r["f1"])["C"]
    clf = make_classifier(best_c).fit(x_tr, y_tr)

    # Final, one-off evaluation on the test split.
    prob = clf.predict_proba(x_te)[:, 1]
    pred = prob >= 0.5
    test = binary_metrics(y_te, pred)
    ci = {k: bootstrap_ci(y_te, pred, k)[1:] for k in
          ["accuracy", "recall", "precision", "f1", "false_reject_rate"]}

    img = cv2.imread(str(te_p[0]))
    t = benchmark(lambda: clf.predict_proba(extract_features(img)[None]), repeats=50)

    result = {
        "chosen_C": best_c,
        "valid_sweep": sweep,
        "test": test,
        "test_ci95": ci,
        "n_test": int(len(y_te)),
        "n_test_defective": int(y_te.sum()),
        # Trivial reference: reject every tray. High accuracy on this
        # defect-heavy test set but useless in practice (100% false rejects).
        "reject_all_test": binary_metrics(y_te, np.ones_like(y_te)),
        "latency_ms_median": round(t["median"], 2),
        "n_features": int(x_tr.shape[1]),
    }
    (args.out / "metrics.json").write_text(json.dumps(result, indent=2))
    pd.DataFrame({"image": [p.name for p in te_p], "defective": y_te,
                  "prob_defective": prob.round(4), "rejected": pred}).to_csv(
        args.out / "test_predictions.csv", index=False)

    print(f"\nchosen C = {best_c}")
    for k, (lo, hi) in ci.items():
        print(f"test {k:18s} {test[k]:.3f}  (95% CI {lo:.3f}-{hi:.3f})")
    print(f"confusion: TP={test['tp']} FP={test['fp']} FN={test['fn']} TN={test['tn']}")
    ra = result["reject_all_test"]
    print(f"reject-all rule: accuracy={ra['accuracy']:.3f} recall={ra['recall']:.3f} "
          f"false-reject={ra['false_reject_rate']:.3f}")
    print(f"latency {t['median']:.1f} ms/image (CPU)")


if __name__ == "__main__":
    main()
