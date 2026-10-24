"""Statistical comparison of models and analysis of their errors.

Uses the per-image outputs written by ``evaluate.py`` and ``baseline.py`` to
answer the questions that decide model selection:

* Are differences in tray-level recall / false-reject rate between two
  inspectors real, or within sampling noise? (paired bootstrap)
* Which good trays are rejected, and is it the same tray for every model?
  (a tray rejected by every independent method suggests a labelling error)
* Which defect classes cause the missed trays?
* Mean AP over only the classes with enough test boxes to be trusted.

Writes ``results/compare/error_analysis.md`` and ``paired_tests.csv``.

Usage::

    python scripts/error_analysis.py --main y8n_640
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from porkqc.labels import load_class_names, read_label_file
from porkqc.metrics import paired_bootstrap_diff

RUNS = ["y8n_640", "y8s_640", "y8n_320", "y8n_960", "y8n_640_f50", "y8n_640_f25"]
#: (A, B) pairs: is B better than A?
PAIRS = [("baseline", "y8n_640"), ("y8n_640", "y8s_640"), ("y8n_640", "y8n_960"),
         ("y8n_320", "y8n_640"), ("y8n_640_f50", "y8n_640"), ("y8n_640_f25", "y8n_640")]


def load_predictions(results: Path) -> dict[str, pd.DataFrame]:
    """Per-image test decisions for every model and the baseline, aligned by image."""
    preds = {r: pd.read_csv(results / "eval" / r / "test_predictions.csv").set_index("image")
             for r in RUNS if (results / "eval" / r / "test_predictions.csv").exists()}
    preds["baseline"] = pd.read_csv(results / "baseline" / "test_predictions.csv").set_index("image")
    order = preds["baseline"].index
    return {k: v.loc[order] for k, v in preds.items()}


def main() -> None:
    """Run the analysis and write a markdown summary."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=Path("data/pork-rasher-v4"))
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--main", default="y8n_640")
    args = ap.parse_args()
    out = args.results / "compare"
    out.mkdir(parents=True, exist_ok=True)
    names = load_class_names(args.data / "data.yaml")
    preds = load_predictions(args.results)
    y = preds["baseline"].defective.values.astype(bool)

    def classes_in(img: str) -> Counter:
        f = args.data / "test" / "labels" / f"{Path(img).stem}.txt"
        return Counter(names[b.cls] for b in read_label_file(f, 720, 540))

    lines = ["# Error analysis and statistical comparison (test split)", ""]

    # 1. Paired bootstrap tests.
    rows = []
    for a, b in PAIRS:
        if a not in preds or b not in preds:
            continue
        for metric in ("recall", "false_reject_rate"):
            r = paired_bootstrap_diff(y, preds[a].rejected.values, preds[b].rejected.values, metric)
            rows.append({"A": a, "B": b, "metric": metric, **{k: round(v, 4) for k, v in r.items()}})
    tests = pd.DataFrame(rows)
    tests.to_csv(out / "paired_tests.csv", index=False)
    lines += ["## Paired bootstrap: metric(B) - metric(A), 95% CI", "",
              "For recall, positive = B catches more defective trays. For false_reject_rate, "
              "negative = B wastes fewer good trays. An interval containing 0 means the "
              "difference is not statistically established on this test set. `p_le_0` is the share "
              "of resamples with diff <= 0 (for recall, a one-sided p-value for 'B is better').", "",
              tests.to_markdown(index=False), ""]

    # 2. False rejects: which good trays, and are they shared across methods?
    lines += ["## Good trays rejected by each method", ""]
    shared = None
    for k, d in preds.items():
        fr = set(d.index[(~d.defective.astype(bool)) & d.rejected.astype(bool)])
        shared = fr if shared is None else shared & fr
        lines.append(f"- **{k}**: {len(fr)} — {', '.join(sorted(x.split('_jpg')[0] for x in fr))}")
    lines += ["", f"Rejected by **every** method: {', '.join(sorted(shared)) or 'none'}. "
              "A tray that every independent method flags is a candidate labelling error.", ""]

    # 3. Missed defective trays of the main model, by class.
    d = preds[args.main]
    missed = d.index[d.defective.astype(bool) & ~d.rejected.astype(bool)]
    lines += [f"## Defective trays missed by {args.main}", "",
              "| class | test trays containing it | caught | missed |", "|---|---|---|---|"]
    for c in names:
        imgs = [i for i in d.index if classes_in(i)[c] > 0]
        if imgs:
            caught = int(d.loc[imgs, "rejected"].astype(bool).sum())
            lines.append(f"| {c} | {len(imgs)} | {caught} | {len(imgs) - caught} |")
    lines += ["", f"Total missed: {len(missed)} of {int(y.sum())} defective trays.", ""]

    # 4. Mean AP over classes with enough test boxes (>= 20).
    lines += ["## AP50 on the reliably-evaluated classes only", "",
              "| model | unsealed | packaging-error | mean |", "|---|---|---|---|"]
    for r in RUNS:
        f = args.results / "eval" / r / "per_class.csv"
        if f.exists():
            pc = pd.read_csv(f).set_index("class")
            rel = pc[pc.reliable]["ap50_ultralytics"]
            lines.append(f"| {r} | {pc.loc['unsealed', 'ap50_ultralytics']:.3f} | "
                         f"{pc.loc['packaging-error', 'ap50_ultralytics']:.3f} | {rel.mean():.3f} |")

    # 5. Threshold sensitivity at fixed thresholds (explains model differences).
    lines += ["", "## Tray recall at fixed thresholds (test)", "",
              "| model | 0.15 | 0.20 | 0.25 | 0.30 | chosen on valid |", "|---|---|---|---|---|---|"]
    for r in RUNS:
        f = args.results / "eval" / r / "threshold_sweep.csv"
        if f.exists():
            s = pd.read_csv(f).query("split == 'test'").set_index("threshold")["recall"]
            v = pd.read_csv(f).query("split == 'valid'")
            thr = v.loc[v.youden_j.idxmax(), "threshold"]
            lines.append(f"| {r} | " + " | ".join(f"{s.loc[t]:.3f}" for t in (0.15, 0.2, 0.25, 0.3))
                         + f" | {thr} |")

    (out / "error_analysis.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    main()
