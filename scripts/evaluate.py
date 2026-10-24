"""Evaluate trained YOLO models: mAP cross-check, PASS/FAIL decision, robustness.

For each weights file this script:

1. predicts on the valid and test splits (conf >= 0.001, like Ultralytics val);
2. computes mAP50 and mAP50-95 with the independent implementation in
   :mod:`porkqc.metrics` and compares them with Ultralytics' own ``val()``
   numbers on the test split (verification of the evaluation code);
3. sweeps the tray-level confidence threshold; the operating threshold is
   chosen on **valid** (maximum Youden J = recall - false-reject rate) and then
   applied once to **test**, reported with 95% bootstrap confidence intervals;
4. optionally (``--robustness``) repeats the test evaluation on corrupted
   images (brightness, blur, noise, JPEG) at increasing severity.

Outputs go to ``results/eval/<run name>/``.

Example::

    python scripts/evaluate.py runs/y8n_640/weights/best.pt --robustness
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from porkqc.labels import load_class_names, read_label_file
from porkqc.metrics import binary_metrics, bootstrap_ci, evaluate_map
from porkqc.perturb import PERTURBATIONS
from porkqc.yolo import predict_boxes, tray_scores

IOU_THRESHOLDS = np.arange(0.5, 0.96, 0.05)
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)


def load_split(root: Path, split: str):
    """Images (BGR), ground-truth boxes and image names for one split."""
    paths = sorted((root / split / "images").glob("*.jpg"))
    imgs = [cv2.imread(str(p)) for p in paths]
    gts = [read_label_file(root / split / "labels" / f"{p.stem}.txt",
                           im.shape[1], im.shape[0]) for p, im in zip(paths, imgs)]
    return imgs, gts, [p.name for p in paths]


def map_scores(preds, gts, n_classes: int) -> dict:
    """mAP50, mAP50-95 and per-class AP50 from the independent implementation."""
    per_iou = [evaluate_map(preds, gts, n_classes, t) for t in IOU_THRESHOLDS]
    ap50 = per_iou[0]["ap"]
    ap_all = np.vstack([r["ap"] for r in per_iou])  # (n_iou, n_classes)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        # classes with no test boxes (wrinkle) are all-NaN columns -> NaN, by design
        warnings.simplefilter("ignore", RuntimeWarning)
        ap50_95 = np.nanmean(ap_all, axis=0)
    return {
        "map50": per_iou[0]["map"],
        "map50_95": float(np.nanmean(ap50_95)),
        "ap50": ap50,
        "ap50_95": ap50_95,
    }


def sweep(scores: np.ndarray, defective: np.ndarray) -> pd.DataFrame:
    """Tray metrics at every candidate confidence threshold."""
    rows = []
    for t in THRESHOLDS:
        m = binary_metrics(defective, scores >= t)
        rows.append({"threshold": t, **m, "youden_j": m["recall"] - m["false_reject_rate"]})
    return pd.DataFrame(rows)


def evaluate(weights: Path, args, names: list[str]) -> dict:
    """Run every evaluation step for one model and save its outputs."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    run = weights.parent.parent.name
    imgsz = args.imgsz or int(model.ckpt.get("train_args", {}).get("imgsz", 640))
    out = args.out / (run if args.imgsz is None else f"{run}@{imgsz}")
    out.mkdir(parents=True, exist_ok=True)
    defect_ids = list(range(len(names)))  # every class in this dataset is a defect
    print(f"\n=== {run} (imgsz {imgsz}) ===")

    res: dict = {"run": run, "imgsz": imgsz}
    preds, gts, defective = {}, {}, {}
    for split in ("valid", "test"):
        imgs, gts[split], _ = load_split(args.data, split)
        preds[split] = predict_boxes(model, imgs, imgsz, device=args.device)
        defective[split] = np.array([len(g) > 0 for g in gts[split]])

    # 1-2. Object-level accuracy and cross-check against Ultralytics.
    own = map_scores(preds["test"], gts["test"], len(names))
    ul = model.val(data=str(args.data_yaml), split="test", imgsz=imgsz, device=args.device,
                   verbose=False, plots=False, project=str(args.out / "_ul"), name=run,
                   exist_ok=True)
    res["test_map50_own"] = round(own["map50"], 4)
    res["test_map50_95_own"] = round(own["map50_95"], 4)
    res["test_map50_ultralytics"] = round(float(ul.box.map50), 4)
    res["test_map50_95_ultralytics"] = round(float(ul.box.map), 4)
    ul_ap50 = dict(zip(ul.box.ap_class_index.tolist(), ul.box.ap50.tolist()))
    n_gt = np.bincount([b.cls for g in gts["test"] for b in g], minlength=len(names))
    per_class = pd.DataFrame({
        "class": names, "test_boxes": n_gt,
        "ap50_own": own["ap50"].round(4), "ap50_95_own": own["ap50_95"].round(4),
        "ap50_ultralytics": [round(ul_ap50.get(c, np.nan), 4) for c in range(len(names))],
    })
    per_class["reliable"] = per_class.test_boxes >= 20  # too few boxes -> AP is mostly noise
    per_class.to_csv(out / "per_class.csv", index=False)
    print(per_class.to_string(index=False))
    print(f"mAP50 own={res['test_map50_own']:.4f} ultralytics={res['test_map50_ultralytics']:.4f}")

    # 3. Tray-level PASS/FAIL: choose threshold on valid, apply once to test.
    scores = {s: tray_scores(preds[s], defect_ids) for s in preds}
    sv, st = sweep(scores["valid"], defective["valid"]), sweep(scores["test"], defective["test"])
    sv.assign(split="valid").pipe(lambda d: pd.concat([d, st.assign(split="test")])).to_csv(
        out / "threshold_sweep.csv", index=False)
    thr = float(sv.loc[sv.youden_j.idxmax(), "threshold"])
    rejected = scores["test"] >= thr
    res["threshold"] = thr
    res["test_tray"] = binary_metrics(defective["test"], rejected)
    res["test_tray_ci95"] = {k: bootstrap_ci(defective["test"], rejected, k)[1:] for k in
                             ["accuracy", "recall", "precision", "f1", "false_reject_rate"]}
    m = res["test_tray"]
    print(f"threshold {thr} (chosen on valid): test recall={m['recall']:.3f} "
          f"false-reject={m['false_reject_rate']:.3f} accuracy={m['accuracy']:.3f}")
    pd.DataFrame({"image": load_split(args.data, "test")[2], "defective": defective["test"],
                  "score": scores["test"].round(4), "rejected": rejected}).to_csv(
        out / "test_predictions.csv", index=False)

    # 4. Robustness to image corruptions (test split, fixed threshold).
    if args.robustness:
        imgs = load_split(args.data, "test")[0]
        rows = []
        for pname, (fn, levels) in PERTURBATIONS.items():
            for lvl in levels:
                p = predict_boxes(model, [fn(im, lvl) for im in imgs], imgsz, device=args.device)
                tm = binary_metrics(defective["test"], tray_scores(p, defect_ids) >= thr)
                rows.append({"perturbation": pname, "severity": lvl,
                             "map50": round(evaluate_map(p, gts["test"], len(names))["map"], 4),
                             "tray_recall": round(tm["recall"], 4),
                             "false_reject_rate": round(tm["false_reject_rate"], 4)})
                print(rows[-1])
        pd.DataFrame(rows).to_csv(out / "robustness.csv", index=False)

    (out / "metrics.json").write_text(json.dumps(res, indent=2, default=float))
    return res


def main() -> None:
    """Evaluate each weights file passed on the command line."""
    ap = argparse.ArgumentParser(description="Evaluate trained YOLO models")
    ap.add_argument("weights", nargs="+", type=Path)
    ap.add_argument("--data", type=Path, default=Path("data/pork-rasher-v4"))
    ap.add_argument("--imgsz", type=int, default=None,
                    help="override inference size (default: training size)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--robustness", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("results/eval"))
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from train import write_data_yaml

    args.data_yaml = write_data_yaml(args.data, Path("runs/data_abs.yaml"))
    names = load_class_names(args.data / "data.yaml")
    for w in args.weights:
        evaluate(w, args, names)


if __name__ == "__main__":
    main()
