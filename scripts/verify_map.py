"""Verify the independent AP implementation against Ultralytics on identical inputs.

Ultralytics ``val()`` is run with ``save_json=True`` so its own predictions are
written to disk. Those exact predictions are then scored with
:func:`porkqc.metrics.evaluate_map`. Any difference can now only come from the
metric code (matching + integration), not from the model or preprocessing.
For comparison, the same model is also scored on boxes from ``predict()``,
which pre-processes images slightly differently.

Writes ``results/compare/map_verification.csv``.

Usage::

    python scripts/verify_map.py runs/y8n_640/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import pandas as pd

from porkqc.labels import Box, load_class_names, read_label_file
from porkqc.metrics import evaluate_map
from porkqc.yolo import predict_boxes


def main() -> None:
    """Score Ultralytics' own predictions with our code and compare."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("weights", type=Path)
    ap.add_argument("--data", type=Path, default=Path("data/pork-rasher-v4"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=Path("results/compare/map_verification.csv"))
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).parent))
    from train import write_data_yaml
    from ultralytics import YOLO

    names = load_class_names(args.data / "data.yaml")
    model = YOLO(str(args.weights))
    imgsz = int(model.ckpt.get("train_args", {}).get("imgsz", 640))
    data_yaml = write_data_yaml(args.data, Path("runs/data_abs.yaml"))
    ul = model.val(data=str(data_yaml), split="test", imgsz=imgsz, device=args.device,
                   save_json=True, plots=False, verbose=False, project="runs/verify",
                   name=args.weights.parent.parent.name, exist_ok=True)

    # Rebuild Ultralytics' predictions (COCO JSON: xywh pixels, 1-based category ids).
    pj = json.loads((Path(ul.save_dir) / "predictions.json").read_text())
    root = args.data / "test"
    stems = sorted(p.stem for p in (root / "images").glob("*.jpg"))
    by_img: dict[str, list[Box]] = {s: [] for s in stems}
    for p in pj:
        x, y, w, h = p["bbox"]
        by_img[str(p["image_id"])].append(Box(p["category_id"] - 1, x, y, x + w, y + h, p["score"]))
    gts = [read_label_file(root / "labels" / f"{s}.txt", 720, 540) for s in stems]

    own_same = evaluate_map([by_img[s] for s in stems], gts, len(names))
    imgs = [cv2.imread(str(root / "images" / f"{s}.jpg")) for s in stems]
    own_predict = evaluate_map(predict_boxes(model, imgs, imgsz, device=args.device), gts,
                               len(names))
    ul_ap = dict(zip(ul.box.ap_class_index.tolist(), ul.box.ap50.tolist()))

    df = pd.DataFrame({
        "class": names + ["mean (mAP50)"],
        "ultralytics_val": [ul_ap.get(c) for c in range(len(names))] + [ul.box.map50],
        "own_on_same_predictions": list(own_same["ap"]) + [own_same["map"]],
        "own_on_predict_boxes": list(own_predict["ap"]) + [own_predict["map"]],
    })
    df["abs_diff_same_inputs"] = (df.own_on_same_predictions - df.ultralytics_val).abs()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(args.out, index=False)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
