"""Fine-tune a pretrained YOLOv8 detector on the pork rasher dataset.

Transfer learning: the network starts from COCO-pretrained weights and every
layer is fine-tuned on our 5 defect classes. All settings are fixed by command
line arguments and a random seed so a run can be repeated exactly.

Examples::

    # main model (run on a GPU, e.g. Google Colab)
    python scripts/train.py --model yolov8n.pt --epochs 50 --imgsz 640 --name y8n_640

    # quick CPU smoke test of the whole pipeline
    python scripts/train.py --model yolov8n.pt --epochs 1 --imgsz 320 --fraction 0.05 \
        --device cpu --name smoke
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def write_data_yaml(root: Path, out: Path) -> Path:
    """Write a data.yaml with absolute split paths.

    The Roboflow export uses paths like ``../train/images`` that resolve
    differently depending on the working directory; absolute paths avoid that.

    Args:
        root: Dataset root containing ``train/``, ``valid/``, ``test/`` and
            the original ``data.yaml``.
        out: Where to write the new YAML file.

    Returns:
        The path written.
    """
    cfg = yaml.safe_load((root / "data.yaml").read_text())
    root = root.resolve()
    fixed = {
        "path": str(root),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": cfg["nc"],
        "names": cfg["names"],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(fixed, sort_keys=False))
    return out


def main() -> None:
    """Parse arguments, train, then evaluate the best weights on the test split."""
    ap = argparse.ArgumentParser(description="Train YOLOv8 on the pork rasher dataset")
    ap.add_argument("--data", type=Path, default=Path("data/pork-rasher-v4"))
    ap.add_argument("--model", default="yolov8n.pt", help="pretrained weights to start from")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640, help="training image size (pixels)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=15,
                    help="stop early if val mAP has not improved for this many epochs")
    ap.add_argument("--fraction", type=float, default=1.0,
                    help="fraction of training images to use (data-requirement study)")
    ap.add_argument("--device", default=None, help="'cpu', '0' for first GPU, or None for auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--project", type=Path, default=Path("runs"))
    ap.add_argument("--name", required=True, help="run name, e.g. y8n_640")
    args = ap.parse_args()

    from ultralytics import YOLO  # imported here so --help works without torch

    data_yaml = write_data_yaml(args.data, args.project / "data_abs.yaml")
    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        fraction=args.fraction,
        device=args.device,
        seed=args.seed,
        deterministic=True,
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=True,
        plots=True,
    )
    # Final, unbiased numbers come from the test split, which was never used
    # for training or for choosing the best epoch.
    best = args.project / args.name / "weights" / "best.pt"
    metrics = YOLO(str(best)).val(data=str(data_yaml), split="test", imgsz=args.imgsz,
                                  device=args.device, project=str(args.project.resolve()),
                                  name=f"{args.name}_test", exist_ok=True)
    print(f"TEST mAP50={metrics.box.map50:.4f} mAP50-95={metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
