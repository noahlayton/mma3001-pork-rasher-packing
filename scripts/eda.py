"""Exploratory data analysis of the pork rasher dataset.

Produces, in ``results/eda/``:

* ``class_counts.csv``   – boxes and images per class for each split
* ``class_counts.png``   – the same as small-multiple bar charts
* ``box_sizes.png``      – distribution of box area (as % of image) per class
* ``leakage_check.csv``  – nearest-training-image similarity for valid/test
* ``leakage_check.png``  – that similarity compared with a train-vs-train baseline
* ``samples.jpg``        – one annotated example of each class

Usage::

    python scripts/eda.py --data data/pork-rasher-v4
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from porkqc.labels import load_class_names, read_label_file, split_stats
from porkqc.plotting import SERIES, TEXT_MUTED, apply_style

SPLITS = ["train", "valid", "test"]
IMG_W, IMG_H = 720, 540


def class_count_table(root: Path, names: list[str]) -> tuple[pd.DataFrame, dict]:
    """Boxes and images per class for every split, plus raw stats per split."""
    rows, stats = [], {}
    for s in SPLITS:
        st = split_stats(root / s / "labels", IMG_W, IMG_H)
        stats[s] = st
        for c, n in enumerate(names):
            rows.append({"split": s, "class": n, "boxes": st["class_counts"][c],
                         "images": st["images_per_class"][c]})
        rows.append({"split": s, "class": "(no defect)", "boxes": 0, "images": st["n_empty"]})
    return pd.DataFrame(rows), stats


def plot_class_counts(df: pd.DataFrame, names: list[str], out: Path) -> None:
    """Small multiples: one panel per split, each with its own x scale."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.8), sharey=True)
    order = names[::-1]
    for ax, s in zip(axes, SPLITS):
        d = df[(df.split == s) & (df["class"].isin(names))].set_index("class").loc[order]
        ax.barh(order, d.boxes, color=SERIES[0], height=0.6)
        for y, v in enumerate(d.boxes):
            ax.text(v, y, f" {v}", va="center", fontsize=8, color=TEXT_MUTED)
        ax.set_title(s)
        ax.set_xlabel("annotated boxes")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, d.boxes.max() * 1.25 + 1)
    fig.suptitle("Boxes per class and split — rare classes have almost no test examples",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def box_areas(root: Path, n_classes: int) -> dict[int, list[float]]:
    """Relative box area (box area / image area) per class over all splits."""
    areas: dict[int, list[float]] = {c: [] for c in range(n_classes)}
    for s in SPLITS:
        for f in (root / s / "labels").glob("*.txt"):
            for b in read_label_file(f, IMG_W, IMG_H):
                areas[b.cls].append(100 * b.area / (IMG_W * IMG_H))
    return areas


def plot_box_sizes(areas: dict[int, list[float]], names: list[str], out: Path) -> None:
    """Horizontal box plot of relative box area per class on a log axis."""
    fig, ax = plt.subplots(figsize=(7, 2.8))
    data = [areas[c] for c in range(len(names))][::-1]
    bp = ax.boxplot(data, orientation="horizontal", widths=0.5, patch_artist=True,
                    medianprops={"color": "#0b0b0b"}, flierprops={"markersize": 2})
    for patch in bp["boxes"]:
        patch.set_facecolor(SERIES[0])
        patch.set_alpha(0.6)
    ax.set_yticks(range(1, len(names) + 1), names[::-1])
    ax.set_xscale("log")
    ax.set_xlabel("box area (% of image, log scale)")
    ax.set_title("Defect size varies by two orders of magnitude between classes", loc="left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _thumbnails(paths: list[Path]) -> np.ndarray:
    """48x36 grey thumbnails, each normalised to zero mean / unit variance.

    Normalising removes global brightness, so the ±10% exposure augmentation
    applied to the training images does not hide genuine duplicates.
    """
    x = np.stack([
        cv2.resize(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE), (48, 36),
                   interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
        for p in paths
    ])
    return (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)


def _source(p: Path) -> str:
    """Original (pre-augmentation) image name, e.g. ``frame_with_red_102_jpg``."""
    return re.sub(r"\.rf\..*$", "", p.name)


def leakage_check(root: Path, out_dir: Path, seed: int = 0) -> pd.DataFrame:
    """Find each valid/test image's most similar training image.

    Similarity is the Pearson correlation between normalised thumbnails. The
    result is compared with a baseline: the same statistic for training images
    against *other* training images (excluding their own augmented copies).
    If held-out images were near-duplicates of training images, their best
    correlation would be clearly higher than this baseline.
    """
    train = sorted((root / "train" / "images").glob("*.jpg"))
    xt = _thumbnails(train)
    src = np.array([_source(p) for p in train])
    rows, best = [], {}
    for s in ["valid", "test"]:
        paths = sorted((root / s / "images").glob("*.jpg"))
        corr = _thumbnails(paths) @ xt.T / xt.shape[1]
        best[s] = corr.max(1)
        for p, j, c in zip(paths, corr.argmax(1), best[s]):
            rows.append({"split": s, "image": p.name, "nearest_train": train[j].name,
                         "correlation": round(float(c), 4)})
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(train), 300, replace=False)
    c = xt[idx] @ xt.T / xt.shape[1]
    for k, i in enumerate(idx):
        c[k, src == src[i]] = -1.0  # exclude the image's own augmented copies
    best["train (baseline)"] = c.max(1)

    fig, ax = plt.subplots(figsize=(7, 2.8))
    bins = np.linspace(0.6, 1.0, 41)
    for col, (k, v) in zip(SERIES, best.items()):
        ax.hist(v, bins=bins, density=True, histtype="step", color=col, linewidth=2,
                label=f"{k}: median {np.median(v):.3f}")
    ax.axvline(0.98, color=TEXT_MUTED, linestyle="--", linewidth=1)
    ax.text(0.981, ax.get_ylim()[1] * 0.9, "near-duplicate\nthreshold", fontsize=7,
            color=TEXT_MUTED, va="top")
    ax.set_xlabel("correlation with most similar training image")
    ax.set_ylabel("density")
    ax.set_title("No held-out image is a near-duplicate of a training image", loc="left")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "leakage_check.png")
    plt.close(fig)
    return pd.DataFrame(rows)


def sample_grid(root: Path, names: list[str], out: Path) -> None:
    """Save one annotated training example per class plus one defect-free image."""
    bgr = [tuple(int(h[i:i + 2], 16) for i in (5, 3, 1)) for h in SERIES]
    labels = sorted((root / "train" / "labels").glob("*.txt"))
    picks = []
    for c in range(len(names)):
        picks.append(next(f for f in labels if any(
            ln.split()[0] == str(c) for ln in f.read_text().splitlines() if ln.strip())))
    picks.append(next(f for f in labels if not f.read_text().strip()))
    tiles = []
    for f in picks:
        img = cv2.imread(str(root / "train" / "images" / (f.stem + ".jpg")))
        for b in read_label_file(f, IMG_W, IMG_H):
            p1, p2 = (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2))
            cv2.rectangle(img, p1, p2, bgr[b.cls], 3)
            cv2.putText(img, names[b.cls], (p1[0], max(22, p1[1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr[b.cls], 2)
        if not f.read_text().strip():
            cv2.putText(img, "no defect (PASS)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.1, (255, 255, 255), 2)
        tiles.append(cv2.resize(img, (360, 270)))
    grid = np.vstack([np.hstack(tiles[0:3]), np.hstack(tiles[3:6])])
    cv2.imwrite(str(out), grid)


def main() -> None:
    """Run every EDA step and print a short summary."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=Path("data/pork-rasher-v4"))
    ap.add_argument("--out", type=Path, default=Path("results/eda"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    apply_style()

    names = load_class_names(args.data / "data.yaml")
    df, stats = class_count_table(args.data, names)
    df.to_csv(args.out / "class_counts.csv", index=False)
    print(df.pivot(index="class", columns="split", values="boxes")[SPLITS])
    plot_class_counts(df, names, args.out / "class_counts.png")
    plot_box_sizes(box_areas(args.data, len(names)), names, args.out / "box_sizes.png")

    leak = leakage_check(args.data, args.out)
    leak.to_csv(args.out / "leakage_check.csv", index=False)
    print(leak.groupby("split").correlation.describe())
    sample_grid(args.data, names, args.out / "samples.jpg")
    print(f"saved figures and tables to {args.out}/")


if __name__ == "__main__":
    main()
