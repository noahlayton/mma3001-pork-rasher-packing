"""Combine all evaluation outputs into the comparison tables and figures.

Reads ``results/eval/*/``, ``results/baseline/``, ``results/training/`` and
the benchmark CSVs, and writes to ``results/compare/``:

* ``summary.csv`` / ``summary.md`` – one row per model with every headline number
* ``training_curves.png``  – validation mAP per epoch + train/val loss (overfitting check)
* ``tradeoff.png``         – test mAP50-95 vs CPU latency for every model
* ``resolution.png``       – per-class AP50 vs training image size
* ``data_size.png``        – test mAP vs fraction of training data
* ``threshold_sweep.png``  – tray recall / false-reject rate vs confidence threshold
* ``robustness.png``       – detection and tray metrics vs corruption severity

Usage::

    python scripts/compare.py --main y8n_640
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from porkqc.plotting import SERIES, TEXT_MUTED, apply_style

RUNS = ["y8n_640", "y8s_640", "y8n_320", "y8n_960", "y8n_640_f50", "y8n_640_f25"]
LABELS = {
    "y8n_640": "YOLOv8n 640", "y8s_640": "YOLOv8s 640", "y8n_320": "YOLOv8n 320",
    "y8n_960": "YOLOv8n 960", "y8n_640_f50": "YOLOv8n 640, 50% data",
    "y8n_640_f25": "YOLOv8n 640, 25% data",
}
#: Fixed colour per run so a model looks the same in every figure.
COLOUR = dict(zip(RUNS, SERIES))


def load_eval(root: Path) -> dict[str, dict]:
    """Read each model's ``metrics.json`` keyed by run name."""
    out = {}
    for r in RUNS:
        f = root / r / "metrics.json"
        if f.exists():
            out[r] = json.loads(f.read_text())
    return out


def summary_table(ev: dict, baseline: dict, bench_cpu: pd.DataFrame,
                  bench_gpu: pd.DataFrame) -> pd.DataFrame:
    """One row per model with accuracy, tray decision, cost and speed."""
    rows = []
    for r, m in ev.items():
        t, ci = m["test_tray"], m["test_tray_ci95"]
        cpu = bench_cpu[bench_cpu.model == r]
        gpu = bench_gpu[bench_gpu.model == r]
        rows.append({
            "model": LABELS[r],
            "test mAP50": m["test_map50_ultralytics"],
            "test mAP50-95": m["test_map50_95_ultralytics"],
            "mAP50 (own code)": m["test_map50_own"],
            "threshold": m["threshold"],
            "tray recall": f"{t['recall']:.3f} [{ci['recall'][0]:.2f}-{ci['recall'][1]:.2f}]",
            "false-reject rate": f"{t['false_reject_rate']:.3f} "
                                 f"[{ci['false_reject_rate'][0]:.2f}-{ci['false_reject_rate'][1]:.2f}]",
            "params (M)": cpu.params_M.iloc[0] if len(cpu) else np.nan,
            "GFLOPs": cpu.gflops.iloc[0] if len(cpu) else np.nan,
            "CPU ms": cpu.latency_median_ms.iloc[0] if len(cpu) else np.nan,
            "GPU ms": gpu.latency_median_ms.median() if len(gpu) else np.nan,
        })
    b, bci = baseline["test"], baseline["test_ci95"]
    rows.append({
        "model": "HOG + colour baseline", "test mAP50": np.nan, "test mAP50-95": np.nan,
        "mAP50 (own code)": np.nan, "threshold": 0.5,
        "tray recall": f"{b['recall']:.3f} [{bci['recall'][0]:.2f}-{bci['recall'][1]:.2f}]",
        "false-reject rate": f"{b['false_reject_rate']:.3f} "
                             f"[{bci['false_reject_rate'][0]:.2f}-{bci['false_reject_rate'][1]:.2f}]",
        "params (M)": round(baseline["n_features"] / 1e6, 4), "GFLOPs": np.nan,
        "CPU ms": baseline["latency_ms_median"], "GPU ms": np.nan,
    })
    return pd.DataFrame(rows)


def plot_training(train_dir: Path, main: str, out: Path) -> None:
    """Left: val mAP50-95 per epoch for every run. Right: main run's losses."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.2))
    for r in RUNS:
        f = train_dir / f"{r}_results.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        d.columns = [c.strip() for c in d.columns]
        a.plot(d.epoch, d["metrics/mAP50-95(B)"], color=COLOUR[r], label=LABELS[r])
    a.set_xlabel("epoch")
    a.set_ylabel("validation mAP50-95")
    a.set_title("Validation accuracy during training", loc="left")
    a.legend(fontsize=7)
    d = pd.read_csv(train_dir / f"{main}_results.csv")
    d.columns = [c.strip() for c in d.columns]
    b.plot(d.epoch, d["train/box_loss"] + d["train/cls_loss"], color=SERIES[0], label="train")
    b.plot(d.epoch, d["val/box_loss"] + d["val/cls_loss"], color=SERIES[1], label="validation")
    best = int(d.loc[d["metrics/mAP50-95(B)"].idxmax(), "epoch"])
    b.axvline(best, color=TEXT_MUTED, linestyle="--", linewidth=1)
    b.text(best, b.get_ylim()[1] * 0.97, f" best epoch {best}", fontsize=7, color=TEXT_MUTED,
           va="top")
    b.set_xlabel("epoch")
    b.set_ylabel("box + class loss")
    b.set_title(f"{LABELS[main]}: training vs validation loss", loc="left")
    b.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_tradeoff(summary: pd.DataFrame, ev: dict, out: Path) -> None:
    """Test mAP50-95 against CPU latency: the accuracy/speed trade-off."""
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for r in ev:
        row = summary[summary.model == LABELS[r]].iloc[0]
        ax.scatter(row["CPU ms"], row["test mAP50-95"], s=60, color=COLOUR[r], zorder=3,
                   edgecolor="white", linewidth=1.5)
        ax.annotate(LABELS[r], (row["CPU ms"], row["test mAP50-95"]), fontsize=7,
                    xytext=(6, 4), textcoords="offset points", color=TEXT_MUTED)
    ax.set_xlabel("CPU latency per image (ms, median)")
    ax.set_ylabel("test mAP50-95")
    ax.set_title("Accuracy vs speed", loc="left")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_resolution(eval_root: Path, out: Path) -> None:
    """Per-class test AP50 for models trained at 320 / 640 / 960 px."""
    runs = [("y8n_320", 320), ("y8n_640", 640), ("y8n_960", 960)]
    frames = {s: pd.read_csv(eval_root / r / "per_class.csv") for r, s in runs
              if (eval_root / r / "per_class.csv").exists()}
    if len(frames) < 2:
        return
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    sizes = sorted(frames)
    classes = frames[sizes[0]].query("test_boxes > 0")["class"].tolist()
    for k, c in enumerate(classes):
        y = [frames[s].set_index("class").loc[c, "ap50_ultralytics"] for s in sizes]
        n = int(frames[sizes[0]].set_index("class").loc[c, "test_boxes"])
        reliable = n >= 20
        ax.plot(sizes, y, marker="o", markersize=8, color=SERIES[k],
                linestyle="-" if reliable else ":",
                label=f"{c} ({n} test boxes{'' if reliable else ', unreliable'})")
    ax.set_xticks(sizes)
    ax.set_xlabel("training and inference image size (px)")
    ax.set_ylabel("test AP50")
    ax.set_ylim(0, 1.05)
    ax.set_title("Effect of input resolution per defect class", loc="left")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_data_size(ev: dict, out: Path) -> None:
    """Test mAP50 and mAP50-95 against the fraction of training images used."""
    pts = [(f, ev[r]) for r, f in [("y8n_640_f25", 25), ("y8n_640_f50", 50), ("y8n_640", 100)]
           if r in ev]
    if len(pts) < 2:
        return
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    x = [p[0] for p in pts]
    for k, (key, lab) in enumerate([("test_map50_ultralytics", "mAP50"),
                                    ("test_map50_95_ultralytics", "mAP50-95")]):
        ax.plot(x, [p[1][key] for p in pts], marker="o", markersize=8, color=SERIES[k], label=lab)
    ax.set_xticks(x, [f"{v}%\n(~{round(1085 * v / 100)} originals)" for v in x])
    ax.set_xlabel("share of training data")
    ax.set_ylabel("test score")
    ax.set_ylim(0, 1)
    ax.set_title("How much data is needed?", loc="left")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_threshold(eval_root: Path, main: str, out: Path) -> None:
    """Tray recall and false-reject rate vs threshold on valid and test."""
    d = pd.read_csv(eval_root / main / "threshold_sweep.csv")
    thr = json.loads((eval_root / main / "metrics.json").read_text())["threshold"]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for k, (col, lab) in enumerate([("recall", "defect catch rate (recall)"),
                                    ("false_reject_rate", "false-reject rate")]):
        for split, ls in [("valid", "--"), ("test", "-")]:
            s = d[d.split == split]
            ax.plot(s.threshold, s[col], color=SERIES[k], linestyle=ls,
                    label=f"{lab}, {split}")
    ax.axvline(thr, color=TEXT_MUTED, linewidth=1)
    ax.text(thr, 0.5, f" chosen on valid: {thr}", fontsize=7, color=TEXT_MUTED)
    ax.set_xlabel("confidence threshold for rejecting a tray")
    ax.set_ylabel("rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"{LABELS[main]}: choosing the PASS/FAIL threshold", loc="left")
    ax.legend(fontsize=7, loc="center right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_robustness(eval_root: Path, runs: list[str], out: Path) -> None:
    """Small multiples: one panel per corruption, mAP50 vs severity."""
    frames = {r: pd.read_csv(eval_root / r / "robustness.csv") for r in runs
              if (eval_root / r / "robustness.csv").exists()}
    if not frames:
        return
    kinds = list(next(iter(frames.values())).perturbation.unique())
    xlabels = {"brightness": "brightness factor (1 = original)", "blur": "blur sigma (px)",
               "noise": "noise std (grey levels)", "jpeg": "JPEG quality (100 = original)"}
    fig, axes = plt.subplots(2, len(kinds), figsize=(11, 4.8), sharey="row")
    for j, kind in enumerate(kinds):
        for r, d in frames.items():
            s = d[d.perturbation == kind]
            axes[0, j].plot(s.severity, s.map50, marker="o", markersize=5, color=COLOUR[r],
                            label=LABELS[r])
            axes[1, j].plot(s.severity, s.tray_recall, marker="o", markersize=5,
                            color=COLOUR[r])
        axes[0, j].set_title(kind, loc="left")
        axes[1, j].set_xlabel(xlabels.get(kind, kind), fontsize=8)
        if kind == "jpeg":
            axes[0, j].invert_xaxis()
            axes[1, j].invert_xaxis()
    axes[0, 0].set_ylabel("test mAP50")
    axes[1, 0].set_ylabel("tray defect catch rate")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Robustness to image corruption (test split)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    """Build every comparison table and figure."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--main", default="y8n_640", help="run treated as the selected model")
    args = ap.parse_args()
    out = args.results / "compare"
    out.mkdir(parents=True, exist_ok=True)
    apply_style()

    ev = load_eval(args.results / "eval")
    baseline = json.loads((args.results / "baseline" / "metrics.json").read_text())
    bench_cpu = pd.read_csv(args.results / "benchmark_cpu.csv")
    bench_gpu = pd.read_csv(args.results / "benchmark_gpu.csv")

    summary = summary_table(ev, baseline, bench_cpu, bench_gpu)
    summary.to_csv(out / "summary.csv", index=False)
    (out / "summary.md").write_text(summary.to_markdown(index=False))
    print(summary.to_string(index=False))

    plot_training(args.results / "training", args.main, out / "training_curves.png")
    plot_tradeoff(summary, ev, out / "tradeoff.png")
    plot_resolution(args.results / "eval", out / "resolution.png")
    plot_data_size(ev, out / "data_size.png")
    plot_threshold(args.results / "eval", args.main, out / "threshold_sweep.png")
    plot_robustness(args.results / "eval", [args.main, "y8s_640"], out / "robustness.png")
    print(f"saved to {out}/")


if __name__ == "__main__":
    main()
