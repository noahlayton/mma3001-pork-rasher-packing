"""Measure inference speed, model size, FLOPs and memory of trained detectors.

For each weights file, a fixed set of test images is run through the model
one image at a time (batch size 1, as on a production line where trays arrive
one by one). Latency statistics come from :func:`porkqc.timing.benchmark`.

Example::

    python scripts/benchmark.py runs/y8n_640/weights/best.pt runs/y8s_640/weights/best.pt \
        --device cpu --out results/benchmark_cpu.csv
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import pandas as pd

from porkqc.timing import benchmark


def peak_memory_mb(device: str) -> float:
    """Peak memory in MB: GPU allocator peak on CUDA, process max RSS on CPU."""
    if device != "cpu":
        import torch

        return torch.cuda.max_memory_allocated() / 2**20
    import resource

    # ru_maxrss is in kilobytes on Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main() -> None:
    """Benchmark each weights file and append one row per model to a CSV."""
    ap = argparse.ArgumentParser(description="Benchmark YOLO inference")
    ap.add_argument("weights", nargs="+", type=Path)
    ap.add_argument("--data", type=Path, default=Path("data/pork-rasher-v4"))
    ap.add_argument("--imgsz", type=int, nargs="*", default=None,
                    help="inference sizes; default = the size each model was trained at")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-images", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=3, help="passes over the image set")
    ap.add_argument("--threads", type=int, default=None, help="limit CPU threads")
    ap.add_argument("--out", type=Path, default=Path("results/benchmark.csv"))
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops

    if args.threads:
        torch.set_num_threads(args.threads)
    imgs = [cv2.imread(str(p)) for p in
            sorted((args.data / "test" / "images").glob("*.jpg"))[: args.n_images]]

    rows = []
    for w in args.weights:
        model = YOLO(str(w))
        n_params = sum(p.numel() for p in model.model.parameters())
        trained_sz = int(model.ckpt.get("train_args", {}).get("imgsz", 640))
        for sz in args.imgsz or [trained_sz]:
            i = iter(range(10**9))

            def run_one() -> None:
                # Cycle through the test images so caching cannot help.
                model.predict(imgs[next(i) % len(imgs)], imgsz=sz, device=args.device,
                              verbose=False)

            if args.device != "cpu":
                torch.cuda.reset_peak_memory_stats()
            t = benchmark(run_one, repeats=args.repeats * len(imgs), warmup=5)
            rows.append({
                "model": w.parent.parent.name,
                "device": args.device if args.device == "cpu" else torch.cuda.get_device_name(),
                "imgsz": sz,
                "params_M": round(n_params / 1e6, 3),
                "gflops": round(get_flops(model.model, sz), 2),
                "weights_MB": round(os.path.getsize(w) / 2**20, 2),
                "latency_median_ms": round(t["median"], 2),
                "latency_p95_ms": round(t["p95"], 2),
                "fps": round(t["fps"], 1),
                "peak_mem_MB": round(peak_memory_mb(args.device), 1),
            })
            print(rows[-1])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, mode="a", header=not args.out.exists(), index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
