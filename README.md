# Automated Detection of Pork Rasher Packaging Defects

MMA3001 Numerical Methods and Machine Learning — individual project, Monash University (2026).
Author: Noah Layton

## Purpose

Food-packaging lines still rely heavily on manual visual inspection. This project
builds a computer-vision inspector that takes an RGB image of a packaged tray of
pork rashers and:

1. **detects and localises packaging/product-placement defects** with a YOLOv8
   object detector (bounding box + class + confidence for each defect), and
2. **converts the detections into a tray-level PASS/FAIL decision** with a
   transparent, per-class confidence-threshold rule.

The detector is compared against a baseline and alternative model sizes,
validated on a withheld test split, stress-tested against lighting, blur, noise
and compression, and benchmarked for production-line speed.

## Repository structure

```
├── src/porkqc/        Python package (all reusable code, documented with docstrings)
│   ├── labels.py      YOLO label parsing and dataset statistics
│   ├── metrics.py     IoU, Average Precision (independent mAP check), PASS/FAIL metrics
│   ├── decision.py    detections -> tray PASS/FAIL rule
│   ├── perturb.py     image corruptions for robustness testing
│   ├── timing.py      latency benchmarking
│   ├── baseline.py    classical HOG + colour-histogram PASS/FAIL classifier
│   ├── yolo.py        adapter between Ultralytics YOLO and the package
│   └── plotting.py    shared figure style
├── notebooks/
│   └── train_colab.ipynb   trains every model on a free Colab GPU
├── scripts/
│   ├── eda.py         dataset statistics, sample images, train/test leakage check
│   ├── train.py       fine-tune YOLOv8 and evaluate on the test split
│   ├── baseline.py    train/evaluate the classical baseline
│   ├── evaluate.py    mAP cross-check, PASS/FAIL threshold, robustness
│   ├── benchmark.py   latency, FLOPs, parameters, memory
│   ├── verify_map.py  checks our AP code against Ultralytics on identical predictions
│   ├── error_analysis.py  paired bootstrap tests between models, error breakdown
│   └── compare.py     summary table and report figures
├── tests/             pytest unit tests
├── docs/              generated HTML API documentation (pdoc)
└── results/           metrics tables and figures used in the report
```

## Setup

Python 3.10+ is required.

```bash
git clone https://github.com/noahlayton/mma3001-pork-rasher-packing.git
cd mma3001-pork-rasher-packing
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Data

The dataset is *Pork Rasher Error Packaging* (version 4) from Roboflow Universe,
exported in YOLOv8 format:
<https://universe.roboflow.com/hello-2aqe0/pork-rasher-error-packaging/dataset/4>

It is **not stored in this repository** (size and licensing). Download the
YOLOv8 export and unzip it to `data/pork-rasher-v4/` so that
`data/pork-rasher-v4/data.yaml` exists.

## Workflow

Run from the repository root, in this order:

```bash
python scripts/eda.py                         # 1. explore the data      -> results/eda/
python scripts/baseline.py                    # 2. classical baseline    -> results/baseline/
# 3. train the YOLO models: open notebooks/train_colab.ipynb in Google Colab (GPU),
#    or on a local GPU:  python scripts/train.py --model yolov8n.pt --name y8n_640
python scripts/evaluate.py runs/*/weights/best.pt --robustness     # 4. -> results/eval/
python scripts/benchmark.py runs/*/weights/best.pt --device cpu --out results/benchmark_cpu.csv  # 5.
python scripts/error_analysis.py              # 6. statistical comparison -> results/compare/
python scripts/compare.py                     # 7. summary table + figures -> results/compare/
```

Trained weights are not stored in git (≈70 MB). Re-create them with the Colab
notebook; the per-epoch training logs are kept in `results/training/`.

All scripts take `--help`. Random seeds are fixed, so results are repeatable.

## Testing

```bash
pytest                                              # run all unit tests
pytest --html=results/test_report.html --self-contained-html   # HTML test report
```

The latest test report is committed at `results/test_report.html`.

## Documentation

HTML API documentation is generated from the docstrings with pdoc:

```bash
pdoc src/porkqc -o docs --docformat google
```

Open `docs/index.html` in a browser.

## Licence

MIT — see [LICENSE](LICENSE). The dataset is subject to its own Roboflow Universe licence.

## AI use

AI tools (Claude, by Anthropic) were used during this project. Details of what
they were used for and how the output was checked are given in the AI-use
reflection section of the report.
