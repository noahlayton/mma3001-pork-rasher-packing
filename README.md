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
│   └── timing.py      latency benchmarking
├── notebooks/         Jupyter / Colab notebooks (EDA, training, evaluation)
├── scripts/           command-line entry points
├── tests/             pytest unit tests
├── docs/              generated HTML API documentation (pdoc)
├── results/           metrics tables and figures used in the report
└── report/            written project report
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

## Testing

```bash
pytest                                              # run all unit tests
pytest --html=results/test_report.html --self-contained-html   # HTML test report
```

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
