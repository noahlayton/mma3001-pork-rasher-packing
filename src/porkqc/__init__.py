"""porkqc – automated quality inspection of packaged pork rashers.

MMA3001 Numerical Methods and Machine Learning project (Monash University).

The package detects and localises packaging defects in RGB images of pork
rasher trays using a YOLOv8 object detector and converts the detections into
a tray-level PASS/FAIL decision.

Modules:
    labels: Reading YOLO annotation files and dataset statistics.
    metrics: IoU, Average Precision and tray-level PASS/FAIL metrics.
    decision: Converting detections into a PASS/FAIL inspection result.
    perturb: Image corruptions used for robustness testing.
    timing: Latency benchmarking helpers.
"""

__version__ = "0.1.0"
