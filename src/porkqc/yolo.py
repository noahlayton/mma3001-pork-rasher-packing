"""Thin adapter between Ultralytics YOLO and the rest of the package.

Keeping all Ultralytics-specific code here means the metrics and decision
logic stay library-independent (and unit-testable without a model).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from porkqc.labels import Box


def result_to_boxes(result) -> list[Box]:
    """Convert one Ultralytics ``Results`` object into :class:`Box` objects.

    Args:
        result: An element of the list returned by ``YOLO.predict``.

    Returns:
        Predicted boxes in pixel coordinates of the original image.
    """
    b = result.boxes
    if b is None or len(b) == 0:
        return []
    xyxy = b.xyxy.cpu().numpy()
    cls = b.cls.cpu().numpy().astype(int)
    conf = b.conf.cpu().numpy()
    return [Box(int(c), *map(float, xy), conf=float(p)) for xy, c, p in zip(xyxy, cls, conf)]


def predict_boxes(model, images: Sequence[np.ndarray], imgsz: int, conf: float = 0.001,
                  device: str | None = None, batch: int = 16) -> list[list[Box]]:
    """Run a YOLO model on in-memory images and return boxes per image.

    A very low confidence threshold (0.001, the Ultralytics validation default)
    is used so the full precision-recall curve can be computed; higher
    thresholds are applied later by the PASS/FAIL rule.

    Args:
        model: A loaded ``ultralytics.YOLO`` model.
        images: BGR ``uint8`` images.
        imgsz: Inference image size in pixels.
        conf: Minimum confidence kept by the model.
        device: ``"cpu"``, ``"0"``, or ``None`` for automatic.
        batch: Images per forward pass.

    Returns:
        One list of boxes per input image.
    """
    out: list[list[Box]] = []
    for i in range(0, len(images), batch):
        res = model.predict(list(images[i:i + batch]), imgsz=imgsz, conf=conf, iou=0.7,
                            device=device, verbose=False, max_det=300)
        out.extend(result_to_boxes(r) for r in res)
    return out


def tray_scores(preds: Sequence[Sequence[Box]], defect_ids: Sequence[int]) -> np.ndarray:
    """Highest defect-class confidence per image (0 if no defect detected).

    A tray is rejected at threshold ``t`` exactly when its score is >= ``t``,
    so this single number per image is enough to sweep thresholds cheaply.

    Args:
        preds: Predicted boxes per image.
        defect_ids: Class ids that count as defects.

    Returns:
        Array of scores in [0, 1], one per image.
    """
    ids = set(defect_ids)
    return np.array([max((b.conf for b in p if b.cls in ids), default=0.0) for p in preds])
