"""Evaluation metrics for defect detection and tray-level inspection.

Two levels of evaluation are provided:

* **Object level** – intersection-over-union (IoU), greedy matching of
  predictions to ground truth, and Average Precision (AP) computed with
  all-point interpolation of the precision-recall curve. COCO and Ultralytics
  sample the same monotone precision envelope at 101 recall points instead of
  integrating it exactly, so results agree to within ~0.005. This independent
  implementation is used to cross-check the mAP reported by Ultralytics.
* **Tray level** – the binary PASS/FAIL decision that matters on a production
  line, summarised by a confusion matrix, precision, recall, F1 and accuracy.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from porkqc.labels import Box


def iou(a: Box, b: Box) -> float:
    """Intersection-over-union of two boxes.

    Args:
        a: First box.
        b: Second box.

    Returns:
        IoU in [0, 1]. Returns 0 when the union is empty (two degenerate boxes).
    """
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def match_detections(
    preds: Sequence[Box], gts: Sequence[Box], iou_thr: float = 0.5
) -> list[bool]:
    """Label each prediction as a true positive (TP) or false positive (FP).

    Predictions are processed in descending confidence order. Each one is
    matched to the unmatched ground-truth box of the same class with the
    highest IoU; it is a TP if that IoU is at least ``iou_thr``. Each ground
    truth can be matched at most once, so duplicate detections count as FPs.

    Args:
        preds: Predicted boxes for one image.
        gts: Ground-truth boxes for the same image.
        iou_thr: Minimum IoU for a match.

    Returns:
        List of booleans aligned with ``preds`` (in the original order),
        ``True`` where the prediction is a TP.
    """
    order = sorted(range(len(preds)), key=lambda i: -preds[i].conf)
    used = [False] * len(gts)
    is_tp = [False] * len(preds)
    for i in order:
        p = preds[i]
        best_j, best_iou = -1, iou_thr
        for j, g in enumerate(gts):
            if used[j] or g.cls != p.cls:
                continue
            o = iou(p, g)
            if o >= best_iou:
                best_j, best_iou = j, o
        if best_j >= 0:
            used[best_j] = True
            is_tp[i] = True
    return is_tp


def average_precision(
    confs: Sequence[float], is_tp: Sequence[bool], n_gt: int
) -> float:
    """Area under the precision-recall curve using all-point interpolation.

    The precision envelope is made monotonically decreasing (each precision
    is replaced by the maximum precision at any higher recall) and the area
    is then integrated exactly as a sum of rectangles at each recall step.

    Args:
        confs: Confidence of every prediction of one class over the dataset.
        is_tp: Whether each prediction is a TP (same order as ``confs``).
        n_gt: Total number of ground-truth boxes of that class.

    Returns:
        AP in [0, 1]. Returns ``nan`` if there are no ground-truth boxes
        (AP is undefined), and 0 if there are ground truths but no predictions.
    """
    if n_gt == 0:
        return float("nan")
    if len(confs) == 0:
        return 0.0
    order = np.argsort(-np.asarray(confs, dtype=float), kind="stable")
    tp = np.asarray(is_tp, dtype=float)[order]
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(1.0 - tp)
    recall = tp_cum / n_gt
    precision = tp_cum / (tp_cum + fp_cum)
    # Sentinel points so the curve starts at recall 0 and ends at recall 1.
    r = np.concatenate(([0.0], recall, [1.0]))
    p = np.concatenate(([1.0], precision, [0.0]))
    p = np.maximum.accumulate(p[::-1])[::-1]
    steps = np.where(r[1:] != r[:-1])[0]
    return float(np.sum((r[steps + 1] - r[steps]) * p[steps + 1]))


def evaluate_map(
    preds_per_image: Sequence[Sequence[Box]],
    gts_per_image: Sequence[Sequence[Box]],
    n_classes: int,
    iou_thr: float = 0.5,
) -> dict:
    """Compute per-class AP and mean AP over a dataset.

    Args:
        preds_per_image: Predictions for each image.
        gts_per_image: Ground truth for each image (same order).
        n_classes: Number of classes in the dataset.
        iou_thr: IoU threshold for a true positive (0.5 gives mAP@50).

    Returns:
        Dictionary with ``ap`` (array of per-class AP, ``nan`` for classes
        without ground truth) and ``map`` (mean over classes with ground truth).

    Raises:
        ValueError: If the two lists have different lengths.
    """
    if len(preds_per_image) != len(gts_per_image):
        raise ValueError("predictions and ground truth must cover the same images")
    confs: list[list[float]] = [[] for _ in range(n_classes)]
    tps: list[list[bool]] = [[] for _ in range(n_classes)]
    n_gt = np.zeros(n_classes, dtype=int)
    for preds, gts in zip(preds_per_image, gts_per_image):
        for g in gts:
            n_gt[g.cls] += 1
        for p, t in zip(preds, match_detections(preds, gts, iou_thr)):
            confs[p.cls].append(p.conf)
            tps[p.cls].append(t)
    ap = np.array(
        [average_precision(confs[c], tps[c], int(n_gt[c])) for c in range(n_classes)]
    )
    return {"ap": ap, "map": float(np.nanmean(ap)) if np.any(n_gt) else float("nan")}


def bootstrap_ci(
    y_true: Sequence[bool],
    y_pred: Sequence[bool],
    metric: str = "recall",
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile-bootstrap confidence interval for a tray-level metric.

    The test set is small (80 images), so a single score hides a lot of
    uncertainty. Images are resampled with replacement ``n_boot`` times, the
    metric is recomputed on each resample, and the ``alpha/2`` and
    ``1 - alpha/2`` percentiles of those values form the interval.

    Args:
        y_true: ``True`` where the tray is truly defective.
        y_pred: ``True`` where the tray was rejected.
        metric: Any key returned by :func:`binary_metrics` (e.g. ``"recall"``).
        n_boot: Number of bootstrap resamples.
        alpha: 1 - confidence level (0.05 gives a 95% interval).
        seed: Random seed so the interval is reproducible.

    Returns:
        ``(point_estimate, lower, upper)``.
    """
    yt = np.asarray(y_true, dtype=bool)
    yp = np.asarray(y_pred, dtype=bool)
    point = binary_metrics(yt, yp)[metric]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, yt.size, size=(n_boot, yt.size))
    stats = [binary_metrics(yt[i], yp[i])[metric] for i in idx]
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)


def binary_metrics(y_true: Sequence[bool], y_pred: Sequence[bool]) -> dict:
    """Confusion matrix and summary scores for the tray PASS/FAIL decision.

    The *positive* class is FAIL (a defective tray), because missing a
    defective tray is the costly error on a production line.

    Args:
        y_true: ``True`` where the tray is truly defective.
        y_pred: ``True`` where the tray was rejected by the inspector.

    Returns:
        Dictionary with ``tp``, ``fp``, ``fn``, ``tn``, ``precision``,
        ``recall`` (defect catch rate), ``f1``, ``accuracy`` and
        ``false_reject_rate`` (good trays wrongly rejected).

    Raises:
        ValueError: If the inputs have different lengths or are empty.
    """
    yt = np.asarray(y_true, dtype=bool)
    yp = np.asarray(y_pred, dtype=bool)
    if yt.shape != yp.shape or yt.size == 0:
        raise ValueError("y_true and y_pred must be non-empty and the same length")
    tp = int(np.sum(yt & yp))
    fp = int(np.sum(~yt & yp))
    fn = int(np.sum(yt & ~yp))
    tn = int(np.sum(~yt & ~yp))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / yt.size,
        "false_reject_rate": fp / (fp + tn) if fp + tn else 0.0,
    }
