"""Tests for porkqc.metrics, using hand-computed reference values."""

import math

import pytest

from porkqc.labels import Box
from porkqc.metrics import (
    average_precision,
    binary_metrics,
    evaluate_map,
    iou,
    match_detections,
)


def test_iou_identical_boxes_is_one():
    b = Box(0, 10, 10, 50, 50)
    assert iou(b, b) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero():
    assert iou(Box(0, 0, 0, 10, 10), Box(0, 20, 20, 30, 30)) == 0.0


def test_iou_half_overlap():
    # Two 10x10 boxes overlapping by a 5x10 strip: inter=50, union=150.
    assert iou(Box(0, 0, 0, 10, 10), Box(0, 5, 0, 15, 10)) == pytest.approx(1 / 3)


def test_iou_is_symmetric():
    a, b = Box(0, 0, 0, 12, 7), Box(0, 3, 2, 20, 9)
    assert iou(a, b) == pytest.approx(iou(b, a))


def test_iou_degenerate_boxes():
    z = Box(0, 5, 5, 5, 5)
    assert iou(z, z) == 0.0


def test_match_requires_same_class():
    gt = [Box(0, 0, 0, 10, 10)]
    pred = [Box(1, 0, 0, 10, 10, conf=0.9)]
    assert match_detections(pred, gt) == [False]


def test_duplicate_detection_is_false_positive():
    gt = [Box(0, 0, 0, 10, 10)]
    preds = [Box(0, 0, 0, 10, 10, conf=0.6), Box(0, 0, 0, 10, 10, conf=0.9)]
    # The higher-confidence prediction claims the ground truth first.
    assert match_detections(preds, gt) == [False, True]


def test_match_respects_iou_threshold():
    gt = [Box(0, 0, 0, 10, 10)]
    pred = [Box(0, 5, 0, 15, 10, conf=0.9)]  # IoU = 1/3
    assert match_detections(pred, gt, iou_thr=0.5) == [False]
    assert match_detections(pred, gt, iou_thr=0.3) == [True]


def test_average_precision_perfect():
    assert average_precision([0.9, 0.8], [True, True], n_gt=2) == pytest.approx(1.0)


def test_average_precision_hand_computed():
    # 3 ground truths, predictions ranked TP, FP, TP, TP.
    # recall    = 1/3, 1/3, 2/3, 1
    # precision = 1,   1/2, 2/3, 3/4  -> envelope 1, 3/4, 3/4, 3/4
    # AP = 1/3*1 + 1/3*3/4 + 1/3*3/4 = 5/6
    ap = average_precision([0.9, 0.8, 0.7, 0.6], [True, False, True, True], n_gt=3)
    assert ap == pytest.approx(5 / 6)


def test_average_precision_order_independent():
    # Input order must not matter because AP sorts by confidence internally.
    a = average_precision([0.6, 0.9, 0.7, 0.8], [True, True, True, False], n_gt=3)
    assert a == pytest.approx(5 / 6)


def test_average_precision_missed_objects_lower_ap():
    # Only half the ground truths are ever found -> AP capped at 0.5.
    assert average_precision([0.9], [True], n_gt=2) == pytest.approx(0.5)


def test_average_precision_edge_cases():
    assert math.isnan(average_precision([], [], n_gt=0))
    assert average_precision([], [], n_gt=3) == 0.0


def test_evaluate_map_two_classes():
    gts = [[Box(0, 0, 0, 10, 10)], [Box(1, 0, 0, 10, 10)]]
    preds = [[Box(0, 0, 0, 10, 10, conf=0.9)], []]  # class 1 missed entirely
    res = evaluate_map(preds, gts, n_classes=2)
    assert res["ap"][0] == pytest.approx(1.0)
    assert res["ap"][1] == pytest.approx(0.0)
    assert res["map"] == pytest.approx(0.5)


def test_evaluate_map_ignores_classes_without_ground_truth():
    gts = [[Box(0, 0, 0, 10, 10)]]
    preds = [[Box(0, 0, 0, 10, 10, conf=0.9)]]
    res = evaluate_map(preds, gts, n_classes=3)
    assert res["map"] == pytest.approx(1.0)


def test_evaluate_map_length_mismatch():
    with pytest.raises(ValueError):
        evaluate_map([[]], [[], []], n_classes=1)


def test_binary_metrics_counts():
    y_true = [True, True, False, False, True]
    y_pred = [True, False, True, False, True]
    m = binary_metrics(y_true, y_pred)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 1)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["accuracy"] == pytest.approx(3 / 5)
    assert m["false_reject_rate"] == pytest.approx(1 / 2)


def test_binary_metrics_no_positives_predicted():
    m = binary_metrics([True, False], [False, False])
    assert m["precision"] == 0.0 and m["f1"] == 0.0


def test_binary_metrics_rejects_bad_input():
    with pytest.raises(ValueError):
        binary_metrics([], [])
    with pytest.raises(ValueError):
        binary_metrics([True], [True, False])
