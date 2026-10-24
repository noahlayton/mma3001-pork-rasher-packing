"""Tests for porkqc.decision."""

import pytest

from porkqc.decision import inspect_tray, tray_is_defective
from porkqc.labels import Box

NAMES = ["good", "loose", "unsealed"]
DEFECTS = ["loose", "unsealed"]


def test_tray_with_no_detections_passes():
    res = inspect_tray([], NAMES, DEFECTS)
    assert res.passed and res.defects == []


def test_non_defect_class_never_fails_tray():
    res = inspect_tray([Box(0, 0, 0, 10, 10, conf=0.99)], NAMES, DEFECTS)
    assert res.passed


def test_confident_defect_fails_tray():
    res = inspect_tray([Box(2, 0, 0, 10, 10, conf=0.8)], NAMES, DEFECTS, conf_thr=0.5)
    assert not res.passed
    assert res.reasons == ["unsealed (0.80)"]


def test_low_confidence_defect_is_ignored():
    res = inspect_tray([Box(1, 0, 0, 10, 10, conf=0.3)], NAMES, DEFECTS, conf_thr=0.5)
    assert res.passed


def test_threshold_is_inclusive():
    res = inspect_tray([Box(1, 0, 0, 10, 10, conf=0.5)], NAMES, DEFECTS, conf_thr=0.5)
    assert not res.passed


def test_per_class_thresholds():
    dets = [Box(1, 0, 0, 10, 10, conf=0.4), Box(2, 0, 0, 10, 10, conf=0.4)]
    res = inspect_tray(dets, NAMES, DEFECTS, conf_thr={"loose": 0.6, "unsealed": 0.3})
    assert not res.passed
    assert [d.cls for d in res.defects] == [2]


def test_unknown_defect_class_raises():
    with pytest.raises(ValueError):
        inspect_tray([], NAMES, ["torn"])


def test_invalid_threshold_raises():
    with pytest.raises(ValueError):
        inspect_tray([], NAMES, DEFECTS, conf_thr=1.5)


def test_ground_truth_tray_label():
    assert tray_is_defective([Box(1, 0, 0, 1, 1)], NAMES, DEFECTS)
    assert not tray_is_defective([Box(0, 0, 0, 1, 1)], NAMES, DEFECTS)
    assert not tray_is_defective([], NAMES, DEFECTS)
