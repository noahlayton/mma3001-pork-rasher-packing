"""Tests for porkqc.yolo that do not need a trained model."""

from types import SimpleNamespace

import numpy as np
import pytest

from porkqc.decision import inspect_tray
from porkqc.labels import Box
from porkqc.yolo import result_to_boxes, tray_scores


class _T:
    """Minimal stand-in for a torch tensor (only .cpu().numpy() is used)."""

    def __init__(self, a):
        self.a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class _Boxes(SimpleNamespace):
    def __len__(self):
        return len(self.cls.a)


def test_result_to_boxes_converts_fields():
    r = SimpleNamespace(boxes=_Boxes(xyxy=_T([[1, 2, 3, 4]]), cls=_T([2.0]), conf=_T([0.7])))
    assert result_to_boxes(r) == [Box(2, 1.0, 2.0, 3.0, 4.0, conf=pytest.approx(0.7))]


def test_result_to_boxes_empty():
    assert result_to_boxes(SimpleNamespace(boxes=None)) == []


def test_tray_scores_takes_max_defect_confidence():
    preds = [[Box(0, 0, 0, 1, 1, 0.3), Box(1, 0, 0, 1, 1, 0.8)], [], [Box(2, 0, 0, 1, 1, 0.9)]]
    assert tray_scores(preds, defect_ids=[0, 1]).tolist() == [0.8, 0.0, 0.0]


def test_tray_scores_agree_with_inspect_tray():
    # Thresholding the score must give the same decision as the full rule.
    names = ["a", "b"]
    preds = [[Box(0, 0, 0, 1, 1, c)] for c in (0.1, 0.4, 0.6, 0.9)] + [[]]
    scores = tray_scores(preds, [0, 1])
    for t in (0.05, 0.4, 0.5, 0.95):
        rule = [not inspect_tray(p, names, names, conf_thr=t).passed for p in preds]
        assert (scores >= t).tolist() == rule
