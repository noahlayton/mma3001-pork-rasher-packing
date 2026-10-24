"""Turning object detections into a tray-level PASS/FAIL inspection decision.

The detector outputs boxes; a production line needs a single accept/reject
decision per tray. The rule used here is deliberately simple and auditable:
a tray FAILS if any detection of a *defect* class has confidence at or above
that class's threshold. Thresholds can differ by class, so a costly defect
(e.g. an unsealed pack) can be made more sensitive than a cosmetic one.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from porkqc.labels import Box


@dataclass
class InspectionResult:
    """Outcome of inspecting one tray.

    Attributes:
        passed: ``True`` if the tray is accepted.
        defects: Detections that caused the rejection (empty when passed).
        reasons: Human-readable reason per rejecting detection, e.g.
            ``"unsealed (0.87)"``, for the inspection report.
    """

    passed: bool
    defects: list[Box] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


def inspect_tray(
    detections: Iterable[Box],
    class_names: Sequence[str],
    defect_classes: Iterable[str],
    conf_thr: float | Mapping[str, float] = 0.25,
) -> InspectionResult:
    """Decide whether a tray passes inspection.

    Args:
        detections: Boxes predicted for the tray image.
        class_names: Dataset class names ordered by class id.
        defect_classes: Names of the classes that count as defects. Classes
            not listed (e.g. a "good pack" class) never cause a rejection.
        conf_thr: Either one confidence threshold for all defect classes, or a
            mapping from class name to threshold. Classes missing from the
            mapping default to 0.25.

    Returns:
        An :class:`InspectionResult`.

    Raises:
        ValueError: If a defect class is not in ``class_names`` or a threshold
            lies outside [0, 1].
    """
    defect_set = set(defect_classes)
    unknown = defect_set - set(class_names)
    if unknown:
        raise ValueError(f"unknown defect classes: {sorted(unknown)}")
    if isinstance(conf_thr, Mapping):
        thresholds = {name: float(conf_thr.get(name, 0.25)) for name in defect_set}
    else:
        thresholds = {name: float(conf_thr) for name in defect_set}
    if not all(0.0 <= t <= 1.0 for t in thresholds.values()):
        raise ValueError("confidence thresholds must lie in [0, 1]")

    defects, reasons = [], []
    for d in detections:
        name = class_names[d.cls]
        if name in defect_set and d.conf >= thresholds[name]:
            defects.append(d)
            reasons.append(f"{name} ({d.conf:.2f})")
    return InspectionResult(passed=not defects, defects=defects, reasons=reasons)


def tray_is_defective(
    gt_boxes: Iterable[Box], class_names: Sequence[str], defect_classes: Iterable[str]
) -> bool:
    """Ground-truth tray label: defective if any annotated box is a defect class.

    Args:
        gt_boxes: Annotated boxes for the image.
        class_names: Dataset class names ordered by class id.
        defect_classes: Names of the classes that count as defects.

    Returns:
        ``True`` if the tray should be rejected according to the annotations.
    """
    defect_set = set(defect_classes)
    return any(class_names[b.cls] in defect_set for b in gt_boxes)
