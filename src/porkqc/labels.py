"""Reading and summarising YOLO-format annotation files.

A YOLO label file is a plain-text file with one object per line::

    <class_id> <x_center> <y_center> <width> <height>

where the four box values are *normalised* to the range [0, 1] by the image
width and height. This module converts those lines into :class:`Box` objects
in pixel coordinates and computes simple dataset statistics (class counts,
box sizes, images with no objects) used in the exploratory data analysis.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Box:
    """An axis-aligned bounding box in pixel coordinates.

    Attributes:
        cls: Integer class id (index into the dataset's ``names`` list).
        x1: Left edge in pixels.
        y1: Top edge in pixels.
        x2: Right edge in pixels.
        y2: Bottom edge in pixels.
        conf: Detection confidence in [0, 1]. Ground-truth boxes use 1.0.
    """

    cls: int
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float = 1.0

    @property
    def width(self) -> float:
        """Box width in pixels."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Box height in pixels."""
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        """Box area in square pixels (0 for degenerate boxes)."""
        return max(0.0, self.width) * max(0.0, self.height)


def yolo_to_box(line: str, img_w: int, img_h: int) -> Box:
    """Convert one line of a YOLO label file into a pixel-space :class:`Box`.

    Args:
        line: A line of the form ``"cls xc yc w h"`` with normalised values.
        img_w: Image width in pixels.
        img_h: Image height in pixels.

    Returns:
        The corresponding box in pixel coordinates.

    Raises:
        ValueError: If the line does not have exactly five fields, if any
            normalised coordinate lies outside [0, 1], or if the image size
            is not positive.

    Example:
        >>> yolo_to_box("0 0.5 0.5 0.2 0.4", 100, 50)
        Box(cls=0, x1=40.0, y1=15.0, x2=60.0, y2=35.0, conf=1.0)
    """
    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"image size must be positive, got {img_w}x{img_h}")
    parts = line.split()
    if len(parts) != 5:
        # Roboflow can also export polygons (more than 5 fields); these are not
        # supported by the box-based pipeline so we reject them explicitly.
        raise ValueError(f"expected 5 fields, got {len(parts)}: {line!r}")
    cls = int(parts[0])
    xc, yc, w, h = (float(p) for p in parts[1:])
    if not all(0.0 <= v <= 1.0 for v in (xc, yc, w, h)):
        raise ValueError(f"normalised coordinates must lie in [0, 1]: {line!r}")
    return Box(
        cls=cls,
        x1=(xc - w / 2) * img_w,
        y1=(yc - h / 2) * img_h,
        x2=(xc + w / 2) * img_w,
        y2=(yc + h / 2) * img_h,
    )


def read_label_file(path: str | Path, img_w: int, img_h: int) -> list[Box]:
    """Read every box from a YOLO label file.

    A missing file is treated as an image with no objects, which is how YOLO
    represents background-only (defect-free) images.

    Args:
        path: Path to the ``.txt`` label file.
        img_w: Width of the matching image in pixels.
        img_h: Height of the matching image in pixels.

    Returns:
        List of boxes (possibly empty).
    """
    path = Path(path)
    if not path.exists():
        return []
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    return [yolo_to_box(ln, img_w, img_h) for ln in lines]


def load_class_names(data_yaml: str | Path) -> list[str]:
    """Return the class names listed in a YOLO ``data.yaml`` file.

    Args:
        data_yaml: Path to the dataset description file exported by Roboflow.

    Returns:
        Class names ordered by class id.

    Raises:
        KeyError: If the file has no ``names`` entry.
    """
    cfg = yaml.safe_load(Path(data_yaml).read_text())
    names = cfg["names"]
    # Ultralytics accepts names either as a list or as an {id: name} mapping.
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return list(names)


def split_stats(labels_dir: str | Path, img_w: int = 720, img_h: int = 540) -> dict:
    """Summarise the annotations of one dataset split.

    Args:
        labels_dir: Directory containing the split's ``.txt`` label files.
        img_w: Image width in pixels (all Roboflow v4 images are 720 wide).
        img_h: Image height in pixels (all Roboflow v4 images are 540 high).

    Returns:
        Dictionary with keys ``n_images``, ``n_empty`` (images with no boxes),
        ``n_boxes``, ``class_counts`` (Counter of class id -> boxes),
        ``images_per_class`` (Counter of class id -> images containing it) and
        ``rel_areas`` (list of box area / image area for every box).
    """
    labels_dir = Path(labels_dir)
    class_counts: Counter = Counter()
    images_per_class: Counter = Counter()
    rel_areas: list[float] = []
    n_images = n_empty = 0
    for f in sorted(labels_dir.glob("*.txt")):
        boxes = read_label_file(f, img_w, img_h)
        n_images += 1
        if not boxes:
            n_empty += 1
        class_counts.update(b.cls for b in boxes)
        images_per_class.update({b.cls for b in boxes})
        rel_areas.extend(b.area / (img_w * img_h) for b in boxes)
    return {
        "n_images": n_images,
        "n_empty": n_empty,
        "n_boxes": sum(class_counts.values()),
        "class_counts": class_counts,
        "images_per_class": images_per_class,
        "rel_areas": rel_areas,
    }
