"""Classical (non-deep-learning) baseline for tray PASS/FAIL classification.

This is the "credible alternative" to the YOLO detector. Each image is
described by two hand-crafted feature vectors:

* an **HSV colour histogram** – captures how much meat, tray plastic and
  conveyor belt is visible and where colours shift (e.g. loose meat);
* a **HOG descriptor** (histogram of oriented gradients) on a downscaled grey
  image – captures edge structure such as seal lines, folds and wrinkles.

The concatenated features are standardised and fed to an L2-regularised
logistic regression with class weights balanced, because defect-free trays
are the minority class. The model outputs a probability that the tray is
defective; thresholding it gives PASS/FAIL. Unlike the detector, it cannot say
*where* the defect is.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog

#: Image size used for HOG (width, height); keeps the 4:3 aspect of 720x540.
HOG_SIZE = (128, 96)


def colour_histogram(img: np.ndarray, bins: tuple[int, int, int] = (8, 4, 4)) -> np.ndarray:
    """Normalised 3-D HSV colour histogram, flattened.

    Args:
        img: BGR ``uint8`` image.
        bins: Number of bins for hue, saturation and value.

    Returns:
        1-D float32 array of length ``prod(bins)`` summing to 1.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], None, list(bins), [0, 180, 0, 256, 0, 256])
    h = h.ravel().astype(np.float32)
    return h / max(h.sum(), 1.0)


def hog_features(img: np.ndarray) -> np.ndarray:
    """HOG descriptor of the image after resizing to :data:`HOG_SIZE`.

    Args:
        img: BGR ``uint8`` image.

    Returns:
        1-D float32 HOG feature vector: 9 orientations, 8x8-pixel cells and
        2x2-cell blocks give 11 * 15 blocks * 36 = 5940 values for 128x96.
    """
    grey = cv2.cvtColor(cv2.resize(img, HOG_SIZE, interpolation=cv2.INTER_AREA),
                        cv2.COLOR_BGR2GRAY)
    return hog(grey, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
               block_norm="L2-Hys", feature_vector=True).astype(np.float32)


def extract_features(img: np.ndarray) -> np.ndarray:
    """Concatenate the colour histogram and HOG features of one image.

    Args:
        img: BGR ``uint8`` image of shape ``(H, W, 3)``.

    Returns:
        1-D float32 feature vector.

    Raises:
        ValueError: If the image is not a 3-channel ``uint8`` array.
    """
    if img is None or img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("expected a BGR uint8 image of shape (H, W, 3)")
    return np.concatenate([colour_histogram(img), hog_features(img)])


def features_for_paths(paths: Sequence[Path]) -> np.ndarray:
    """Read every image and stack its features into a ``(n_images, n_features)`` matrix."""
    return np.stack([extract_features(cv2.imread(str(p))) for p in paths])


def make_classifier(c: float = 1.0, seed: int = 0) -> Pipeline:
    """Standardise features, then class-balanced L2 logistic regression.

    Args:
        c: Inverse regularisation strength (smaller = simpler model).
        seed: Random seed for the solver.

    Returns:
        An unfitted scikit-learn pipeline.
    """
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, class_weight="balanced", max_iter=5000, random_state=seed),
    )
