"""Image perturbations for robustness (sensitivity) testing.

Production-line cameras see lighting changes, motion blur, sensor noise and
compression. Each function here applies one controlled corruption with a
single *severity* parameter so detector performance can be plotted against
severity. All functions take and return ``uint8`` BGR images of shape
``(H, W, 3)`` (OpenCV convention) and never modify the input in place.
"""

from __future__ import annotations

import cv2
import numpy as np


def _check(img: np.ndarray) -> None:
    """Raise ``ValueError`` unless ``img`` is a uint8 ``(H, W, 3)`` array."""
    if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"expected uint8 HxWx3 image, got {img.dtype} {img.shape}")


def adjust_brightness(img: np.ndarray, factor: float) -> np.ndarray:
    """Scale pixel intensities to simulate over/under exposure.

    Args:
        img: Input image.
        factor: Multiplier; 1.0 leaves the image unchanged, <1 darkens,
            >1 brightens. Results are clipped to [0, 255].

    Returns:
        The adjusted image.

    Raises:
        ValueError: If ``factor`` is negative or the image is invalid.
    """
    _check(img)
    if factor < 0:
        raise ValueError("brightness factor must be non-negative")
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Blur with a Gaussian kernel to simulate defocus or motion.

    Args:
        img: Input image.
        sigma: Standard deviation of the kernel in pixels; 0 returns a copy.

    Returns:
        The blurred image.
    """
    _check(img)
    if sigma <= 0:
        return img.copy()
    # Kernel size of ~6 sigma (rounded up to odd) captures >99% of the weight.
    k = int(2 * np.ceil(3 * sigma) + 1)
    return cv2.GaussianBlur(img, (k, k), sigma)


def gaussian_noise(img: np.ndarray, std: float, seed: int | None = 0) -> np.ndarray:
    """Add zero-mean Gaussian noise to simulate sensor noise.

    Args:
        img: Input image.
        std: Noise standard deviation in intensity units (0-255 scale).
        seed: Random seed so experiments are repeatable; ``None`` for random.

    Returns:
        The noisy image.
    """
    _check(img)
    rng = np.random.default_rng(seed)
    noisy = img.astype(np.float32) + rng.normal(0.0, std, img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def jpeg_compress(img: np.ndarray, quality: int) -> np.ndarray:
    """Round-trip the image through JPEG to simulate compression artefacts.

    Args:
        img: Input image.
        quality: JPEG quality in [1, 100]; lower means stronger artefacts.

    Returns:
        The decoded image.

    Raises:
        ValueError: If ``quality`` is outside [1, 100].
    """
    _check(img)
    if not 1 <= quality <= 100:
        raise ValueError("JPEG quality must be in [1, 100]")
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


#: Perturbation name -> (function, list of severities tested in the report).
PERTURBATIONS = {
    "brightness": (adjust_brightness, [0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 1.8]),
    "blur": (gaussian_blur, [0, 1, 2, 3, 5]),
    "noise": (gaussian_noise, [0, 5, 10, 20, 30]),
    "jpeg": (jpeg_compress, [100, 50, 30, 15, 5]),
}
