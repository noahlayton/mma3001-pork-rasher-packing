"""Tests for porkqc.perturb and porkqc.timing."""

import numpy as np
import pytest

from porkqc.perturb import (
    PERTURBATIONS,
    adjust_brightness,
    gaussian_blur,
    gaussian_noise,
    jpeg_compress,
)
from porkqc.timing import benchmark


@pytest.fixture
def img():
    rng = np.random.default_rng(1)
    return rng.integers(0, 256, (54, 72, 3), dtype=np.uint8)


def test_brightness_identity_and_clipping(img):
    assert np.array_equal(adjust_brightness(img, 1.0), img)
    assert adjust_brightness(img, 10.0).max() <= 255
    assert adjust_brightness(img, 0.0).max() == 0


def test_brightness_rejects_negative(img):
    with pytest.raises(ValueError):
        adjust_brightness(img, -1)


def test_blur_zero_sigma_is_copy(img):
    out = gaussian_blur(img, 0)
    assert np.array_equal(out, img) and out is not img


def test_blur_reduces_variation(img):
    # Blurring is low-pass filtering, so neighbouring-pixel differences shrink.
    def roughness(x):
        return np.abs(np.diff(x.astype(float), axis=1)).mean()

    assert roughness(gaussian_blur(img, 2)) < roughness(img)


def test_noise_is_repeatable_with_seed(img):
    assert np.array_equal(gaussian_noise(img, 10, seed=3), gaussian_noise(img, 10, seed=3))


def test_noise_std_roughly_matches():
    flat = np.full((200, 200, 3), 128, dtype=np.uint8)
    diff = gaussian_noise(flat, 10, seed=0).astype(float) - 128
    assert diff.std() == pytest.approx(10, rel=0.05)


def test_jpeg_keeps_shape_and_validates_quality(img):
    assert jpeg_compress(img, 50).shape == img.shape
    with pytest.raises(ValueError):
        jpeg_compress(img, 0)


def test_perturbations_do_not_modify_input(img):
    original = img.copy()
    for fn, levels in PERTURBATIONS.values():
        fn(img, levels[-1])
    assert np.array_equal(img, original)


def test_perturbations_reject_grayscale():
    with pytest.raises(ValueError):
        adjust_brightness(np.zeros((10, 10), dtype=np.uint8), 1.0)


def test_benchmark_counts_calls_and_stats():
    calls = []
    res = benchmark(lambda: calls.append(1), repeats=10, warmup=3)
    assert len(calls) == 13
    assert len(res["samples"]) == 10
    assert res["min"] <= res["median"] <= res["p95"]


def test_benchmark_rejects_bad_arguments():
    with pytest.raises(ValueError):
        benchmark(lambda: None, repeats=0)
