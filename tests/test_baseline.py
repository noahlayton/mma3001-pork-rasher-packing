"""Tests for porkqc.baseline."""

import numpy as np
import pytest

from porkqc.baseline import colour_histogram, extract_features, hog_features, make_classifier


@pytest.fixture
def img():
    return np.random.default_rng(0).integers(0, 256, (540, 720, 3), dtype=np.uint8)


def test_colour_histogram_is_normalised(img):
    h = colour_histogram(img)
    assert h.shape == (8 * 4 * 4,)
    assert h.sum() == pytest.approx(1.0, abs=1e-5)


def test_hog_length_independent_of_input_size(img):
    assert hog_features(img).shape == hog_features(img[:300, :400]).shape == (5940,)


def test_feature_vector_is_concatenation(img):
    assert extract_features(img).shape == (128 + 5940,)


def test_extract_features_rejects_bad_input():
    with pytest.raises(ValueError):
        extract_features(np.zeros((10, 10), dtype=np.uint8))
    with pytest.raises(ValueError):
        extract_features(None)


def test_classifier_learns_separable_problem():
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal(0, 1, (50, 5)), rng.normal(3, 1, (50, 5))])
    y = np.array([0] * 50 + [1] * 50)
    clf = make_classifier().fit(x, y)
    assert (clf.predict(x) == y).mean() > 0.95
