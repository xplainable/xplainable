import pytest
import numpy as np
import pandas as pd
from xplainable.core.ml._constructor import XConstructor
from xplainable.core.ml.classification import XClassifier

# Testing XConstructor Class
def test_psplits():
    obj = XConstructor(alpha=0.7)
    X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    expected_output = np.array([1.5, 3.5, 5.5, 7.5])

    assert np.array_equal(obj._psplits(X), expected_output)


def _fit_tiny_classifier():
    rng = np.random.RandomState(0)
    x = pd.DataFrame({
        "a": rng.normal(size=60),
        "b": rng.normal(size=60),
    })
    y = pd.Series((x["a"] + rng.normal(scale=0.1, size=60) > 0).astype(int))
    return XClassifier().fit(x, y)


def test_constructs_round_trip_through_json():
    """constructs_to_json -> constructs_from_json must not raise and must
    restore the constructs. Guards the serialize/deserialize path the
    inference server uses to rehydrate a deployed model."""
    model = _fit_tiny_classifier()
    blob = model.constructs_to_json()
    assert blob  # model produced constructs

    restored = XClassifier()
    restored.constructs_from_json(blob)   # previously raised TypeError

    assert len(restored._constructs) == len(model._constructs)
