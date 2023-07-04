import pytest
import numpy as np
from xplainable.core.ml._constructor import XConstructor

# Testing XConstructor Class
def test_psplits():
    obj = XConstructor(alpha=0.7)
    X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    expected_output = np.array([1.5, 3.5, 5.5, 7.5])

    assert np.array_equal(obj._psplits(X), expected_output)
