import numpy as np
from numba import njit, prange

@njit(fastmath=False)
def nansum_numba_1d(arr):
    total = 0.0
    for i in range(arr.shape[0]):
        val = arr[i]
        if not np.isnan(val):
            total += float(val)
    return total

@njit(fastmath=False)
def nansum_numba_2d_axis0(arr):
    result = np.zeros(arr.shape[1], dtype=np.float64)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            if not np.isnan(val):
                result[j] += float(val)
    return result

@njit(fastmath=False, parallel=True)
def nansum_numba_2d_axis1(arr):
    result = np.zeros(arr.shape[0], dtype=np.float64)
    for i in prange(arr.shape[0]):
        for j in prange(arr.shape[1]):
            val = arr[i, j]
            if not np.isnan(val):
                result[i] += float(val)
    return result

def nansum_numba(arr, axis=None):
    if axis is None or arr.ndim == 1:
        return nansum_numba_1d(arr)
    elif axis == 0:
        return nansum_numba_2d_axis0(arr)
    elif axis == 1:
        return nansum_numba_2d_axis1(arr)
    else:
        raise ValueError("Invalid axis value")
