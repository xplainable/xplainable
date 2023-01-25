include "ctree.pyx"
import numpy as np
cimport numpy as np
cimport cython
from cpython cimport array
from numpy cimport ndarray
from libc.math cimport fabs

from cython.parallel import prange
from cython cimport dict

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class XRProfiler:
    # params
    cdef int max_depth
    cdef double min_info_gain
    cdef double min_leaf_size
    cdef double alpha
    cdef double tail_sensitivity

    # Storage
    cdef double base_value
    cdef list profile

    def __init__(
        self,
        int max_depth=8,
        double min_info_gain=0.02,
        double min_leaf_size=0.02,
        double tail_sensitivity=1,
        double alpha=0.01):

        self.max_depth = max_depth
        self.min_info_gain = min_info_gain
        self.min_leaf_size = min_leaf_size
        self.alpha = alpha
        self.tail_sensitivity = tail_sensitivity

    @property
    def profile(self):
        return self.profile
    
    @property
    def base_value(self):
        return self.base_value

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef fit(self, double[:,:] x, double[:] y):

        cdef double[:] f
        cdef Py_ssize_t i, n
        cdef Tree tree
        self.base_value = np.mean(y)
        self.profile = []

        for i in range(x.shape[1]):
            f = x[:, i]
            tree = Tree(
                True, # regressor
                self.max_depth,
                self.min_leaf_size,
                self.min_info_gain,
                self.alpha,
                self.tail_sensitivity
                )

            tree.fit(f, y)
            for n in range(len(tree.leaf_nodes)):
                tree.leaf_nodes[n][2] = tree.leaf_nodes[n][2] / x.shape[1]

            self.profile.append(tree.leaf_nodes)
        
        return self
