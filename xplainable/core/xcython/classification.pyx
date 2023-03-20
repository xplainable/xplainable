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
cdef class XCProfiler:
    # params
    cdef int max_depth
    cdef double min_info_gain
    cdef double min_leaf_size
    cdef double alpha
    cdef double weight
    cdef long power_degree
    cdef double sigmoid_exponent

    # Storage
    cdef double base_value
    cdef list profile
    cdef list trees

    def __init__(
        self,
        int max_depth=8,
        double min_info_gain=0.02,
        double min_leaf_size=0.02,
        double alpha=0.01,
        double weight=1,
        long power_degree=1,
        double sigmoid_exponent=0
        ):

        self.max_depth = max_depth
        self.min_info_gain = min_info_gain
        self.min_leaf_size = min_leaf_size
        self.alpha = alpha
        self.weight = weight
        self.power_degree = power_degree
        self.sigmoid_exponent = sigmoid_exponent

    @property
    def profile(self):
        return self.profile
    
    @property
    def base_value(self):
        return self.base_value

    @property
    def trees(self):
        return self.trees

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.inline
    @cython.cdivision(True)
    cdef _normalise_score(self, double score, double _sum_min, double _sum_max):
        """ Normalise the scores to fit between 0 - 1 relative to base value.

        Args:
            score (float): The score to normalise.

        Returns:
            float: The normalised score.
        """

        # Return 0 scores as float
        if score == 0.0:
            return 0.0

        # Negative scores normalise relative to worst case scenario
        elif score < 0:
            return fabs(score) / _sum_min * self.base_value

        # Positive scores normalise relative to best case scenario
        else:
            return score / _sum_max * (1 - self.base_value)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef _build_profile(self):

        self.profile = []
        cdef double[:] _min_scores = np.empty(0)
        cdef double[:] _max_scores = np.empty(0)
        cdef Tree tree
        cdef double _sum_min, _sum_max
        cdef Py_ssize_t i, idx
        cdef list v, node

        for i in range(len(self.trees)):
            tree = self.trees[i]
            _max_scores = np.append(_max_scores, tree.max_score)
            _min_scores = np.append(_min_scores, tree.min_score)

            # don't update the original leaf nodes
            self.profile.append(
                [np.array(arr) for arr in tree.leaf_nodes])

        _sum_min = np.sum(_min_scores)
        _sum_max = np.sum(_max_scores)

        for idx in range(len(self.profile)):
            v = self.profile[idx]
            for i in range(len(v)):
                node = list(v[i])
                self.profile[idx][i][2] = self._normalise_score(
                    node[5], _sum_min, _sum_max)

        return self

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef fit(self, double[:,:] x, double[:] y):

        cdef double[:] f
        cdef Py_ssize_t i
        cdef Tree tree
        self.base_value = np.mean(y)
        self.trees = []

        for i in range(x.shape[1]):
            f = x[:, i]
            tree = Tree(
                regressor=False, # classifier
                max_depth=self.max_depth,
                min_info_gain=self.min_info_gain,
                min_leaf_size=self.min_leaf_size,
                alpha=self.alpha,
                tail_sensitivity=1, # No need for tail sensitivity
                weight=self.weight,
                power_degree=self.power_degree,
                sigmoid_exponent=self.sigmoid_exponent,
                )

            tree.fit(f, y)
            self.trees.append(tree)

        self._build_profile()
        
        return self

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef rebuild_tree(
        self, Py_ssize_t idx, int max_depth, double min_info_gain,
        double min_leaf_size, double alpha, double weight, long power_degree,
        double sigmoid_exponent):

        self.trees[idx].rebuild_tree(
            max_depth = max_depth,
            min_info_gain = min_info_gain,
            min_leaf_size = min_leaf_size,
            alpha = alpha,
            weight = weight,
            power_degree = power_degree,
            sigmoid_exponent = sigmoid_exponent
        )

        # rebuild the profile with new values
        self._build_profile()

        return self
