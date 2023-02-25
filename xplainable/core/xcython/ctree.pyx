import numpy as np
cimport numpy as np
cimport cython
from cpython cimport array
from numpy cimport ndarray
from libc.math cimport log, fabs

from cython.parallel import prange
from cython cimport dict
import copy

DTYPE = np.intc
cdef double INFINITY = 1e308

cdef inline double log2(double x):
    return log(x) / log(2)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef class Tree:
    # params
    cdef int max_depth
    cdef double min_info_gain
    cdef double min_leaf_size
    cdef double alpha
    cdef double tail_sensitivity
    cdef bint regressor

    # Storage
    cdef double base_value
    cdef int samples
    cdef list leaf_nodes
    cdef int abs_min_leaf_size
    cdef double min_score
    cdef double max_score

    # LEAF DATA
    cdef dict root_node
    cdef Py_ssize_t idx
    cdef double score
    cdef double upper
    cdef double lower

    def __init__(
        self,
        bint regressor=False,
        int max_depth=8,
        double min_info_gain=0.02,
        double min_leaf_size=0.02,
        double alpha=0.01,
        double tail_sensitivity=1.0):

        self.regressor = regressor
        self.max_depth = max_depth
        self.min_info_gain = min_info_gain
        self.min_leaf_size = min_leaf_size
        self.tail_sensitivity = tail_sensitivity
        self.alpha = alpha

    @property
    def regressor(self):
        return self.regressor
    
    @property
    def max_depth(self):
        return self.max_depth
    
    @property
    def min_info_gain(self):
        return self.min_info_gain

    @property
    def min_leaf_size(self):
        return self.min_leaf_size

    @property
    def alpha(self):
        return self.alpha

    @property
    def base_value(self):
        return self.base_value

    @property
    def abs_min_leaf_size(self):
        return self.abs_min_leaf_size

    @property
    def samples(self):
        return self.samples

    @property
    def leaf_nodes(self):
        return self.leaf_nodes

    @property
    def min_score(self):
        return self.min_score

    @property
    def max_score(self):
        return self.max_score

    @property
    def root_node(self):
        return self.root_node

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef get_possible_splits(self, double[:] x):

        # Sort unique categories ascending
        cdef np.ndarray[double, ndim=1] unq = np.unique(x)
        unq = np.sort(unq)

        cdef int nunq = unq.size

        # Reduce number of bins with alpha value
        cdef int bins = int((nunq ** (1 - self.alpha) - 1) / (1 - self.alpha)) + 1

        cdef np.ndarray[double, ndim=1] possible_splits = (unq[:-1] + unq[1:]) / 2

        possible_splits = possible_splits[:: int(nunq / bins)]

        return possible_splits


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.inline
    cpdef initialise_splits_clf(self, double[:] splits, double[:] x, double[:] y):
        cdef double[:,:,:] meta = np.empty((len(splits), 2, 2), dtype=np.float64)
        cdef double split
        cdef int left
        cdef int right
        cdef double pos_l
        cdef double pos_r
        cdef double lmean
        cdef double rmean
        cdef int i, v
        cdef int len_y = len(y)

        for i in range(len(splits)):

            split = splits[i]

            left = 0
            pos_l = 0.0
            right = 0
            pos_r = 0.0

            # Create splits
            for v in range(len_y):
                if x[v] <= split:
                    left += 1
                    if y[v] == 1:
                        pos_l += 1

                else:
                    right += 1
                    if y[v] == 1:
                        pos_r += 1

            lmean = pos_l / left
            rmean = pos_r / right
            
            meta[i,0,0] = left
            meta[i,0,1] = lmean
            meta[i,1,0] = right
            meta[i,1,1] = rmean

        return meta

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.inline
    cpdef initialise_splits_reg(self, double[:] splits, double[:] x, double[:] y):
        cdef double[:,:,:] meta = np.empty((len(splits), 2, 2), dtype=np.float64)
        cdef double split
        cdef int left
        cdef int right
        cdef double pos_l
        cdef double pos_r
        cdef double lmean
        cdef double rmean
        cdef int i, v
        cdef int len_y = len(y)

        for i in range(len(splits)):

            split = splits[i]

            left = 0
            left_total = 0.0
            right = 0
            right_total = 0.0

            # Create splits
            for v in range(len_y):
                if x[v] <= split:
                    left += 1
                    left_total += y[v]

                else:
                    right += 1
                    right_total += y[v]

            lmean = left_total / left
            rmean = right_total / right
            
            meta[i,0,0] = left
            meta[i,0,1] = lmean
            meta[i,1,0] = right
            meta[i,1,1] = rmean

        return meta

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.inline
    @cython.cdivision(True)
    cdef split_score(self, double[:] left, double[:] right):
        cdef ld = fabs(left[1] - self.base_value)
        cdef rd = fabs(right[1] - self.base_value)
        
        cdef  md = np.amax([ld, rd])

        if md < self.min_info_gain:
            return INFINITY
        cdef wd = (ld * log2(left[0] / self.samples * 100)) + (
            rd * log2(right[0] / self.samples * 100))

        return wd

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.inline
    @cython.cdivision(True)
    cdef best_split(self, double[:, :, :] meta):
        cdef double best = 0.0
        cdef Py_ssize_t best_idx = -1
        cdef int meta_length = len(meta)
        cdef Py_ssize_t i
        cdef double s
        cdef double[:] l, r

        for i in range(meta_length):
            l = meta[i][0]
            r = meta[i][1]
            if (l[0] < self.abs_min_leaf_size) or (r[0] < self.abs_min_leaf_size):
                continue
            s = self.split_score(l, r)
            if s == INFINITY:
                continue
            if s > best:
                best = s
                best_idx = i
        return best_idx, best

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef build_tree(self, dict node):

        cdef double diff
        cdef double score
        cdef double split, split_path
        cdef int direction
        cdef double n_left, n_right, path_len_l, path_len_r
        cdef double l_freq, r_freq
        cdef double[:, :] split_meta
        cdef double[:, :, :] left_meta, right_meta
        cdef double[:] psplits_left, psplits_right
        cdef double[:, :] lm, rm
        cdef double[:] _path
        cdef long[:] left_dir, right_dir
        cdef Py_ssize_t i
        cdef np.ndarray[double, ndim=1] leaf

        self.idx, self.score = self.best_split(node['meta'])

        if self.idx == -1 or node['depth'] >= self.max_depth:
            
            if not self.regressor:
                score = log2(node['freq'] * 100) * (node['mean'] - self.base_value)
            else:
                diff = node['mean'] - self.base_value

                if diff >= 0:
                    score = diff ** self.tail_sensitivity

                else:
                    score = (abs(diff) ** self.tail_sensitivity) * - 1

            if score < self.min_score:
                self.min_score = score

            if score > self.max_score:
                self.max_score = score

            self.upper = np.inf
            self.lower = -np.inf

            for i in range(len(node['split_path'])):
                split_path = node['split_path'][i]
                direction = node['directions'][i]
                if direction == 0:
                    self.upper = split_path
                
                elif direction == 1:
                    self.lower = split_path

            leaf = np.array([self.lower, self.upper, score, node['mean'], node['freq']])

            self.leaf_nodes.append(leaf)
            return
                        
        split = node['psplits'][self.idx]
        split_meta = node['meta'][self.idx]
        n_left = split_meta[0][0]
        n_right = split_meta[1][0]
        left_meta = np.array(node['meta'][:self.idx])
        right_meta = np.array(node['meta'][self.idx+1:])

        for i in prange(left_meta.shape[0], nogil=True):
            left_meta[i, 1, 1] = (left_meta[i, 1, 0] * left_meta[i, 1, 1]) - (split_meta[1, 0] * split_meta[1, 1])
            left_meta[i, 1, 0] = left_meta[i, 1, 0] - n_right
            left_meta[i, 1, 1] = left_meta[i, 1, 1] / left_meta[i, 1, 0]

        for i in prange(right_meta.shape[0], nogil=True):
            right_meta[i, 0, 1] = (right_meta[i, 0, 0] * right_meta[i, 0, 1]) - (split_meta[0, 0] * split_meta[0, 1])
            right_meta[i, 0, 0] = right_meta[i, 0, 0] - n_left
            right_meta[i, 0, 1] = right_meta[i, 0, 1] / right_meta[i, 0, 0]

        psplits_left = np.array(node['psplits'][:self.idx])
        psplits_right = np.array(node['psplits'][self.idx+1:])

        _path = np.asarray(node['split_path'])
        _path = np.append(_path, split)

        left_dir = right_dir = node['directions']
        left_dir = np.hstack((node['directions'], np.array([0])))
        right_dir = np.hstack((node['directions'], np.array([1])))

        l_freq = split_meta[0][0] / self.samples
        r_freq = split_meta[1][0] / self.samples
        
        cdef dict left_node = {
            'meta': left_meta,
            'psplits': psplits_left,
            'parent_split': split,
            'depth': node['depth']+1,
            'mean': split_meta[0][1],
            'freq': l_freq,
            'directions': left_dir,
            'split_path': _path
        }
        
        cdef dict right_node = {
            'meta': right_meta,
            'psplits': psplits_right,
            'parent_split': split,
            'depth': node['depth']+1,
            'mean': split_meta[1][1],
            'freq': r_freq,
            'directions': right_dir,
            'split_path': _path
        }
                
        self.build_tree(left_node)
        self.build_tree(right_node)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef rebuild_tree(
        self, int max_depth, double min_info_gain, double min_leaf_size, double alpha):
        
        cdef dict node = self._copy_root_node()

        # Update the params
        self.max_depth = max_depth
        self.min_info_gain = min_info_gain
        self.min_leaf_size = min_leaf_size
        self.alpha = alpha
        self.abs_min_leaf_size = np.amax(
            [1, int(self.min_leaf_size * self.samples)])

        # reset class variables
        self.leaf_nodes = []
        self.max_score = -INFINITY
        self.min_score = INFINITY

        # Rebuild the tree
        self.build_tree(node)

    cdef _copy_root_node(self):
        cdef double[:] split_path = np.array([])
        cdef np.ndarray[np.int_t, ndim=1] directions = np.empty(0, dtype=np.int64)
        
        cdef double[:,:,:] new_meta = np.array(self.root_node['meta'])
        cdef double[:] new_splits = np.array(self.root_node['psplits'])

        cdef dict new_root = {
            'meta': new_meta,
            'psplits': new_splits,
            'parent_split': None,
            'depth': 0,
            'mean': self.base_value,
            'freq': self.samples,
            'directions': directions,
            'split_path': split_path
        }

        return new_root

    cpdef copy_root_node(self):
        cdef dict new_root = self._copy_root_node()

        return new_root

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef fit(self, double[:] x, double[:] y):
        
        self.leaf_nodes = []
        self.base_value = np.mean(y)
        self.samples = y.shape[0]
        self.abs_min_leaf_size = np.amax([1, int(self.min_leaf_size * self.samples)])
        cdef double[:] split_path = np.array([])
        cdef np.ndarray[np.int_t, ndim=1] directions = np.empty(0, dtype=np.int64)
        cdef double[:] splits = self.get_possible_splits(x)
        cdef double[:, :, :] meta
        self.max_score = -INFINITY
        self.min_score = INFINITY

        if not self.regressor:
            meta = self.initialise_splits_clf(splits, x, y)
        else:
            meta = self.initialise_splits_reg(splits, x, y)

        cdef dict root_node = {
            'meta': meta,
            'psplits': splits,
            'parent_split': None,
            'depth': 0,
            'mean': self.base_value,
            'freq': self.samples,
            'directions': directions,
            'split_path': split_path
        }

        self.root_node = root_node

        cdef dict root = self._copy_root_node()

        self.build_tree(root)

        return self
