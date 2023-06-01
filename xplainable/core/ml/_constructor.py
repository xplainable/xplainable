""" Copyright Xplainable Pty Ltd, 2023"""

import numpy as np
from numba import njit, prange


class XConstructor:
    
    def __init__(
        self,
        max_depth=8,
        min_info_gain=0.0001,
        min_leaf_size=0.0001,
        alpha=0.01,
        tail_sensitivity=1.0,
        weight=1,
        power_degree=1,
        sigmoid_exponent=0,
        regressor=False,
        *args,
        **kwargs
    ):
    
        self.regressor = regressor
        self.max_depth = max_depth
        self.min_info_gain = min_info_gain
        self.min_leaf_size = min_leaf_size
        self.tail_sensitivity = tail_sensitivity
        self.alpha = alpha
        self.weight = weight
        self.power_degree = power_degree
        self.sigmoid_exponent = sigmoid_exponent
        
        self._nodes = None
        self._max_score = -np.inf
        self._min_score = np.inf
        self.__root = None
        
    def _psplits(self, X: np.array):
        """ Calculates possible splits for feature """

        # Sort unique categories ascending
        unq = np.unique(X)
        unq = np.sort(unq)

        nunq = unq.size

        # Reduce number of bins with alpha value
        bins = int((nunq ** (1 - self.alpha) - 1) / (1 - self.alpha)) + 1

        # Calculate bin indices
        psplits = (unq[:-1] + unq[1:]) / 2

        # Get possible splits
        psplits = psplits[:: int(nunq / bins)]

        return psplits
    
    def _activation(self, v):
        """ Activation function for frequency weighting """
        
        _w, _pd, _sig = self.weight, self.power_degree, self.sigmoid_exponent

        _nval = (v**_w) / (10**(_w*2))

        _dval = (((_nval*100 - 50) ** _pd) + (50 ** _pd)) / (2 * (50 ** _pd))

        if _sig < 1:
            return _dval
            
        else:
            return 1 / (1 + np.exp(-((_dval-0.5) * (10 ** _sig))))
    
    @staticmethod
    @njit(parallel=True, fastmath=True, nogil=True)
    def _init_clf(splits, X, y):
        """ Instantiates metadata at each split """

        _meta = np.empty((len(splits), 2, 2), dtype=np.float64)

        _len_y = y.size
        _n_splits = splits.size

        for i in prange(_n_splits):

            _split = splits[i]

            _0_cnt = 0
            _0_pos = 0
            _1_cnt = 0
            _1_pos = 0

            # Create splits
            for v in prange(_len_y):
                if X[v] <= _split:
                    _0_cnt += 1
                    if y[v] == 1:
                        _0_pos += 1

                else:
                    _1_cnt += 1
                    if y[v] == 1:
                        _1_pos += 1

            _0_mean = _0_pos / _0_cnt
            _1_mean = _1_pos / _1_cnt

            _meta[i,0,0] = _0_cnt
            _meta[i,0,1] = _0_mean
            _meta[i,1,0] = _1_cnt
            _meta[i,1,1] = _1_mean

        return _meta

    @staticmethod
    @njit(parallel=True, fastmath=True, nogil=True)
    def _init_reg(splits, X, y):
        """ Instantiates metadata at each split """

        _meta = np.empty((len(splits), 2, 2), dtype=np.float64)

        _len_y = y.size
        _n_splits = splits.size

        for i in prange(_n_splits):

            _split = splits[i]

            _0_cnt = 0
            _0_tot = 0
            _1_cnt = 0
            _1_tot = 0

            # Create splits
            for v in prange(_len_y):
                if X[v] <= _split:
                    _0_cnt += 1
                    _0_tot += y[v]

                else:
                    _1_cnt += 1
                    _1_tot += y[v]

            _0_mean = _0_tot / _0_cnt
            _1_mean = _1_tot / _1_cnt

            _meta[i,0,0] = _0_cnt
            _meta[i,0,1] = _0_mean
            _meta[i,1,0] = _1_cnt
            _meta[i,1,1] = _1_mean

        return _meta
    
    @staticmethod
    @njit(parallel=False, fastmath=True, nogil=True)
    def _best_split(meta, mls, bv, samp, mig):
        """ Finds the best split across all splits """

        bst = 0
        _idx = -1

        for i in range(len(meta)):
            l = meta[i][0]
            r = meta[i][1]
            if (l[0] < mls) or (r[0] < mls):
                continue

            ld = abs(l[1] - bv)
            rd = abs(r[1] - bv)

            md = max([ld, rd])

            if md < mig:
                continue

            s = (ld * np.log2(l[0] / samp * 100)) + (
                rd * np.log2(r[0] / samp * 100))

            if s > bst:
                bst = s
                _idx = i

        return _idx
    
    def _construct(self, root):
        """ Constructs nodes for score binning """
        
        stack = [root]
        _nodes = []

        while stack:

            if len(stack) == 0:
                break
            
            # First parent split (_) is ignored
            _meta, _splits, _, _depth, _mean, _freq, _dir, _path = stack.pop()
            
            idx = self._best_split(
                _meta,
                self.abs_min_leaf_size,
                self.base_value,
                self.samples,
                self.min_info_gain
            )
            
            if (idx == -1) or (_depth >= self.max_depth):
                
                if not self.regressor:
                    score = self._activation(_freq*100) * (
                        _mean - self.base_value)

                else:
                    diff = _mean - self.base_value

                    if diff >= 0:
                        score = diff ** self.tail_sensitivity
                        
                    else:
                        score = (abs(diff) ** self.tail_sensitivity) * - 1

                if score < self._min_score:
                        self._min_score = score

                if score > self._max_score:
                    self._max_score = score

                _upper = np.inf
                _lower = -np.inf

                for i in range(len(_path)):
                    _split_path = _path[i]
                    _direction = _dir[i]
                    if _direction == 0:
                        _upper = _split_path

                    elif _direction == 1:
                        _lower = _split_path

                # score at end to persist non-normalised score
                _nodes.append([_lower, _upper, score, _mean, _freq, score])

                continue
            
            # 0=l, 1=r
            _split = _splits[idx]
            _s_meta = _meta[idx]
            _0_n = _s_meta[0][0]
            _1_n = _s_meta[1][0]
            _0_meta = np.array(_meta[:idx])
            _1_meta = np.array(_meta[idx+1:])

            for i in range(_0_meta.shape[0]):
                _0_meta[i, 1, 1] = (_0_meta[i, 1, 0] * _0_meta[i, 1, 1]) \
                - (_s_meta[1, 0] * _s_meta[1, 1])

                _0_meta[i, 1, 0] = _0_meta[i, 1, 0] - _1_n
                _0_meta[i, 1, 1] = _0_meta[i, 1, 1] / _0_meta[i, 1, 0]

            for i in range(_1_meta.shape[0]):
                _1_meta[i, 0, 1] = (_1_meta[i, 0, 0] * _1_meta[i, 0, 1]) \
                - (_s_meta[0, 0] * _s_meta[0, 1])

                _1_meta[i, 0, 0] = _1_meta[i, 0, 0] - _0_n
                _1_meta[i, 0, 1] = _1_meta[i, 0, 1] / _1_meta[i, 0, 0]

            _0_psplits = np.array(_splits[:idx])
            _1_psplits = np.array(_splits[idx+1:])

            _path = np.append(_path, _split)

            _0_dir = _1_dir = _dir
            _0_dir = np.hstack((_dir, np.array([0])))
            _1_dir = np.hstack((_dir, np.array([1])))

            _0_freq = _s_meta[0][0] / self.samples
            _1_freq = _s_meta[1][0] / self.samples

            _0_node = [
                _0_meta,
                _0_psplits,
                _split,
                _depth+1,
                _s_meta[0][1],
                _0_freq,
                _0_dir,
                _path
            ]

            _1_node = [
                _1_meta,
                _1_psplits,
                _split,
                _depth+1,
                _s_meta[1][1],
                _1_freq,
                _1_dir,
                _path
            ]

            stack.append(_1_node)
            stack.append(_0_node)

        return np.array(_nodes)

    def reconstruct(
        self,
        max_depth,
        min_info_gain,
        min_leaf_size,
        alpha,
        weight=None,
        power_degree=None,
        sigmoid_exponent=None,
        tail_sensitivity=None,
        *args,
        **kwargs
    ):
        """ Reconstructs nodes with new params without reinitiating splits """
    
        self.max_depth = max_depth
        self.min_info_gain = min_info_gain
        self.min_leaf_size = min_leaf_size
        self.alpha = alpha
        self.weight = weight
        self.power_degree = power_degree
        self.sigmoid_exponent = sigmoid_exponent
        self.tail_sensitivity = tail_sensitivity
        
        self._nodes = []
        self.max_score = -np.inf
        self.min_score = np.inf

        self.abs_min_leaf_size = np.max(
            [1, int(self.min_leaf_size * self.samples)])
        
        root = self._copy_root()
        
        self._nodes = self._construct(root)

        return self

    def _copy_root(self):
        return self.__root.copy()
    
    def fit(self, X, y):
        """ Fits feature data to target """
        
        self.base_value = np.mean(y)
        self.samples = X.size
        self.abs_min_leaf_size = np.max(
            [1, int(self.min_leaf_size * self.samples)])
        
        _psplits = self._psplits(X)

        _init = self._init_reg if self.regressor else self._init_clf
        _meta = _init(_psplits, X, y)
        
        _parent = _depth = 0
        _dir = _path = np.array([])
        
        self.__root = [
            _meta,
            _psplits,
            _parent,
            _depth,
            self.base_value,
            self.samples,
            _dir,
            _path
        ]

        root = self._copy_root()
        
        self._nodes = self._construct(root)
        
        return self
