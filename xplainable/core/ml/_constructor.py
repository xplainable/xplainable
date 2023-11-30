""" Copyright Xplainable Pty Ltd, 2023"""

import numpy
import numpy as np
from numba import njit, prange
from scipy.interpolate import CubicSpline
from ._constructor_parameters import ConstructorParams
from ...utils.encoders import NpEncoder
import json


class XConstructor:
    
    def __init__(
        self,
        regressor: bool,
        parameters: ConstructorParams
    ):
        self.type = ""
        self.regressor = regressor
        self.parameters = parameters
        self.abs_min_leaf_size = 1

        # input data related
        self.fitted_samples = 0

        self.base_value = 0
        self.base_partition = None
        self.base_meta = None

        self.null_meta = [0, 0]

        self._nodes = None

    @property
    def params(self):
        return self.parameters

    @params.setter
    def params(self, parameters: ConstructorParams):
        self.parameters = parameters

    @property
    def min_raw_score(self):
        """Retruns the Least raw score across all bin, including nan in applicable"""
        return min([a[-1] for a in self._nodes[:-1]]) if self.params.ignore_nan else min([a[-1] for a in self._nodes])

    @property
    def max_raw_score(self):
        """Retruns the Most raw score across all bin, including nan in applicable"""
        return max([a[-1] for a in self._nodes[:-1]]) if self.params.ignore_nan else max([a[-1] for a in self._nodes])

    def set_parameters(
        self,
        parameters: ConstructorParams,
        **kwargs
    ):
        self.parameters = parameters
        self.construct()

    def _get_base_partition(self, X: np.array, alpha=0.1):
        """ Gets all the categories for the feature """
        # Sort unique categories ascending
        cats = np.unique(X)
        cats = numpy.delete(cats, (numpy.where(np.isnan(cats))))  # Remove nan
        return cats

    def _activation(self, v):
        """ Activation function for frequency weighting """

        _w, _pd, _sig = self.params.weight, self.params.power_degree, self.params.sigmoid_exponent

        _nval = (v**_w) / (10**(_w*2))

        _dval = (((_nval*100 - 50) ** _pd) + (50 ** _pd)) / (2 * (50 ** _pd))

        if _sig < 1:
            return _dval
        else:
            return 1 / (1 + np.exp(-((_dval-0.5) * (10 ** _sig))))
        
    def normalise_scores(self, min, max, base_value, min_seen=0, max_seen=1):
        """ Normalise the scores to fit between 0 - 1 relative to base value.
        Args:
            score (float): The score to normalise.
        Returns:
            float: The normalised score.
        """
        if self.regressor:
            spline = CubicSpline([min, 0, max], [min_seen-base_value, 0, max_seen-base_value])
            normal_bins, nan_bin = self._nodes[:-1], self._nodes[-1]
            for i, node in enumerate(normal_bins):
                self._nodes[i][-4] = spline(node[-1])
            self._nodes[-1][-4] = 0 if self.params.ignore_nan else spline(nan_bin[-1])

            return

        def clf_normalise(score):
            # Return 0 scores as float
            if score == 0:
                return 0
            # Negative scores normalise relative to worst case scenario
            elif score < 0:
                return abs(score) / min * self.base_value
            # Positive scores normalise relative to best case scenario
            else:
                return score / max * (1 - self.base_value)
            
        normal_bins, nan_bin = self._nodes[:-1], self._nodes[-1]

        for i, node in enumerate(normal_bins):
            self._nodes[i][-4] = clf_normalise(node[-1])

        self._nodes[-1][-4] = 0 if self.params.ignore_nan else clf_normalise(
            nan_bin[-1])
        
        return

    def _construct(self) -> np.ndarray:
        """ Constructs nodes for score binning """
        return np.array([])

    @staticmethod
    @njit(parallel=True, fastmath=True, nogil=True)
    def _get_base_meta(base_partition, X, y):
        return np.empty((len(base_partition), 2), dtype=np.float64)

    def fit(self, X, y, alpha=0.1):
        """ Fits feature data to target """

        self.fitted_samples = X.size

        self.base_value = np.mean(y)
        self.base_partition = self._get_base_partition(X, alpha)
        self.base_meta = self._get_base_meta(self.base_partition, X, y)
        self.null_meta = self._get_null_meta(X, y)

        return self

    def _get_null_meta(self, X, y):
        """ Instantiates metadata for nan values"""
        x_nans = np.isnan(X)
        cnt = np.sum(x_nans)
        pos = np.sum(y*x_nans)
        mean = 0 if cnt == 0 else pos/cnt
        return [cnt, mean]

    def construct(self):
        self._nodes = self._construct()
        return self
    
    def to_json(self, default_parameters: ConstructorParams = None):
        return {
            "type": self.type,
            "regressor": self.regressor,
            "fitted_samples": self.fitted_samples,
            "base_value": self.base_value,
            "base_partition": json.loads(json.dumps(self.base_partition, cls=NpEncoder)),
            "base_meta": json.loads(json.dumps(self.base_meta, cls=NpEncoder)),
            "null_meta": json.loads(json.dumps(self.null_meta, cls=NpEncoder)),
            'params': None if self.params == default_parameters else self.params.to_json()
        }

    @staticmethod
    def from_json(data, default_parameters: ConstructorParams):
        type = data["type"]
        if type == "categorical":
            constructor = XCatConstructor(
                data["regressor"],
                ConstructorParams.from_json(data["params"]) if data["params"] is not None
                else default_parameters
            )
        elif type == "numeric":
            constructor = XNumConstructor(
                data["regressor"],
                ConstructorParams.from_json(data["params"]) if data["params"] is not None
                else default_parameters
            )
        else:
            raise ValueError(f"Unknown constructor type: {type}")

        constructor.fitted_samples = data["fitted_samples"]
        constructor.base_value = data["base_value"]
        constructor.base_partition = data["base_partition"]
        constructor.base_meta = data["base_meta"]
        constructor.null_meta = data["null_meta"]


class XCatConstructor(XConstructor):

    def __init__(self, regressor: bool, parameters: ConstructorParams):
        super().__init__(regressor, parameters)
        self.type = "categorical"

    @staticmethod
    def _get_base_meta(base_partition, X, y):
        """ Instantiates metadata at each category"""
        new_X = np.repeat(X[np.newaxis, :], len(base_partition), 0)
        cat_mask = np.where(new_X == base_partition[:, np.newaxis], 1, 0)

        pos = np.sum(cat_mask*y, axis=1)
        cnt = np.sum(cat_mask, axis=1)
        mean = pos/cnt

        meta = np.transpose(
            np.array([
                cnt,
                mean
            ])
        )
        return meta

    def _construct(self):
        """ Constructs nodes for score binning """

        _nodes = []

        for i in range(len(self.base_meta)):
            _count, _mean = self.base_meta[i]

            _freq = _count/self.fitted_samples

            diff = _mean - self.base_value
            if self.regressor:
                score = (abs(diff) ** self.params.tail_sensitivity) * np.sign(diff)
            else:
                score = self._activation(_freq*100) * diff

            _nodes.append(
                [
                    i,
                    score,
                    _mean,
                    _freq,
                    score
                ]
            )

        _count, _mean = self.null_meta
        _freq = _count/self.fitted_samples

        diff = _mean - self.base_value
        if _count == 0:
            score = 0
        else:
            if self.regressor:
                score = (abs(diff) ** self.params.tail_sensitivity) * np.sign(diff)
            else:
                score = self._activation(_freq*100) * diff

        _nodes.append(
            [
                np.nan,
                score,
                _mean,
                _freq,
                score
            ]
        )

        return np.array(_nodes)


class XNumConstructor(XConstructor):

    def __init__(self, regressor: bool, parameters: ConstructorParams):
        super().__init__(regressor, parameters)
        self.type = "numeric"

    def _get_base_partition(self, X: np.array, alpha=0.1):
        """ Calculates possible splits for feature """

        # Sort unique categories ascending
        unq = super()._get_base_partition(X, alpha)

        nunq = unq.size

        # Reduce number of bins with alpha value
        num_bins = int((nunq ** (1 - alpha) - 1) / (1 - alpha)) + 1

        # Calculate bin indices
        psplits = (unq[:-1] + unq[1:]) / 2

        # Get possible splits
        psplits = psplits[:: int(nunq / num_bins)]

        return psplits

    @staticmethod
    @njit(parallel=True, fastmath=True, nogil=True)
    def _get_base_meta(base_partition, X, y):

        """ Instantiates metadata at each split """

        _meta = np.empty((len(base_partition), 2, 2), dtype=np.float64)

        _len_y = y.size
        _n_splits = base_partition.size

        for i in prange(_n_splits):

            _split = base_partition[i]

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

            _meta[i, 0, 0] = _0_cnt
            _meta[i, 0, 1] = _0_mean
            _meta[i, 1, 0] = _1_cnt
            _meta[i, 1, 1] = _1_mean

        return _meta

        """ Instantiates metadata at each split 
        _meta = np.empty((len(base_partition), 2, 2), dtype=np.float64)
        _len_y = y.size

        for i in prange(len(base_partition)):
            _split = base_partition[i]
            new_X = np.repeat(X[np.newaxis, :], 2, 0)
            div_mask = np.where(new_X < _split, 1, 0) * np.array([[1], [-1]]) \
                + np.array([[0], [1]])
            
            pos = np.sum(div_mask * y, axis=1)
            cnt = np.sum(div_mask, axis=1)
            mean = pos / cnt
            _meta[i] = np.transpose(
                np.array([
                    cnt,
                    mean
                ])
            )

        return _meta"""

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

    def _construct(self):
        """ Constructs nodes for score binning """

        stack = [(
            self.base_partition,
            self.base_meta,
            0,
            self.base_value,
            self.fitted_samples,
            np.array([]),
            np.array([])
            )]
        
        _nodes = []

        while stack:

            # First parent split (_) is ignored
            _splits, _meta, _depth, _mean, _count, _dir, _path = stack.pop()

            idx = self._best_split(
                _meta,
                self.abs_min_leaf_size,
                self.base_value,
                self.fitted_samples,
                self.params.min_info_gain
            )

            if (idx == -1) or (_depth >= self.params.max_depth):

                diff = _mean - self.base_value

                _freq = _count / self.fitted_samples

                if self.regressor:
                    score = (abs(diff) ** self.params.tail_sensitivity) * np.sign(diff)
                else:
                    score = self._activation(_freq*100) * diff

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

            _0_cnt = _s_meta[0][0]
            _1_cnt = _s_meta[1][0]

            _0_node = [
                _0_psplits,
                _0_meta,
                _depth+1,
                _s_meta[0][1],
                _0_cnt,
                _0_dir,
                _path
            ]

            _1_node = [
                _1_psplits,
                _1_meta,
                _depth+1,
                _s_meta[1][1],
                _1_cnt,
                _1_dir,
                _path
            ]

            stack.append(_1_node)
            stack.append(_0_node)

        _count, _mean = self.null_meta
        _freq = _count / self.fitted_samples

        if _count == 0:
            score = 0
        else:
            diff = _mean - self.base_value
            if self.regressor:
                score = (abs(diff) ** self.params.tail_sensitivity) * np.sign(diff)
            else:
                score = self._activation(_freq * 100) * diff

        _nodes.append(
            [
                np.nan,
                np.nan,
                score,
                _mean,
                _freq,
                score
            ]
        )

        return np.array(_nodes)

    def fit(self, X, y, alpha=0.1):
        """ Fits feature data to target """
        super().fit(X, y, alpha)
        self.abs_min_leaf_size = np.max(
            [1, int(self.params.min_leaf_size * self.fitted_samples)])
        
        return self
