""" Copyright Xplainable Pty Ltd, 2023"""
import numpy
import numpy as np
from numba import njit, prange


class XConstructor:
    
    def __init__(
        self,
        regressor,
        weight=1,
        power_degree=1,
        sigmoid_exponent=0,
        tail_sensitivity: float = 1.0
    ):  # TODO checked
        self.regressor = regressor

        # score normilise related
        self.tail_sensitivity = tail_sensitivity
        self.weight = weight
        self.power_degree = power_degree
        self.sigmoid_exponent = sigmoid_exponent

        # input data related
        self.base_value = 0
        self.samples = 0
        self.base_partition = None
        self.base_meta = None

        self._nodes = None
        self._max_score = -np.inf
        self._min_score = np.inf

    def set_parameters(
        self,
        weight=1,
        power_degree=1,
        sigmoid_exponent=0,
        tail_sensitivity=1.0,
        **kwargs
    ):
        self.weight = weight  # ClfModel
        self.power_degree = power_degree  # ClfModel
        self.sigmoid_exponent = sigmoid_exponent  # ClfModel
        self.tail_sensitivity = tail_sensitivity  # RegModel
        self._max_score = -np.inf
        self._min_score = np.inf  # TODO fix default / inhert issue

    def _get_base_partition(self, X: np.array):  # TODO checked
        """ Gets all the categories for the feature """
        # Sort unique categories ascending
        return np.sort(np.unique(X))

    def _activation(self, v):  # TODO checked
        """ Activation function for frequency weighting """

        _w, _pd, _sig = self.weight, self.power_degree, self.sigmoid_exponent

        _nval = (v**_w) / (10**(_w*2))

        _dval = (((_nval*100 - 50) ** _pd) + (50 ** _pd)) / (2 * (50 ** _pd))

        if _sig < 1:
            return _dval
        else:
            return 1 / (1 + np.exp(-((_dval-0.5) * (10 ** _sig))))

    def _construct(self) -> np.ndarray:  # TODO checked
        """ Constructs nodes for score binning """
        return np.array([])

    @staticmethod
    @njit(parallel=True, fastmath=True, nogil=True)
    def _get_base_meta(base_partition, X, y):  # TODO checked
        return np.empty((len(base_partition), 2), dtype=np.float64)

    def fit(self, X, y):  # TODO checked
        """ Fits feature data to target """

        self.base_value = np.mean(y)
        self.samples = X.size

        self.base_partition = self._get_base_partition(X)
        self.base_meta = self._get_base_meta(self.base_partition, X, y)

        return self

    def construct(self):
        self._nodes = self._construct()
        return self


class XClfConstructor(XConstructor):

    def __init__(
            self,
            regressor=False,
            weight=1,
            power_degree=1,
            sigmoid_exponent=0,
            tail_sensitivity=1.0,
            *args,
            **kwargs
    ):  # TODO checked
        super().__init__(
            regressor,
            weight,
            power_degree,
            sigmoid_exponent,
            tail_sensitivity,
        )
        print("Categorical")

    @staticmethod
    @njit(parallel=True, fastmath=True, nogil=True)
    def _get_base_meta(base_partition, X, y):  # TODO checked
        """ Instantiates metadata at each category"""
        _meta = np.empty((len(base_partition), 2), dtype=np.float64)
        _len_y = y.size

        for i in prange(len(base_partition)):

            _cat = base_partition[i]

            cat_cnt = 0  # count left
            cat_pos = 0  # positives in left

            # Create splits
            for v in prange(_len_y):
                if X[v] == _cat:
                    cat_cnt += 1
                    cat_pos += y[v]

            cat_mean = cat_pos / cat_cnt

            _meta[i] = np.array(
                [
                    [cat_cnt, cat_mean]
                ]
            )
        return _meta

    def _construct(self):  # TODO checked
        """ Constructs nodes for score binning """

        _nodes = []

        for i in range(len(self.base_meta)):
            _count, _mean = self.base_meta[i]

            _freq = _count/self.samples

            diff = _mean - self.base_value
            if self.regressor:
                score = (abs(diff) ** self.tail_sensitivity) * np.sign(diff)
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

        self._min_score = min([a[-4] for a in _nodes])
        self._max_score = max([a[-4] for a in _nodes])

        return np.array(_nodes)


class XRegConstructor(XConstructor):

    def __init__(
            self,
            regressor=False,
            max_depth=8,
            min_info_gain=0.0001,
            min_leaf_size=0.0001,
            alpha=0.01,
            tail_sensitivity=1.0,
            weight=1,
            power_degree=1,
            sigmoid_exponent=0,
            *args,
            **kwargs
    ):  # TODO checked
        self.abs_min_leaf_size = 1
        self.max_depth = max_depth
        self.min_info_gain = min_info_gain
        self.min_leaf_size = min_leaf_size
        self.alpha = alpha
        super().__init__(
            regressor,
            weight,
            power_degree,
            sigmoid_exponent,
            tail_sensitivity,
        )
        print("Numeric")

    def set_parameters(
        self,
        weight=None,
        power_degree=None,
        sigmoid_exponent=None,
        tail_sensitivity=None,
        max_depth=None,
        min_info_gain=None,
        min_leaf_size=None,
        alpha=None,
    ):
        super().set_parameters(weight, power_degree, sigmoid_exponent, tail_sensitivity)
        self.abs_min_leaf_size = np.max([1, int(self.min_leaf_size * self.samples)])
        self.max_depth = max_depth if max_depth else self.max_depth
        self.min_info_gain = min_info_gain if min_info_gain else self.min_info_gain
        self.min_leaf_size = min_leaf_size if min_leaf_size else self.min_leaf_size
        self.alpha = alpha if alpha else self.alpha

    def _get_base_partition(self, X: np.array):
        """ Calculates possible splits for feature """

        # Sort unique categories ascending
        unq = super()._get_base_partition(X)

        nunq = unq.size

        # Reduce number of bins with alpha value
        bins = int((nunq ** (1 - self.alpha) - 1) / (1 - self.alpha)) + 1

        # Calculate bin indices
        psplits = (unq[:-1] + unq[1:]) / 2

        # Get possible splits
        psplits = psplits[:: int(nunq / bins)]

        return psplits

    @staticmethod
    @njit(parallel=True, fastmath=True, nogil=True)
    def _get_base_meta(base_partition, X, y):  # TODO checked
        """ Instantiates metadata at each split """
        _meta = np.empty((len(base_partition), 2, 2), dtype=np.float64)
        _len_y = y.size

        for i in prange(len(base_partition)):

            _split = base_partition[i]
            _0_cnt = _0_pos = _1_cnt = _1_pos = 0

            # Create splits
            for v in prange(_len_y):
                if X[v] <= _split:
                    _0_cnt += 1
                    _0_pos += y[v]
                else:
                    _1_cnt += 1
                    _1_pos += y[v]

            _0_mean = _0_pos / _0_cnt
            _1_mean = _1_pos / _1_cnt

            _meta[i] = np.array(
                [
                    [_0_cnt, _0_mean],
                    [_1_cnt, _1_mean]
                ]
            )

        return _meta

    @staticmethod
    # @njit(parallel=False, fastmath=True, nogil=True)
    def _best_split(meta, mls, bv, samp, mig):
        """ Finds the best split across all splits """

        bst = 0
        _idx = -1

        for i in range(len(meta)):  # TODO why not prange? not parallel?
            l = meta[i][0]
            r = meta[i][1]
            if (l[0] < mls) or (r[0] < mls):  # TODO less than min leaf size
                continue

            ld = abs(l[1] - bv)
            rd = abs(r[1] - bv)

            md = max([ld, rd])

            if md < mig:
                continue  # TODO less than min info gain, could be issue for categorical

            s = (ld * np.log2(l[0] / samp * 100)) + (rd * np.log2(r[0] / samp * 100))  # TODO info gain equation?

            if s > bst:
                bst = s
                _idx = i

        return _idx

    def _construct(self):
        """ Constructs nodes for score binning """

        stack = [(self.base_partition, self.base_meta, 0, self.base_value, self.samples, np.array([]), np.array([]))]
        _nodes = []

        while stack:

            # First parent split (_) is ignored
            _splits, _meta, _depth, _mean, _count, _dir, _path = stack.pop()

            idx = self._best_split(
                _meta,
                self.abs_min_leaf_size,
                self.base_value,
                self.samples,
                self.min_info_gain
            )

            if (idx == -1) or (_depth >= self.max_depth):

                diff = _mean - self.base_value

                _freq = _count / self.samples

                if self.regressor:
                    score = (abs(diff) ** self.tail_sensitivity) * np.sign(diff)
                else:
                    score = self._activation(_freq*100) * diff

                self._min_score = min(self._min_score, score)  # update min score
                self._max_score = max(self._max_score, score)  # update max score

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

        return np.array(_nodes)

    def fit(self, X, y):
        """ Fits feature data to target """
        super().fit(X, y)
        self.abs_min_leaf_size = np.max([1, int(self.min_leaf_size * self.samples)])
        return self
