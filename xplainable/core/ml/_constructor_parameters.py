""" Copyright Xplainable Pty Ltd, 2023"""


class ConstructorParams:

    def __init__(
        self,
        max_depth=8,
        min_info_gain=0.0001,
        min_leaf_size=0.0001,
        ignore_nan=False,
        weight=1,
        power_degree=1,
        sigmoid_exponent=0,
        tail_sensitivity: float = 1.0,
    ):
        # regressor parameters
        self.max_depth = max_depth
        self.min_info_gain = min_info_gain
        self.min_leaf_size = min_leaf_size
        # score normalisation parameters
        self.ignore_nan = ignore_nan
        self.weight = weight
        self.power_degree = power_degree
        self.sigmoid_exponent = sigmoid_exponent
        self.tail_sensitivity = tail_sensitivity

        self._check_param_bounds()

    def _check_param_bounds(self):
        assert self.max_depth >= 0, 'max_depth must be greater than or equal to 0'
        assert -1 <= self.min_info_gain < 1, 'min_info_gain must be between -1 and 1'
        assert -1 <= self.min_leaf_size < 1, 'min_leaf_size must be between -1 and 1'
        assert self.ignore_nan in [True, False, 1, 0]
        assert 0 <= self.weight <= 3, 'weight must be between 0 and 3'
        assert self.power_degree in [1, 3, 5], 'powed_r_degree must be 1, 3, or 5'
        assert 0 <= self.sigmoid_exponent <= 1, 'sigmoiexponent must be between 0 and 1'

    def update_parameters(
        self,
        max_depth=None,
        min_info_gain=None,
        min_leaf_size=None,
        ignore_nan=None,
        weight=None,
        power_degree=None,
        sigmoid_exponent=None,
        tail_sensitivity: float = None,
    ):
        self.max_depth = max_depth if max_depth is not None else self.max_depth
        self.min_info_gain = min_info_gain if min_info_gain is not None else self.min_info_gain
        self.min_leaf_size = min_leaf_size if min_leaf_size is not None else self.min_leaf_size
        self.ignore_nan = ignore_nan if ignore_nan is not None else self.ignore_nan
        self.weight = weight if weight is not None else self.weight
        self.power_degree = power_degree if power_degree is not None else self.power_degree
        self.sigmoid_exponent = sigmoid_exponent if sigmoid_exponent is not None else self.sigmoid_exponent
        self.tail_sensitivity = tail_sensitivity if tail_sensitivity is not None else self.tail_sensitivity

        self._check_param_bounds()

    def to_json(self):
        return {
            "max_depth": self.max_depth,
            "min_info_gain": self.min_info_gain,
            "min_leaf_size": self.min_leaf_size,
            "ignore_nan": self.ignore_nan,
            "weight": self.weight,
            "power_degree": self.power_degree,
            "sigmoid_exponent": self.sigmoid_exponent,
            "tail_sensitivity": self.tail_sensitivity
        }

    @staticmethod
    def from_json(data):
        return ConstructorParams(
            max_depth=data["max_depth"],
            min_info_gain=data["min_info_gain"],
            min_leaf_size=data["min_leaf_size"],
            ignore_nan=data["ignore_nan"],
            weight=data["weight"],
            power_degree=data["power_degree"],
            sigmoid_exponent=data["sigmoid_exponent"],
            tail_sensitivity=data["tail_sensitivity"]
        )

    def __copy__(self):
        return ConstructorParams.from_json(self.to_json())

    def __repr__(self):
        return str(self.to_json())

    def __eq__(self, other):
        if type(other) == ConstructorParams:
            if self.to_json() == other.to_json():
                return True
        return False
