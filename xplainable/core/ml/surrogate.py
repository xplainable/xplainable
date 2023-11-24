import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
from typing import Union, Any

from classification import XClassifier
from regression import XRegressor


class XSurrogateClassifier(XClassifier):

    def __init__(
        self,
        child_model,
        child_predict_method: None,
        max_depth: int = 8,
        min_info_gain: float = -1,
        min_leaf_size: float = -1,
        alpha: float = 0.1,
        weight: float = 0.05,
        power_degree: float = 1,
        sigmoid_exponent: float = 1,
        map_calibration: bool = True
    ):
        super().__init__(
            max_depth,
            min_info_gain,
            min_leaf_size,
            alpha,
            weight,
            power_degree,
            sigmoid_exponent,
            map_calibration
        )
        self._child_model = child_model
        self._child_predict_method = child_predict_method

    def fit(
        self,
        x: Union[pd.DataFrame, np.ndarray],
        child_model_prediction_args: tuple = (),
        id_columns: list = [],
        column_names: list = None,
        target_name: str = 'target'
    ) -> 'XSurrogateClassifier':
        """ Fits the model to the data.

        Args:
            x (pd.DataFrame | np.ndarray): The x variables used for training.
            child_model_prediction_args tuple(Any): any args to the child's predict method.
            id_columns (list, optional): id_columns to ignore from training.
            column_names (list, optional): column_names to use for training if using a np.ndarray
            target_name (str, optional): The name of the target column if using a np.array

        Returns:
            XClassifier: The fitted model.
        """

        if self._child_predict_method is not None:
            assert callable(self._child_predict_method)
            y = self._child_predict_method(x, *child_model_prediction_args)
        else:
            assert hasattr(self._child_model, "predict") and callable(self._child_model.predict)
            y = self._child_model.predict(x, *child_model_prediction_args)

        super().fit(
            x,
            y,
            id_columns,
            column_names,
            target_name
        )

        return self


class XSurrogateRegressor(XRegressor):

    def __init__(
        self,
        child_model,
        child_predict_method: None,
        max_depth: int = 8,
        min_leaf_size: float = 0.02,
        min_info_gain: float = 0.02,
        alpha: float = 0.01,
        tail_sensitivity: float = 1,
        prediction_range: tuple = (-np.inf, np.inf)
    ):
        super().__init__(
            max_depth,
            min_leaf_size,
            min_info_gain,
            alpha,
            tail_sensitivity,
            prediction_range
        )
        self._child_model = child_model
        self._child_predict_method = child_predict_method

    def fit(
        self,
        x: Union[pd.DataFrame, np.ndarray],
        child_model_prediction_args: tuple = (),
        id_columns: list = [],
        column_names: list = None,
        target_name: str = 'target'

    ) -> 'XSurrogateRegressor':
        """ Fits the model to the data.

        Args:
            x (pd.DataFrame | np.ndarray): The x variables used for training.
            child_model_prediction_args tuple(Any): any args to the child's predict method.
            id_columns (list, optional): id_columns to ignore from training.
            column_names (list, optional): column_names to use for training if using a np.ndarray
            target_name (str, optional): The name of the target column if using a np.array

        Returns:
            XSurrogateRegressor: The fitted model.
        """

        if self._child_predict_method is not None:
            assert callable(self._child_predict_method)
            y = self._child_predict_method(x, *child_model_prediction_args)
        else:
            assert hasattr(self._child_model, "predict") and callable(self._child_model.predict)
            y = self._child_model.predict(x, *child_model_prediction_args)

        super().fit(
            x,
            y,
            id_columns,
            column_names,
            target_name
        )

        return self


