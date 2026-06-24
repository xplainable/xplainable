import numpy as np
import pandas as pd
from typing import Callable,Union,Optional
from .classification import XClassifier
from .regression import XRegressor
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def build_surrogate_model(model_predict: Callable,
                          X: Union[pd.DataFrame,np.ndarray],
                          objective : str ='auto',
                         model_predict_args : Optional[dict] = None,
                         surrogate_model_params : Optional[dict] = None,
                         surrogate_model_fit_params : Optional[dict] = None) -> Union[XClassifier,XRegressor]:
    
    """The function builds a surrogate model based on the objective. 
    XRegressor in case of regression and XClassifier in case of classification

    Args:
        model_predict: the predict function for the surrogate model to build.
        X: Data on which surrogate model should be fit.
        objective : possible values ['auto','classification','regression']. When auto, it infers the mode
        model_predict_args : arguments to be passed to model predict callable
        surrogate_model_params : arguments to be passed during surrogate  model intialisation
        surrogate_model_fit_params : arguments to be passed during surrogate model fit.

    Returns:
        surrogate_model : Fitted XRegressor or XClassifier based on objective.
    """
    

    if model_predict_args is None:
        model_predict_args = {}
    
    if surrogate_model_params is None:
        surrogate_model_params = {}
    
    if surrogate_model_fit_params is None:
        surrogate_model_fit_params = {}

    y = model_predict(X,**model_predict_args)

    if objective=='auto':
        if pd.Series(y).nunique() == 2:
            objective = 'classification'
        else:
            objective = 'regression'

    if objective == 'classification':
        surrogate_model = XClassifier(**surrogate_model_params)
    
    elif objective == 'regression':
        surrogate_model = XRegressor(**surrogate_model_params)
    
    
    surrogate_model.fit(X,y,**surrogate_model_fit_params)

    return surrogate_model


        

        
        


