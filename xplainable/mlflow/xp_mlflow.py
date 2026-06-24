import mlflow
from mlflow.models import infer_signature
import copy
from ..core.models import XClassifier,XRegressor,build_surrogate_model
import os
import xplainable as xp
import pickle
from .utils import CustomXplainableModel
import warnings
from datetime import datetime
from typing import Union,Optional
import sklearn
import pandas as pd
import numpy as np

ModelTypeHint = Union[XClassifier,XRegressor,sklearn.base.BaseEstimator]

def log_explanation(model: ModelTypeHint,
                    X: Union[pd.DataFrame,np.ndarray],
                    y: Union[pd.Series,np.ndarray],
                    model_predict: str = 'predict',
                    model_name: Optional[str] = None,
                    model_description: Optional[str] = None,
                    mlflow_log_path : Optional[str] = None,
                    build_surrogate_model_params: Optional[dict] = None):
    
    """This functions is logs the explanation of a model by building a surrogate model. 
    A surrogate model is built and logged to xplainable cloud  (if api key is active) and mlflow (if tracking server is active).
    If model is of type XRegressor/XClassifier surrogate model is not built and only explanation is logged.

    
    Args:
        model: Preditor model of which explanation needs to be logged.
        X: Data on which surrogate model should be fit.
        y: Target for Data
        objective : possible values ['auto','classification','regression']. When auto, it infers the mode
        model_predict : callable for model to obtain model predictions
        model_name : name of model to be logged to xplainable cloud and ml flow
        model_description : description of model to be logged 
        mlflow_log_path : Path for artifact_path in mlflow.pyfunc.log_model 
        build_surrogate_model_params : arguments to be passed to build_surrogate_model func
    
    Returns:
        surrogate_model : Fitted XRegressor or XClassifier based on objective.
        model_id : model id of logged surrogate model
        version : version id of logged surrogate model
    """
    
    surrogate_model = None
    model_id = None
    version_id = None
  
    if isinstance(model,(XClassifier,XRegressor)):
        surrogate_model  = copy.deepcopy(model)
        warnings.warn(f'{type(model)} model already present. Skipping building surrogate model')

    else:

        if not hasattr(model,model_predict):
            raise ValueError(f'{model_predict} predict method not present in model')
        else:
            model_predict_func = getattr(model,model_predict)

        if build_surrogate_model_params is None:
            build_surrogate_model_params  = {}
        surrogate_model = build_surrogate_model(model_predict_func,X,**build_surrogate_model_params)
    

    try:
        if model_name is None:
            model_name = 'Surrogate_Model_'+str(datetime.now())
            warnings.warn(f'model name not provided. Logging model with name {model_name}')
        
        if model_description is None:
            model_description = ''

        model_id = xp.client.create_model_id(
        surrogate_model,
        model_name=model_name,
        model_description=model_description)
  
        version_id = xp.client.create_model_version(
        surrogate_model,
        model_id,
        X,
        y)

    except Exception as e:
        warnings.warn(e)

    try:
        if mlflow_log_path is None:

            mlflow_log_path = "xplainable_model"
            warnings.warn(f'mlflow_log_path not provided. Logging model to {mlflow_log_path}/')

        with mlflow.start_run() as run:

            run_id = run.info.run_id
            warnings.warn(f'Logging model with to run id {run_id}')
            model_signature  = infer_signature(X,surrogate_model.predict(X))

            if not os.path.exists(mlflow_log_path):
                os.makedirs(mlflow_log_path,exist_ok=True)

            model_path = os.path.join(mlflow_log_path, "model.pkl")

            
            with open(model_path, "wb") as f:
                pickle.dump(surrogate_model, f)

            # Create a dictionary to tell MLflow where the necessary artifacts are
            artifacts = {
                "model": model_path
            }

            mlflow.pyfunc.log_model(
                artifact_path = mlflow_log_path,
                python_model=CustomXplainableModel(),
                artifacts=artifacts,
                signature=model_signature,
                registered_model_name=model_name,
            )

    except Exception as e:
        warnings.warn(e)

    return surrogate_model,model_id,version_id

    