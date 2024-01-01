import mlflow
import pandas as pd
import numpy as np
from typing import Union,Optional


class CustomXplainableModel(mlflow.pyfunc.PythonModel):
  
    def load_context(self, context):
        with open(context.artifacts["model"], "rb") as f:
            import pickle
            self.model = pickle.load(f)
      
    def predict(self, context, model_input : Union[pd.DataFrame,np.ndarray], 
                predict_params : Optional[dict] = None) -> np.ndarray:
        """Predict method for custom mlflow XRegressor/XClassifier Wrapper.
        Args:
            context:  mlflow context
            model_input: Data for which to generate predictions    
            predict_params: Predict parameters.

        Returns:
            outputs: predictions from model

        """
        if predict_params is None:
            predict_params = {}
        outputs = self.model.predict(model_input,**predict_params)
        return outputs