""" Copyright Xplainable Pty Ltd, 2023"""

from ..utils.exceptions import TransformerError
import pandas as pd
import numpy as np


class XPipeline:
    """Pipeline builder for xplainable transformers.

    Args:
        stages (list): list containing xplainable pipeline stages.
    """

    def __init__(self):
        self.stages = []

    def add_stages(self, stages: list) -> 'XPipeline':
        """ Adds multiple stages to the pipeline.

        Args:
            stages (list): list containing xplainable pipeline stages.
        
        Returns:
            XPipeline: self
        """

        # Error handling
        if len(stages) == 0:
            raise ValueError('You must include at least one pipeline stage.')

        for stage in stages:
            # Searches for transformer funtion in stage class
            if 'transform' not in [f for f in dir(stage['transformer'])]:
                raise TypeError(f'{type(stage)} type is not supported.')
            
            # Searches for dataset level transformers
            stage['feature'] = stage.get('feature', None)
            if stage['feature'] is None:
                stage['feature'] = '__dataset__'

            stage['name'] = stage['transformer'].__class__.__name__

            self.stages.append(stage)

        return self

    def drop_stage(self, stage: int) -> 'XPipeline':
        """ Drops a stage from the pipeline.

        Args:
            stage (int): index of the stage to drop.

        Returns:
            XPipeline: self
        """

        if len(self.stages) == 0:
            raise ValueError(f"There are no stages for in the pipeline.")

        if stage > len(self.stages) - 1:
            raise IndexError(f"Index {stage} out of bounds")

        self.stages.pop(stage)

        return self

    def fit(self, x: pd.DataFrame) -> 'XPipeline':
        """ Sequentially iterates through pipeline stages and fits data.
        
        Args:
            x (pd.DataFrame): A non-empty DataFrame to fit.

        Returns:
            XPipeline: The fitted pipeline.
        """

        x = x.copy()

        for i, stage in enumerate(self.stages):

            if stage['feature'] == '__dataset__':
                continue

            # Check for features that have appeared before
            prev_feature_transformers = [s for s in self.stages[:i] if s['feature'] == stage["feature"]]
            
            # Apply previous transformation if appeared before (for chaining)
            if len(prev_feature_transformers) > 0:
                tf = prev_feature_transformers[-1]['transformer']
                x[stage['feature']] = tf.transform(x[stage['feature']])

            # Fit data to transformer
            
            stage['transformer'].fit(x[stage['feature']])

        return self

    def transform(self, x: pd.DataFrame):
        """ Iterates through pipeline stages applying transformations.
        
        Args:
            x (pd.DataFrame): A non-empty DataFrame to transform.
        
        Returns:
            pd.DataFrame: The transformed dataframe.
        """

        x = x.copy()

        # Apply all transformers to dataset
        for stage in self.stages:
            try:
                if stage['feature'] == '__dataset__':
                
                    x = stage['transformer'].transform(x)
                
                    continue
            
                if stage['feature'] not in x.columns:
                    continue

                x[stage['feature']] = stage['transformer'].transform(x[stage['feature']])
            except Exception:
                tf_name = stage['transformer'].__class__.__name__
                raise TransformerError(
                    f"Transformer {tf_name} for feature {stage['feature']} failed. Ensure the datatypes are compatible") from None

        return x
    
    def fit_transform(self, x: pd.DataFrame, start: int = 0):
        """ Runs the fit method followed by the transform method.
        
        Args:
            x (pd.DataFrame): A non-empty DataFrame to fit.
            start (int): index of the stage to start fitting from.

        Returns:
            pd.DataFrame: The transformed dataframe.
        """

        x = x.copy()

        for stage in self.stages[start:]:
            try:
                if stage['feature'] == '__dataset__':
                    stage['transformer'].fit(x)
                    x = stage['transformer'].transform(x)

                    continue

                # Fit data to transformer
            
                stage['transformer'].fit(x[stage['feature']])
            
                # Apply transformation for chaining
                x[stage['feature']] = stage['transformer'].transform(
                    x[stage['feature']])
            
            except Exception:
                tf_name = stage['transformer'].__class__.__name__
                raise TransformerError(
                    f"Transformer {tf_name} for {stage['feature']} failed. Ensure the datatypes are compatible") from None

        return x
    
    def transform_generator(self, x):
        """transform generator"""

        x = x.copy()

        for stage in self.stages:
            try:
                # If the stage is for the entire dataset, transform and yield
                if stage['feature'] == '__dataset__':
                    x = stage['transformer'].transform(x)
                    yield x
                    continue

                # Apply transformation for a specific feature and yield
                x_transformed = x.copy()
                x_transformed[stage['feature']] = stage['transformer'].transform(
                    x[stage['feature']])
                yield x_transformed
            
            except Exception:
                tf_name = stage['transformer'].__class__.__name__
                raise TransformerError(
                    f"Transformer {tf_name} for {stage['feature']} failed. Ensure the datatypes are compatible") from None
    
    def inverse_transform(self, x: pd.DataFrame):
        """ Iterates through pipeline stages applying inverse transformations.
        
        Args:
            x (pd.DataFrame): A non-empty DataFrame to inverse transform.
        
        Returns:
            pd.DataFrame: The inverse transformed dataframe.
        """

        x = x.copy()

        # Apply all transformers to dataset
        for stage in self.stages:
            try:
                if stage['feature'] == '__dataset__':
                
                    x = stage['transformer'].inverse_transform(x)
                
                    continue
            
                if stage['feature'] not in x.columns:
                    continue

                x[stage['feature']] = stage['transformer'].inverse_transform(x[stage['feature']])
            except Exception:
                raise TransformerError(
                    f"Transformer for feature {stage['feature']} failed. Ensure the datatypes are compatible") from None

        return x
    
    def get_blueprint(self):
        """ Returns a blueprint of the pipeline.

        Returns:
            list: A list containing the pipeline blueprint.
        """
        
        blueprint = []
        for stage in self.stages:
            bstage = {"feature": stage['feature']}
            bstage['transformer'] = stage['transformer'].__class__.__name__
            bstage['args'] = stage['transformer'].__dict__
            blueprint.append(bstage)

        return blueprint
