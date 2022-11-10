

class XPipeline:
    """Pipeline builder for xplainable transformers.
    """

    def __init__(self):
        self.stages = []

    def add_stages(self, stages: list):
        """ Adds multiple stages to the pipeline.
        Args:
            stages (list): list containing xplainable pipeline stages.
        Returns:
            self
        """

        # Error handling
        if len(stages) == 0:
            raise ValueError('You must include at least one pipeline stage.')

        for stage in stages:
            # Searches for transformer funtion in stage class
            if 'transform' not in [f for f in dir(stage['transformer'])]:
                raise TypeError(f'{type(stage)} type is not supported.')

            self.stages.append(stage)

        return

    def drop_stage(self, stage: int):

        if len(self.stages) == 0:
            raise ValueError(f"There are no stages for in the pipeline.")

        if stage > len(self.stages) - 1:
            raise IndexError(f"Index {stage} out of bounds")

        self.stages.pop(stage)

        return

    def fit(self, X):
        """ Iterates through pipeline stages and fitting data.
        Args:
            X (pandas.DataFrame): A non-empty DataFrame to fit.
        """

        X = X.copy()

        for i, stage in enumerate(self.stages):

            if stage['feature'] == '__dataset__':
                continue

            # Check for features that have appeared before
            prev_feature_transformers = [s for s in self.stages[:i] if s['feature'] == stage["feature"]]
            
            # Apply previous transformation if appeared before (for chaining)
            if len(prev_feature_transformers) > 0:
                tf = prev_feature_transformers[-1]['transformer']
                X[stage['feature']] = tf.transform(X[stage['feature']])

            # Fit data to transformer
            
            stage['transformer'].fit(X[stage['feature']])

        return self

    def fit_transform(self, X, start=0):
        """ Iterates through pipeline stages and fitting data.
        Args:
            X (pandas.DataFrame): A non-empty DataFrame to fit.
        """

        X = X.copy()

        for stage in self.stages[start:]:

            if stage['feature'] == '__dataset__':
                stage['transformer'].fit(X)
                X = stage['transformer'].transform(X)
                continue

            # Fit data to transformer
            stage['transformer'].fit(X[stage['feature']])

            # Apply transformation for chaining
            X[stage['feature']] = stage['transformer'].transform(X[stage['feature']])

        return X

    def transform(self, X):
        """ Iterates through pipeline stages applying transformations.
        Args:
            X (pandas.DataFrame): A non-empty DataFrame to transform.
        Returns:
            pandas.DataFrame: The transformed dataframe.
        """

        X = X.copy()

        # Apply all transformers to dataset
        for stage in self.stages:
            if stage['feature'] == '__dataset__':
                X = stage['transformer'].transform(X)
                continue
            
            X[stage['feature']] = stage['transformer'].transform(X[stage['feature']])

        return X

    def get_blueprint(self):
        
        blueprint = []
        for stage in self.stages:
            bstage = {"feature": stage['feature']}
            bstage['transformer'] = stage['transformer'].__class__.__name__
            bstage['args'] = stage['transformer'].__dict__
            blueprint.append(bstage)

        return blueprint