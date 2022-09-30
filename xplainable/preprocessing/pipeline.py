

class XPipeline:
    """Pipeline builder for xplainable transformers.
    """

    def __init__(self):
        self.stages = []
        self.stage_names = []

    def add_stage(self, stages=[], stage_names=[]):
        """ Adds a stage to the pipeline.
        Args:
            stages (list): list containing xplainable pipeline stages.
            stage_names (list): List of stage descriptions.
        Returns:
            self
        """

        # Error handling
        if len(stages) == 0:
            raise ValueError('You must include at least one pipeline stage.')

        elif len(stages) > 0 and len(stage_names) == 0:
            stage_names = ["No name"] * len(stages)

        if len(stages) != len(stage_names):
            raise ValueError("stages and stage_names must have same length")

        for stage, name in zip(stages, stage_names):
            
            # Searches for transformer funtion in stage class
            if 'transform' not in [f for f in dir(stage)]:
                raise TypeError(f'{type(stage)} type is not supported.')

            self.stages.append(stage)
            self.stage_names.append(name)

        return self

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
            X = stage.transform(X)

        return X