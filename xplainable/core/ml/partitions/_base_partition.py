import numpy as np


class BasePartition:

    def __init__(self):
        self.partition_on = None
        self.partitions = {}
        self.categorical_columns = []
        self.numeric_columns = []
        self.id_columns = []
        self.target_map = {}
        self.target_map_inv = {}

    def add_partition(self, model, partition="__dataset__"):
        partition = str(partition)
        self.partitions[partition] = {}
        self.partitions[partition]['profile'] = model._profile
        self.partitions[partition]['base_value'] = model.base_value
        self.partitions[partition]['categorical_columns'] = model.categorical_columns
        self.partitions[partition]['numeric_columns'] = model.numeric_columns
        self.partitions[partition]['id_columns'] = model.id_columns
        self.partitions[partition]['feature_map'] = model.feature_map
        self.partitions[partition]['feature_map_inv'] = model.feature_map_inv

        try:
            self.partitions[partition]['calibration_map'] = model._calibration_map
        except:
            pass

        # update column names on first add
        if all([len(self.categorical_columns) == 0, len(self.numeric_columns) == 0]):
            self.id_columns = model.id_columns
            self.categorical_columns = model.categorical_columns
            self.numeric_columns = model.numeric_columns
            self.columns = list(model.columns)
            self.columns.remove(self.partition_on) if self.partition_on in self.columns else None
            self.target_map = model.target_map
            self.target_map_inv = model.target_map_inv
            
    def drop_partition(self, partition):
        self.partitions.pop(partition)

    def _encode(self, x, y=None, partition='__dataset__'):

        x = x.copy()

        # Apply encoding
        for f, m in self.partitions[partition]['feature_map'].items():
            x.loc[:, f] = x.loc[:, f].map(m)

        if y is not None:
            if len(self.target_map) > 0:
                y = y.map(self.target_map)
                
            y = y.astype(float)
            return x, y
        
        return x

    def _preprocess(self, x, y=None):
        
        x = x[self.columns]

        x = x.astype('float64')
        if y is not None:
            y = y.astype('float64')
            return x, y
        
        return x

    def _transform(self, x, partition):
        """ Transforms a dataset into the model weights.
        
        Args:
            x (pandas.DataFrame): The dataframe to be transformed.
            
        Returns:
            pandas.DataFrame: The transformed dataset.
        """

        assert(
            partition in self.partitions.keys(),
            f'Partition {partition} does not exist'
        )

        x = x.copy()

        x = self._encode(x, None, partition)
        x = self._preprocess(x).values

        profile = self.partitions[partition]['profile']

        for i in range(x.shape[1]):
            nodes = np.array(profile[i])
            idx = np.searchsorted(nodes[:, 1], x[:,i])

            known = np.where(idx < len(nodes))
            unknown = np.where(idx >= len(nodes)) # flag unknown categories
            
            x[unknown, i] = 0 # Set new categories to 0 contribution
            x[known, i] = nodes[idx[known], 2]

        return x
