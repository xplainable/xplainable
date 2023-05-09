import numpy as np


class BasePartition:

    def __init__(self):
        self.partition_on = None
        self.partitions = {}

    def __verify_mappings(self, model):
        assert model.target_map == self.partitions['__dataset__'].target_map, \
            "Target mappings are mismatched"

    def add_partition(self, model , partition):
        partition = str(partition)
        self.partitions[partition] = model
        if hasattr(model, 'target_map'):
            self.__verify_mappings(model)
            
    def drop_partition(self, partition):
        self.partitions.pop(partition)

    def _encode(self, x, y=None, partition='__dataset__'):

        x = x.copy()
        partition = str(partition)

        # Apply encoding
        for f, m in self.partitions[partition].feature_map.items():
            x.loc[:, f] = x.loc[:, f].map(m)

        if y is not None:
            if len(self.partitions[partition].target_map) > 0:
                y = y.map(self.partitions[partition].target_map)
                
            y = y.astype(float)
            return x, y
        
        return x

    def _preprocess(self, x, y=None):
        
        x = x[self.partitions['__dataset__'].columns]

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
            str(partition) in self.partitions.keys(),
            f'Partition {partition} does not exist'
        )

        x = x.copy()
        partition = str(partition)

        x = self._encode(x, None, partition)
        x = self._preprocess(x).values

        profile = self.partitions[partition]._profile

        for i in range(x.shape[1]):
            nodes = np.array(profile[i])
            idx = np.searchsorted(nodes[:, 1], x[:,i])

            known = np.where(idx < len(nodes))
            unknown = np.where(idx >= len(nodes)) # flag unknown categories
            
            x[unknown, i] = 0 # Set new categories to 0 contribution
            x[known, i] = nodes[idx[known], 2]

        return x
