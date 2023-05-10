from ._base_partition import BasePartition
import pandas as pd
import numpy as np
from ..regression import XRegressor


class PartitionedRegressor(BasePartition):

    def __init__(self, partition_on=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.partition_on = partition_on

    def predict(self, x):
        x = pd.DataFrame(x).copy().reset_index(drop=True)

        if self.partition_on is None:
            model = self.partitions['__dataset__']
            return model.predict(x)

        else:
            partitions = self.partitions.keys()
            frames = []
            unq = [str(i) for i in list(x[self.partition_on].unique())]

            # replace unknown partition values with __dataset__ for general model
            for u in unq:
                if u not in partitions:
                    x[self.partition_on] = x[self.partition_on].replace(
                        u, '__dataset__')
                        
                    unq.remove(u)
                    if "__dataset__" not in unq:
                        unq.append("__dataset__")

            for partition in unq:
                part = x[x[self.partition_on].astype(str) == partition]
                idx = part.index

                # Use partition model first
                part_trans = self._transform(part, partition)
                _base_value = self.partitions[partition].base_value

                scores = pd.Series(part_trans.sum(axis=1) + _base_value)
                scores.index = idx
                frames.append(scores)
        
            return np.array(pd.concat(frames).sort_index())
