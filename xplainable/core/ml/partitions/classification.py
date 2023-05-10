from ._base_partition import BasePartition
import pandas as pd
import numpy as np
from ..classification import XClassifier


class PartitionedClassifier(BasePartition):

    def __init__(self, partition_on=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.partition_on = partition_on

    def predict_score(self, x, proba=False):
        x = pd.DataFrame(x).copy().reset_index(drop=True)

        if self.partition_on is None:
            model = self.partitions['__dataset__']
            return model.predict_score(x)

        else:
            partitions = self.partitions.keys()
            frames = []
            unq = list(x[self.partition_on].unique())

            # replace unknown partition values with __dataset__ for general model
            for u in unq:
                if u not in partitions:
                    x[self.partition_on] = x[self.partition_on].replace(u, '__dataset__')
                    unq.remove(u)
                    if "__dataset__" not in unq:
                        unq.append("__dataset__")

            partition_map = []
            for partition in unq:
                part = x[x[self.partition_on] == partition]
                idx = part.index

                # Use partition model first
                part_trans = self._transform(part, partition)
                _base_value = self.partitions[partition].base_value

                scores = pd.Series(part_trans.sum(axis=1) + _base_value)
                scores.index = idx
                frames.append(scores)

                if proba:
                    [partition_map.append((i, partition)) for i in idx]
        
            all_scores = np.array(pd.concat(frames).sort_index())

        if proba:
            partition_map = np.array(partition_map)
            partition_map = partition_map[partition_map[:, 0].argsort()][:,1]
            return all_scores, partition_map

        return all_scores

    def predict_proba(self, x):
        """ Predicts probability an observation falls in the positive class.

        Args:
            x: A dataset containing the observations.
            
        Returns:
            numpy.array: An array of predictions.
        """

        if self.partition_on is None:
            model = self.partitions['__dataset__']
            return model.predict_proba(x)

        scores, partition_map = self.predict_score(x, True)
        scores = (scores * 100).astype(int)
        
        def get_proba(p, score):
            mapp = self.partitions[str(p)]._calibration_map
            return mapp.get(score)

        scores = np.vectorize(get_proba)(partition_map, scores)

        return scores

    def predict(self, x, use_prob=False, threshold=0.5):
        """ Predicts if an observation falls in the positive class.
        
        Args:
            x (pandas.DataFrame): A dataset containing the observations.
            use_prob (bool): Uses 'probability' instead of 'score' if True.
            threshold (float): The prediction threshold.
            
        Returns:
            numpy.array: Array of predictions.
        """

        # Get the score for each observation
        y_pred = self.predict_proba(x) if use_prob else self.predict_score(x)

        # Return 1 if feature value > threshold else 0
        pred = pd.Series(y_pred).map(lambda x: 1 if x >= threshold else 0)

        map_inv  = self.partitions['__dataset__'].target_map_inv

        if map_inv:
            return np.array(pred.map(map_inv))
        else:
            return np.array(pred)
