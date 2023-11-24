""" Copyright Xplainable Pty Ltd, 2023"""

import numpy as np
import random
import pandas as pd


class Target:
    
    def __init__(self, model, tolerance=0.005):
        self.base_value = model.base_value
        self.score = model.base_value
        self.columns = list(model.columns)
        self.tolerance = tolerance
        
        self._profile = model._profile
        self.locked = {}
        
    def __start(self):
        mappings = []

        for i, c in enumerate(self.columns):
            _prof = self._profile[i]
            mappings.append(
                list(sorted([
                    (x, y[2]) for x, y in enumerate(_prof)],
                    key=lambda p: p[1]))
                    )
        
        self.score = self.base_value
        self.mappings = mappings
        
        self.not_maxxed = []
        self.not_minned = []
        
        self.scores = []
        self.idxs = []
        self.best_score = None
        self.best_diff = np.inf
        self.best_idxs = []
        self.values = []
        
        self.__random_init()
        self.__update_bounds()
    
    def __random_init(self):
        # init random
        for i, v in enumerate(self.mappings):
            # store scores for locked values
            if self.columns[i] in self.locked.keys():
                nodes = np.array(self._profile[i])
                idx = np.searchsorted(
                    nodes[:, 1], self.locked[self.columns[i]])

                cidx = v.index([i for i in v if i[0] == idx][0])
                score = nodes[idx][2]
                self.score += score
            # randomly select non-locked
            else:
                c = random.choice(v)
                score = c[1]
                self.score += score
                cidx = v.index(c)
                
            self.scores.append(score)
            self.idxs.append(cidx)
    
    def __update_bounds(self):
        self.not_maxxed = [
            False if v == len(self.mappings[i]) -1 or self.columns[i] \
                in self.locked else True for i, v in enumerate(self.idxs)]

        self.not_minned = [
            False if v == 0 or self.columns[i] in self.locked else True \
                for i, v in enumerate(self.idxs)]
    
    def __random_increase(self):
        
        c = random.choice(np.array(self.columns)[self.not_maxxed])
        col_idx = self.columns.index(c)
        idx = self.idxs[col_idx]
        
        new_idx = idx + 1
        new_score = self.mappings[col_idx][new_idx][1]
        self.score -= self.scores[col_idx]
        self.score += new_score
        self.scores[col_idx] = new_score
        self.idxs[col_idx] = new_idx
            
        self.__update_bounds()
    
    def __random_decrease(self):

        c = random.choice(np.array(self.columns)[self.not_minned])
        col_idx = self.columns.index(c)
        idx = self.idxs[col_idx]
        
        new_idx = idx - 1
        new_score = self.mappings[col_idx][new_idx][1]
        self.score -= self.scores[col_idx]
        self.score += new_score
        self.scores[col_idx] = new_score
        self.idxs[col_idx] = new_idx
            
        self.__update_bounds()
    
    def run(self, target, iterations=1000, locked={}):
        """ Finds model leaf nodes to achieve target value.

        Example:

        target = Target(model)

        nodes = target.run(
            target=0.62,
            iterations=1000,
            locked={'Age': 32, 'Balance': 20000, 'IsActiveMember': 1}
            )


        Args:
            target (float): The target value.
            iterations (int): Number of iterations to run.
            locked (dict, optional): Lock feature values and drop from search.

        Returns:
            dict: Optimised leaf nodes for each feature.
        """

        self.locked = locked
        self.__start()

        for _ in range(iterations):
            if self.score < target:
                if sum(self.not_maxxed) == 0:
                    break
                self.__random_increase()
            elif self.score > target:
                if sum(self.not_minned) == 0:
                    break
                self.__random_decrease()
            else:
                break
            
            diff = abs(self.score - target)
            if diff < self.best_diff:
                self.best_diff = diff
                self.best_score = self.score
                self.best_idxs = [self.mappings[i][v] for i, v in enumerate(self.idxs)]

            if diff <= self.tolerance:
                self.values.append(self.best_score)
                break

            self.values.append(self.best_score)

        nodes = [self._profile[i][v[0]] for i, v in enumerate(self.best_idxs)]
                        
        return {f: list(n) for f, n in zip(self.columns, nodes)}


def generate_ruleset(
    df,
    target,
    id_columns=[],
    min_support=10,
    include_numeric=False
    ):

    rule_data = df.drop(columns=id_columns+[target])
    rule_data = rule_data.loc[:, (rule_data != 0).any(axis=0)]
    cols = list(rule_data.columns)

    num_cols = rule_data._get_numeric_data().columns

    if include_numeric:
        for col in num_cols:
            intervals = pd.cut(
                rule_data[col], min([10, rule_data[col].nunique()]))
            left = intervals.copy().apply(lambda x: x.left)
            right = intervals.copy().apply(lambda x: x.right)
            rule_data[col] = pd.Series([tuple(x) for x in zip(left, right)])

    else:
        cols = [col for col in cols if col not in num_cols]
        rule_data = rule_data[cols]
    
    if len(cols) == 0:
        return []

    dummied = pd.get_dummies(rule_data, dummy_na=False, prefix_sep='___')

    support_rules = {i: [u for u in rule_data[i].value_counts()[
        rule_data[i].value_counts() < min_support].index] for i in cols}
    
    frames = []
    
    for col in list(dummied.columns):

        t = dummied.groupby(col).sum().T

        if 0 not in t.columns:
            continue

        matches = t[(t[0] > 0) & (t[1] == 0)]
        _c, _v = col.split('___')
        
        if _v in support_rules[_c]:
            continue
        
        matches = matches[~matches.index.str.startswith(_c)]
        if len(matches) > 0:
            feature_rules = pd.DataFrame({
                'feature': _c,
                'value': _v,
                'rule': matches.index.values
            })
            feature_rules[['rule_feature', 'rule_value']] = feature_rules[
                'rule'].str.split('___', expand=True)

            feature_rules.drop(columns=['rule'], inplace=True)
            
            frames.append(feature_rules)
        
    r = pd.concat(frames)
    r = r[r.apply(lambda row: row['rule_value'] not in support_rules[
        row['rule_feature']], axis=1)]
        
    return r.to_json(orient='records')