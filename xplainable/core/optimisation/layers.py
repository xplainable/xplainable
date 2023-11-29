""" Copyright Xplainable Pty Ltd, 2023"""

import numpy as np
import random
import math
from numba import njit, prange
from ...utils.numba_funcs import *
from .genetic import XEvolutionaryNetwork
import warnings
from tqdm.auto import tqdm


class BaseLayer:
    """ Base class for optimisation layers.

    Args:
        metric (str, optional): Metric to optimise on. Defaults to 'mae'.
    """

    def __init__(self, metric='mae'):
        self.xnetwork = None
        self.metric = metric

    def _calculate_error(self, x, y):
        """ Calculates the error of a set of predictions.

        Args:
            x (numpy.array): An array of transformed values.
            y (numpy.array): An array of the true values.

        Returns:
            np.array: An array of the errors.
        """

        # calculate relative error
        _pred = nansum_numba(x, axis=1) + self.xnetwork.model.base_value

        if self.xnetwork.static_scores is not None:
            _pred = _pred + self.xnetwork.static_scores

        if self.xnetwork.apply_range:
            _pred = _pred.clip(*self.xnetwork.model.prediction_range)

        return _pred - y

    def _increment_error(self, x, y):
        pass
    
    @staticmethod
    def _score():
        pass

    @staticmethod
    @njit
    def _mae(err):
        
        mae = np.mean(np.abs(err))

        return mae

    @staticmethod
    @njit
    def _mse(err):
        
        mse = np.mean(err**2)

        return mse

    def _initialise(self, metric):

        # Infer objective from metric
        if metric == 'mse':
            self._score = self._mse
            self.objective = 'minimise'

        elif metric == 'mae':
            self._score = self._mae
            self.objective = 'minimise'

        else:
            raise ValueError(f'Metric {metric} not supported')


class Evolve(BaseLayer):
    """ Evolutionary algorithm to optimise XRegressor leaf weights.

    The Evolve layer uses a genetic algorithm to optimise the leaf weights
    of an XRegressor model. The algorithm works by mutating the leaf weights
    of the model and scoring the resulting predictions. The best mutations
    are then selected to reproduce and mutate again. This process is repeated
    until the maximum number of generations is reached, or the early stopping
    threshold is reached.

    Args:
        mutations (int, optional): The number of mutations to generate per generation.
        generations (int, optional): The number of generations to run.
        max_generation_depth (int, optional): The maximum depth of a generation.
        max_severity (float, optional): The maximum severity of a mutation.
        max_leaves (int, optional): The maximum number of leaves to mutate.
        early_stopping (int, optional): Stop early if no improvement after n iters.
    """

    def __init__(
            self, mutations: int = 100, generations: int = 50,
            max_generation_depth: int = 10, max_severity: float = 0.5, 
            max_leaves: int = 20, early_stopping: int = None):
        super().__init__()
        
        self.mutations = mutations
        self.generations = generations
        self.max_generation_depth = max_generation_depth
        self.max_severity = max_severity
        self.max_leaves = max_leaves
        self.early_stopping = early_stopping

        self.objective = None

        self.xnetwork = None
        self.y = None
        self._error = None

        self.generation_id = 1

    def _get_params(self) -> dict:
        """ Returns the parameters of the layer.

        Returns:
            dict: The layer parameters.
        """

        params = {
            'mutations': self.mutations,
            'generations': self.generations,
            'max_generation_depth': self.max_generation_depth,
            'max_severity': self.max_severity,
            'max_leaves':self.max_leaves,
            'metric': self.metric
        }

        return params
    
    @property
    def params(self) -> dict:
        """ Returns the parameters of the layer.

        Returns:
            dict: The layer parameters.
        """

        return self._get_params()

    def _mutate(self, chromosome: np.array) -> np.array:
        """ Randomly mutates a chromosome.

        Args:
            chromosome (np.array): The chromosome to mutate.

        Returns:
            np.array: The mutated chromosome.
        """
        
        new_chromosome = np.array(chromosome)#.copy()
        chrome_length = chromosome.shape[0]

        # randomly select number of leaves to mutate
        num_leaves = random.sample(
            [i for i in range(1, int(min([self.max_leaves, chrome_length])+1))],k=1)[0]

        # identify features to mutate
        rand_indices = random.choices([i for i in range(int(chrome_length))], k=num_leaves)

        # Initiate delta mapping (store changes only)
        delta = {}
        for i in rand_indices:

            severity = random.randrange(1, int(self.max_severity * 1000 + 1)) / 1000
            mutation = random.choice([1 + severity, 1 - severity])

            delta[i] = mutation
            new_chromosome[i] *= mutation

        return new_chromosome, delta

    def _n_mutations(self, chromosome: np.array, n: int) -> tuple:
        """ Generates n mutations of a chromosome.

        Args:
            chromosome (np.array): The chromosome to mutate.
            n (int): The number of mutations to generate.

        Returns:
            tuple: The mutated chromosomes and deltas.
        """

        mutations = []
        deltas = []
        for _ in range(n):
            new_chromosome, delta = self._mutate(chromosome)
            mutations.append(new_chromosome)
            deltas.append(delta)

        return np.array(mutations), np.array(deltas)

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def _mutate_transform(x, delta_keys, delta_values):

        for k in prange(len(delta_keys)):
            i = int(delta_keys[k])
            v = delta_values[k]
            mask = ~np.isnan(x[:, i])
            
            for j in prange(x.shape[0]):
                if mask[j]:
                    x[j, i] = x[j, i] * v
                    
        return x

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def _inverse_mutate_transform(x, delta_keys, delta_values):
        """ Inverse mutation is faster than copying and altering the full array """

        for k in prange(len(delta_keys)):
            i = int(delta_keys[k])
            v = delta_values[k]
            mask = ~np.isnan(x[:, i])
            
            for j in prange(x.shape[0]):
                if mask[j]:
                    x[j, i] = x[j, i] / v
                    
        return x

    def _score_mutation(self, x: np.ndarray, delta: dict) -> float:
        """ Scores a mutation of a chromosome.

        Args:
            x (np.ndarray):  The input variables used for prediction.
            delta (dict): The mutation to apply.
        """

        # get the indices and delta changes for each mutation
        dkeys = np.array(list(delta.keys()))
        dvalues = np.array(list(delta.values()))

        # Apply mutation
        x = self._mutate_transform(x, dkeys, dvalues)

        # Score mutation
        err = self._calculate_error(x, self.y)
        score = self._score(err)
        
        # reverse mutation to maintain initial structure
        x = self._inverse_mutate_transform(x, dkeys, dvalues)
        
        return score

    def _reproduce(self, pair: tuple) -> dict:
        """ Merges the genes of a chromosome pair
        
        Args:
            pair (tuple): The pair of chromosomes to merge.
        """
        
        # get parent chromosomes
        a, b = pair
        
        child = (a + b) / 2

        return child

    def _get_delta(self, chromosome: np.array) -> dict:
        """ Gets the delta values for a chromosome.

        Args:
            chromosome (np.array): The chromosome to get deltas for.

        Returns:
            dict: The delta values.
        """

        return {
            i: v / self.target_chromosome[i] for i, v in enumerate(
                chromosome) if v != self.target_chromosome[i]
                }

    def _natural_selection(
            self, x: np.ndarray, mutations: np.array, deltas: dict
            ) -> np.array:
        """ Selects the best (superior) mutations from a pool.

        Args:
            x (np.ndarray): The input variables used for prediction.
            mutations (np.array): The mutations to select from.
            deltas (dict): The delta values for each mutation.

        Returns:
            np.array: The selected superior mutations.
        """

        # Score mutations
        scores = np.array([])
        for delta in deltas:
            scores = np.append(scores, self._score_mutation(x, delta))

        # Filter for superior mutations
        if self.objective == 'minimise':
            _filter = np.array(
                [True if i < self.target_score else False for i in scores])
        else:
            _filter = np.array(
                [True if i > self.target_score else False for i in scores])

        # Select superior mutations
        superiors = mutations[_filter]

        if superiors.shape[0] == 0:
            return np.array([[]])

        top_scores = scores[_filter]

        self.target_score = np.mean(top_scores)

        # when there are 2 equal scores, sorting is skipped
        if superiors.shape[0] > 1:
            try:
                superiors = np.array([x for _, x in sorted(zip(top_scores, superiors))])
            except Exception as e:
                pass

        return superiors

    def _reproduce_and_mutate(self, superiors: np.array) -> tuple:
        """ Reproduces and mutates a pool of superior chromosomes.

        Args:
            superiors (np.array): The superior chromosomes to reproduce.
        """
        # Pair mates together
        mates = [[superiors[i], superiors[i+1]] for i in range(0, superiors.shape[0] - 1, 2)]
        
        # Generate child chromosomes
        offspring = [self._reproduce(mate) for mate in mates]
        
        # Add most superior parent to pool
        offspring.append(superiors[0])

        for child in offspring.copy():
            offspring.append(self._mutate(child)[0])

        # Extract deltas from offspring
        deltas = [self._get_delta(i) for i in offspring]

        return np.array(offspring), np.array(deltas)

    def _run_generation(self, x: np.ndarray) -> np.array:
        """ Runs a single optimisation generation.

        Args:
            x (np.ndarray): The input variables used for prediction.
        """
        
        # Generate first n mutations
        m = self._n_mutations(self.target_chromosome, self.mutations)
        
        # select parent chromosomes
        parents = self._natural_selection(x, *m)

        for _ in range(self.max_generation_depth):
            if parents.shape[0] == 1:
                break
            m = self._reproduce_and_mutate(parents)
            parents = self._natural_selection(x, *m)

        return parents[0]

    def _re_map(self, x: np.ndarray) -> np.ndarray:
        """ Re-maps the optimised chromosome to the model profile.

        Args:
            x (np.ndarray): The input variables used for prediction.

        Returns:
            np.ndarray: A new optimised model profile.
        """

        chromosome = np.nanmax(x, axis=0)

        for l, v in zip(self.xnetwork.leaves, chromosome):
            f, _id = l.split("_")
            self.xnetwork.model._profile[int(f)][int(_id)][-4] = v
            self.xnetwork.model._constructs[int(f)]._nodes[int(_id)][-4] = v

        return self.xnetwork.model._profile

    def transform(self, xnetwork: XEvolutionaryNetwork, x: np.ndarray,
            y: np.array, callback=None):
        """ Optimises an XRegressor profile given the set of parameters.

        Args:
            xnetwork (XEvolutionaryNetwork): The evolutionary network.
            x (np.ndarray): The input variables used for prediction.
            y (np.array): The target values.
            callbacks (list): Callback function for progress tracking.

        Returns:
            np.ndarray: The original x data to pass to the next layer.
            np.ndarray: The final optimised chromosome to pass to the next layer.
        """
        
        self.xnetwork = xnetwork
        x = x.copy()
        self.y = y

        # map target string to int for clf models
        try:
            if len(self.xnetwork.model.target_map) > 0:
                self.y = np.vectorize(
                    lambda x: self.xnetwork.model.target_map[x])(self.y)
        except:
            pass

        self._initialise(self.metric)

        self.target_chromosome = xnetwork.root_chromosome
        delta = self._get_delta(xnetwork.root_chromosome)
        dkeys = np.array(list(delta.keys()))
        dvalues = np.array(list(delta.values()))

        transformed = self._mutate_transform(x, dkeys, dvalues)
        self._error = self._calculate_error(transformed, self.y)
        self.target_score = self._score(self._error)

        bst_score = float(self.target_score)

        if (callback is not None) and (self.xnetwork.layer_id == 0):
            callback.set_metric_bounds(0, bst_score)

        # iterations since best
        isb = 0
        # pbar = tqdm(["a", "b", "c", "d"])
        # for char in pbar:
        #     time.sleep(0.25)
        #     pbar.set_description("Processing %s" % char)
        generation_range = tqdm(range(1, self.generations+1))
        generation_range.set_description(f"Layer {len(self.xnetwork.completed_layers) + 1} (Evolve)")
        for i in generation_range:
            # Handle Early stopping
            if (self.early_stopping) is not None and (isb >= self.early_stopping):
                if callback:
                    callback.stopped_early(self.xnetwork.layer_id)

                self.xnetwork.checkpoint_score = float(bst_score)
                self._re_map(x)
                return x, self.target_chromosome

            gen = self._run_generation(x)
            delta = self._get_delta(gen)
            dkeys = np.array(list(delta.keys()))
            dvalues = np.array(list(delta.values()))

            x = self._mutate_transform(x, dkeys, dvalues)
            self._error = self._calculate_error(x, self.y)
            self.target_score = self._score(self._error)

            if callback:
                callback.set_value(self.xnetwork.layer_id, i)

            # Tracker for early stopping
            if self.target_score < bst_score:
                bst_score = self.target_score
                isb = 0
                if callback:
                    callback.set_metric(self.metric, round(bst_score, 4))
            else:
                isb += 1

            self.generation_id += 1

        self.xnetwork.checkpoint_score = float(bst_score)
        self._re_map(x)

        if callback:
            callback.finalise_bar(self.xnetwork.layer_id)

        return x, self.target_chromosome


class Tighten(BaseLayer):
    """ A leaf boosting algorithm to optimise XRegressor leaf node weights.

    The Tighten layer uses a novel leaf boosting algorithm to optimise the
    leaf weights of an XRegressor model. The algorithm works by iteratively
    identifying the leaf node that will have the greatest impact on the
    overall model score, and then incrementally increasing or decreasing
    the leaf node weight to improve the model score. This process is repeated
    until the maximum number of iterations is reached, or the early stopping
    threshold is reached.

        Args:
            iterations (int): The number of iterations to run.
            learning_rate (float): How fast the model learns. Between 0.001 - 1
            early_stopping (int): Stop early if no improvement after n iters.
    """

    def __init__(
            self, iterations: int = 100, learning_rate: float = 0.03,
            early_stopping: int = None):
        super().__init__()

        # store params
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.early_stopping = early_stopping

        self.xnetwork = None

    def _get_params(self) -> dict:
        """ Returns the parameters of the layer.

        Returns:
            dict: The layer parameters.
        """

        params = {
            'iterations': self.iterations,
            'learning_rate': self.learning_rate,
            'early_stopping': self.early_stopping,
            'metric': self.metric
        }

        return params
    
    @property
    def params(self) -> dict:
        """ Returns the parameters of the layer.

        Returns:
            dict: The layer parameters.
        """

        return self._get_params()

    def _next_best_change(self, errors: np.array) -> tuple:
        """ Identifies the most effective leaf node change to improve model.

        Args:
            errors (np.array): An array of errors for the current iteration.

        Returns:
            int: The index of the best leaf node.
            float: The max amount the score can change.
        """

        # load mask
        mask = self.xnetwork._mask

        # Build error map
        errmp = (np.transpose([errors] * mask.shape[1])) * mask

        # Set non-masked area to nan if zero
        errmp[(~mask) & (errmp == 0)] = np.nan

        # count even values
        ec = (errmp == 0).sum(axis=0)

        # Intantiate computation masks
        o_mask = (errmp > 0)
        u_mask = (errmp < 0)

        # Identify leaves with over-indexed errors
        o_vals = (errmp * o_mask)
        o_vals[o_vals == 0] = np.nan

        # Identify leaves with under-indexed errors
        u_vals = (errmp * u_mask)
        u_vals[u_vals == 0] = np.nan

        if self.metric == 'mse':
            o_vals = o_vals**2
            u_vals = u_vals**2

        # surpress know warning that raises when all values are nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            # get mean error and count of obs that are too high for each leaf
            om = np.nanmean(o_vals, axis=0)
            oc = o_mask.sum(axis=0)

            # get mean error and count of obs that are too low for each leaf
            um = np.nanmean(abs(u_vals), axis=0)
            uc = u_mask.sum(axis=0)

        # calculate max benefit of inc/dec each leaf
        inc = (um * uc) - (um * oc) - (um * ec)
        dec = (om * oc) - (om * uc) - (om * ec)

        # get the best outcomes
        bsts = np.nanmax([inc, dec])

        # find individual best outcome and loc
        bst = np.nanmax(bsts)
        bstloc = np.where(bsts==bst)[0][0]

        # get value to update by
        if inc[bstloc] > dec[bstloc]:
            change = math.sqrt(um[bstloc]) * self.learning_rate

        else:
            change = math.sqrt(om[bstloc]) * -1 * self.learning_rate

        #change = get_change(bstloc) * self.learning_rate
        return bstloc, change

    def _run_iteration(self, x: np.ndarray, y: np.array):
        """ Runs a single optimisation iteration.

        Args:
            x (np.ndarray): An array of transformed values.
            y (np.array): An array of the true values.

        Returns:
            numpy.array: An array of transformed values after opt iteration.
        """

        # calculate error
        err = self._calculate_error(x, y)

        # determine next best change
        bstloc, change = self._next_best_change(err)

        # apply best change
        x[:, bstloc][~np.isnan(x[:, bstloc])] += change

        # Update profile
        f, _id = self.xnetwork.leaves[bstloc].split("_")
        self.xnetwork.model._profile[int(f)][int(_id)][-4] += change

        return x

    def transform(
            self, xnetwork: XEvolutionaryNetwork, x: np.ndarray, y: np.array,
            callback=None) -> tuple:
        """ Optimises an XRegressor profile given the set of parameters.

        Args:
            x (np.ndarray): The input variables used for prediction.
            y (np.array): The target values.
            callback (any): Callback function for progress tracking.

        Returns:
            dict: The optimised feature score map.
        """
        
        self.xnetwork = xnetwork
        x = x.copy()

        self._initialise(self.metric)

        err = self._calculate_error(x, y)
        starting_score = self._score(err)

        # instantiate best df, best mae
        bst_score = starting_score
        best_x = x.copy()

        if (callback is not None) and (self.xnetwork.layer_id == 0):
            callback.set_metric_bounds(0, bst_score)

        # iterations since best
        isb = 0
        
        # start optimisation process
        generation_range = tqdm(range(1, self.iterations + 1))
        generation_range.set_description(f"Layer {len(self.xnetwork.completed_layers) + 1} (Tighten)")
        for i in generation_range:
            # stop early if early stopping threshold reached
            if (self.early_stopping) is not None and (isb >= self.early_stopping):
                if callback:
                    callback.stopped_early(self.xnetwork.layer_id)

                self.xnetwork.checkpoint_score = float(bst_score)

                return best_x, np.nanmax(best_x, axis=0)

            # run iteration
            x = self._run_iteration(x, y)

            # calculate mean absolute error
            err = self._calculate_error(x, y)

            _score = self._score(err)

            # update callback
            if callback is not None:
                callback.set_value(self.xnetwork.layer_id, i)

            # update output if best result
            if _score < bst_score:
                bst_score = _score
                best_x = x.copy()
                isb = 0
                
                if callback:
                    callback.set_metric(self.metric, round(bst_score, 4))

            else:
                isb += 1

        # update callback
        if callback:
            callback.finalise_bar(self.xnetwork.layer_id)

        self.xnetwork.checkpoint_score = float(bst_score)

        return best_x, np.nanmax(best_x, axis=0)
