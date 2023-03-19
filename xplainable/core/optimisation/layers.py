from subprocess import call
import numpy as np
import random
from tqdm.auto import trange
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score

class Evolve():

    def __init__(
        self,
        mutations=100,
        generations=50,
        max_generation_depth=10,
        max_severity=0.5,
        max_leaves=20,
        mutation_type='relative',
        reproduction_strategy='merge'):
        
        self.mutations = mutations
        self.generations = generations
        self.max_generation_depth = max_generation_depth
        self.max_severity = max_severity
        self.max_leaves = max_leaves
        self.mutation_type = mutation_type
        self.reproduction_strategy = reproduction_strategy

        self.objective = None

        self.xnetwork = None
        self.y = None

        self.generation_id = 1

    def _calculate_error(self, x):
        """ Calculates the error of a set of predictions.

        Args:
            x (numpy.array): An array of transformed values.
            y (numpy.array): An array of the true values.

        Returns:
            np.array: An array of the errors.
        """

        # calculate relative error
        err = (np.nansum(x, axis=1) + self.xnetwork.model.base_value) - self.y

        return err

    def _mae(self, x):
        
        mae = np.mean(abs(self._calculate_error(x)))

        return mae

    def _f1(self, x):
        
        pred = (np.nansum(x, axis=1) + self.xnetwork.model.base_value) > 0.5

        return f1_score(pred, self.y, average='macro')

    def _accuracy(self, x):
        
        pred = (np.nansum(x, axis=1) + self.xnetwork.model.base_value) > 0.5

        return accuracy_score(pred, self.y)

    def _score(self):
        pass

    def _mutate(self, chromosome):
        
        new_chromosome = chromosome.copy()
        chrome_length = chromosome.shape[0]

        # randomly select number of leaves to mutate
        num_leaves = random.sample(
            [i for i in range(1, int(min([self.max_leaves, chrome_length])+1))], k=1)[0]

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

    def _n_mutations(self, chromosome, n):

        mutations = []
        deltas = []
        for _ in range(n):
            mutation, delta = self._mutate(chromosome)
            mutations.append(mutation)
            deltas.append(delta)

        return np.array(mutations), np.array(deltas)

    def _mutate_transform(self, x, delta):

        x = x.copy()

        for i, v in delta.items():
            x[:,i][~np.isnan(x[:,i])] *= v
            
        return x

    def score_mutation(self, x, delta):

        x = x.copy()
        x = self._mutate_transform(x, delta)

        return self._score(x)

    def _merge_genes(self, pair):
        
        # get parent chromosomes
        a, b = pair
        
        child = (a + b) / 2

        return child

    def reproduce(self):
        pass

    def _get_delta(self, chromosome):

        return {
            i: v / self.target_chromosome[i] for i, v in enumerate(
                chromosome) if v != self.target_chromosome[i]
                }

    def _natural_selection(self, x, mutations, deltas):

        # Score mutations
        scores = np.array([])
        for delta in deltas:
            scores = np.append(scores, self.score_mutation(x, delta))

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

    def _reproduce_and_mutate(self, superiors):
        # Pair mates together
        mates = [[superiors[i], superiors[i+1]] for i in range(0, superiors.shape[0] - 1, 2)]
        
        # Generate child chromosomes
        offspring = [self.reproduce(mate) for mate in mates]
        
        # Add most superior parent to pool
        offspring.append(superiors[0])

        for child in offspring.copy():
            offspring.append(self._mutate(child)[0])

        # Extract deltas from offspring
        deltas = [self._get_delta(i) for i in offspring]

        return np.array(offspring), np.array(deltas)

    def _run_generation(self, x):
        
        # Reset the target error
        self.target_score = self._score(x)

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

    def _re_map(self, x):

        chromosome = np.nanmax(x, axis=0)

        for l, v in zip(self.xnetwork.leaves, chromosome):
            f, _id = l.split("_")
            self.xnetwork.model._profile[int(f)][int(_id)][2] = v

        return self.xnetwork.model._profile

    def _initialise(self, metric):

        # Infer objective from metric
        if metric == 'mae':
            self._score = self._mae
            self.objective = 'minimise'

        elif metric == 'f1':
            self._score = self._f1
            self.objective = 'maximise'

        elif metric == 'accuracy':
            self._score = self._accuracy
            self.objective = 'maximise'

        else:
            raise ValueError(f'Metric {metric} not supported')

        # Set reproduction strategy
        if self.reproduction_strategy == 'merge':
            self.reproduce = self._merge_genes

    def transform(self, xnetwork, x, y, callback=None):
        """ Optimises a feature score map with respect to the true values.

        Args:
            x (pandas.DataFrame): The input variables used for prediction.
            y (pandas.Series): The true values to fit to the x values.
            x_val ((pandas.DataFrame), optional): Validation x dataset.
            y_val ((pandas.Series), optional): Validation y dataset.
            verbose (bool, optional): Prints status if True. Defaults to False.
            plot (bool): Plots the live optimisation progress if True.
            plot_window (int, optional): The rolling average for plotting.

        Returns:
            dict: The optimised feature score map.
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

        self._initialise(self.xnetwork.metric)

        self.target_chromosome = xnetwork.root_chromosome
        self.target_score = self._score(
            self._mutate_transform(x, self._get_delta(xnetwork.root_chromosome)))

        #if self.xnetwork.layer_id == 1: print(f"Starting {xnetwork.metric}: ", self.target_score)
        
        with trange(self.generations) as pbar:
            for i in pbar:
                pbar.set_description(f'Layer {self.xnetwork.layer_id} ({self.xnetwork.layer_name})')
                gen = self._run_generation(x)
                delta = self._get_delta(gen)
                x = self._mutate_transform(x, delta)
                score = str(round(self._score(x), 4))
                self.generation_id += 1
                pbar.set_postfix(**{xnetwork.metric: score})

                #if callback:
                #    callback(self.xnetwork.layer_id, i+1, score, 0)

        self._re_map(x)

        if callback:
            callback(self.xnetwork.layer_id, i+1, score, 1)

        return x, self.target_chromosome


class Tighten:
    """ Optimises the feature score map for XRegressor Models.

        Args:
            model (xplainable.XRegressor): The regression estimator.
            iterations (int): The number of iterations to run.
            learning_rate (float): How fast the model learns. Between 0.001 - 1
            early_stopping (int): Stop early if no improvement after n iters.
            use_cython (bool): Use cython splitting function if True.
    """

    def __init__(self, iterations=100, learning_rate=0.03, early_stopping=None):

        # store params
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.early_stopping = early_stopping

        self.xnetwork = None
        
    def _calculate_error(self, x, y):
        """ Calculates the error of a set of predictions.

        Args:
            x (numpy.array): An array of transformed values.
            y (numpy.array): An array of the true values.

        Returns:
            np.array: An array of the errors.
        """

        # calculate relative error
        err = (np.nansum(x, axis=1) + self.xnetwork.model.base_value) - y

        return err

    def _next_best_change(self, errors):
        """ Identifies the most effective leaf node change to improve model.

        Args:
            errors (numpy.array): An array of errors for the current iteration.

        Returns:
            int: The index of the best leaf node.
            float: The max amount the score can change.
        """

        def get_change(idx):

            # if 'inc' yeilds higher benefit, change by 'um'
            if inc[idx] > dec[idx]:
                return um[idx]

            else:
                return om[idx] * -1

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
        bsts = np.maximum(inc, dec)

        # find individual best outcome and loc
        bst = np.nanmax(bsts)
        bstloc = np.where(bsts==bst)[0][0]

        # get value to update by
        change = get_change(bstloc) * self.learning_rate

        return bstloc, change

    def _run_iteration(self, x, y):
        """ Runs a single optimisation iteration.

        Args:
            x (numpy.array): An array of transformed values.
            y (numpy.array): An array of the true values.

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
        self.xnetwork.model._profile[int(f)][int(_id)][2] += change

        return x

    def transform(self, xnetwork, x, y, callback=None):
        """ Optimises a feature score map with respect to the true values.

        Args:
            x (pandas.DataFrame): The input variables used for prediction.
            y (pandas.Series): The true values to fit to the x values.
            callbacks (list): Callback function

        Returns:
            dict: The optimised feature score map.
        """
        
        self.xnetwork = xnetwork
        x = x.copy()

        starting_mae = np.mean(abs(self._calculate_error(x, y)))

        #if self.xnetwork.layer_id == 1:
        #print("Starting mae: ", starting_mae)

        # instantiate best df, best mae
        bst_mae = starting_mae
        best_x = x.copy()

        # iterations since best
        isb = 0
        
        # start optimisation process
        with trange(self.iterations) as pbar:

            for i in pbar:

                pbar.set_description(f'Layer {self.xnetwork.layer_id} ({self.xnetwork.layer_name})')

                # stop early if early stopping threshold reached
                if self.early_stopping:
                    if isb >= self.early_stopping:
                        # update callback
                        if callback:
                            callback(self.xnetwork.layer_id, i+1, bst_mae, 'stopped early')
                        print(f'Stopped early after {i} iterations.')
                        return best_x, np.nanmax(best_x, axis=0)

                # run iteration
                x = self._run_iteration(x, y)

                # calculate mean absolute error
                mae = np.mean(abs(self._calculate_error(x, y)))

                # update callback
                if callback is not None:
                    callback(self.xnetwork.layer_id, i+1, bst_mae, 0)

                # update output if best result
                if mae < bst_mae:
                    bst_mae = mae
                    best_x = x.copy()
                    isb = 0
                    pbar.set_postfix(mae=str(round(mae, 2)))

                else:
                    isb += 1

        # update callback
        if callback:
            callback(self.xnetwork.layer_id, i+1, bst_mae, 1)

        return best_x, np.nanmax(best_x, axis=0)