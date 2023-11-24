import pandas as pd
import networkx as nx
from tqdm.auto import tqdm
from..utils.dualdict import TargetMap

class GraphSelector:
    """ A feature selector that uses a graph network to select features.
        
    Args:
        method (str): The correlation method to use. Defaults to 'spearman'.
        min_target_corr (float): The minimum correlation between a feature
            and the target variable. Defaults to 0.02.
        min_feature_corr (float): The minimum correlation between two
            features. Defaults to 0.3.
    """
    
    def __init__(
            self, method='spearman',
            min_target_corr=0.02,
            min_feature_corr=0.3
            ):

        self.method = method
        self.min_target_corr = min_target_corr
        self.min_feature_corr = min_feature_corr
        
        self.matrix = pd.DataFrame()
        
        self.selected_features_ = None
        self.target_map = {}
        self.feature_map = {}
        
        self.dropped = []
        self.graphs = []
        
    @property
    def selected(self):
        return list(self.matrix.columns)
        
    def _encode_target(self, y):
        
        y = y.copy()

        # Cast as category
        target_ = y.astype('category')

        # Get the label map
        self.target_map = TargetMap(dict(enumerate(target_.cat.categories)), True)

        return
    
    def _encode_feature(self, X, y):
        """ Encodes features in order of their relationship with the target.

        Args:
            x (pandas.Series): The feature to encode.
            y (pandas.Series): The target feature.

        Returns:
            pd.Series: The encoded Series.
        """

        name = X.name
        x = X.copy()
        y = y.copy()

        if len(self.target_map) > 0:
            y = y.map(self.target_map)

        # Order categories by their relationship with the target variable
        ordered_values = pd.DataFrame(
            {'X': X, 'y': y}).groupby('X').agg({'y': 'mean'}).sort_values(
            'y', ascending=True).reset_index()

        # Encode feature
        feature_map = {val: i for i, val in enumerate(ordered_values['X'])}

        # Store map for transformation
        self.feature_map[name] = TargetMap(feature_map)

        return
    
    def _learn_encodings(self, X, y):
        
        if y.dtype == 'object': self._encode_target(y)

        for f in self.categorical_columns:
            self._encode_feature(X[f], y)

        return
    
    def _fetch_meta(self, x, y):
        # Assign target variable
        self.target = y.name

        # Store numeric column names
        self.numeric_columns = list(x.select_dtypes('number'))

        # Store categorical column names
        self.categorical_columns = list(x.select_dtypes('object'))

        self.columns = list(x.columns)
        
    def _encode(self, x, y=None):
        # Apply encoding
        for f, m in self.feature_map.items():
            x[f] = x.loc[:, f].map(m)

        if y is not None:
            if len(self.target_map) > 0:
                y = y.map(self.target_map)
                
            y = y.astype(float)
            return x, y
        
        return x
    
    def _build_network(self, min_corr):
        # Create an empty graph
        G = nx.Graph()

        # Add nodes to the graph
        G.add_nodes_from(self.matrix.columns)

        # Add edges to the graph based on correlation values
        for i, col1 in enumerate(self.matrix.columns):
            for j, col2 in enumerate(self.matrix.columns):
                if i < j and abs(self.matrix.loc[col1, col2]) >= min_corr:
                    G.add_edge(col1, col2)
                    
        return G
    
    def _all_values_zero(self, dic):

        if all([v == 0 for v in list(dic.values())]):
            return True
        else:
            return False
        
    def _drop_weakest(self, feature):
        self.matrix = self.matrix.drop(feature, axis=1)
        self.matrix = self.matrix.drop(feature)
    
    def fit(self, X, y, start_threshold=0.9):
        """ Fits the feature selector to the data.
        
        Args:
            X (pandas.DataFrame): The features.
            y (pandas.Series): The target variable.
            start_threshold (float): The starting threshold for feature
                selection. Defaults to 0.9.
            
        Returns:
            None
        """
        
        X = X.copy()
        y = y.copy()
        
        self._fetch_meta(X, y)
        self._learn_encodings(X, y)
        
        X, y = self._encode(X, y)
                
        dff = X.copy()
        dff['_target_'] = y
        self.matrix = dff.corr(method=self.method)
        
        self.target_correlations = self.matrix['_target_'][:-1]
        self.matrix = self.matrix.iloc[:-1, :-1]
        
        # filter features with low target correlation
        low_corr = list(self.matrix[
            abs(self.target_correlations) < self.min_target_corr].index)
        
        for lc in low_corr:
            corr = round(self.target_correlations[lc], 3)
            summary = {
                "feature": lc,
                "reason": f"low correlation with target ({corr})"
            }
            self.dropped.append(summary)
        
        self.matrix = self.matrix.drop(low_corr, axis=1)
        self.matrix = self.matrix.drop(low_corr, axis=0)
        
        pbar = tqdm(total=int((start_threshold-self.min_feature_corr)*100))
        threshold = start_threshold
        while True:
            
            if threshold < self.min_feature_corr:
                break
            
            G = self._build_network(threshold)

            # Get the degree of all nodes
            degree_dict = dict(G.degree())
            
            if len(degree_dict) == 0:
                break
            
            if self._all_values_zero(degree_dict):
                threshold = threshold - 0.01
                pbar.update(1)
                continue

            # Find the node(s) with the maximum degree
            max_degree = max(degree_dict.values())
            most_connected_nodes = [node for node, degree in \
                                    degree_dict.items() if degree == max_degree]
                            
            weakest = abs(self.target_correlations).loc[
                most_connected_nodes].sort_values(ascending=False).index[-1]
        
            self._drop_weakest(weakest)
            most_connected_nodes.remove(weakest)
            
            if len(most_connected_nodes) == 0:
                summary = {
                  "feature": weakest,
                  "reason": f"Central node in correlation cluster."
              }

            else:
                summary = {
                    "feature": weakest,
                    "reason": f"""high chance of multicollinearity with {most_connected_nodes}"""
                }

            self.dropped.append(summary)
            self.graphs.append(G)
        
        return self

    def plot_graph(self, animate=True, index=0):
        """ Plots the graph of the feature selection process.
        
        Args:
            animate (bool): Whether to animate the graph.
            index (int): The index of the graph to plot.

        Returns:
            plotly.Figure: The plotly figure.
        """
        
        from ..visualisation.network import plot_network_graphs

        return plot_network_graphs(self.graphs, animate, index)
