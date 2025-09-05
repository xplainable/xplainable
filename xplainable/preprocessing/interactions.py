""" Copyright Xplainable Pty Ltd, 2023"""

import numpy as np
import pandas as pd
from itertools import combinations
from typing import Union, List, Tuple, Optional

# Optional import for feature selection
try:
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class InteractionGenerator:
    """
    Generate feature interactions for Xplainable models while maintaining explainability.
    
    Example:
        >>> ig = InteractionGenerator(max_interactions=20)
        >>> X_with_interactions = ig.fit_transform(X, y)
        >>> model = XClassifier()
        >>> model.fit(X_with_interactions, y)
    """
    
    def __init__(
        self, 
        max_interactions: int = 50,
        interaction_types: List[str] = ['multiplicative', 'categorical'],
        min_importance: float = 0.01,
        interaction_depth: int = 2
    ):
        """
        Initialize interaction generator.
        
        Args:
            max_interactions: Maximum number of interaction terms to generate
            interaction_types: Types of interactions to create
            min_importance: Minimum importance score for interaction to be included
            interaction_depth: Maximum number of features in one interaction (2 = pairwise)
        """
        self.max_interactions = max_interactions
        self.interaction_types = interaction_types
        self.min_importance = min_importance
        self.interaction_depth = interaction_depth
        self.interaction_features_ = []
        self.interaction_map_ = {}
        
    def fit(self, X: pd.DataFrame, y: np.ndarray, is_regression: bool = False):
        """
        Identify important feature interactions.
        
        Args:
            X: Input features
            y: Target variable
            is_regression: Whether this is a regression problem
        """
        self.interaction_features_ = []
        self.interaction_map_ = {}
        
        # 1. Calculate individual feature importances
        feature_importances = self._calculate_feature_importance(X, y, is_regression)
        
        # 2. Generate candidate interactions
        candidates = self._generate_candidates(X, feature_importances)
        
        # 3. Score interactions
        interaction_scores = self._score_interactions(X, y, candidates, is_regression)
        
        # 4. Select top interactions
        self.interaction_features_ = self._select_top_interactions(
            interaction_scores, self.max_interactions
        )
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add interaction features to dataset.
        
        Args:
            X: Input features
            
        Returns:
            DataFrame with original and interaction features
        """
        X_new = X.copy()
        
        for interaction in self.interaction_features_:
            feat1, feat2, int_type = interaction['features'][0], interaction['features'][1], interaction['type']
            
            if int_type == 'multiplicative':
                # Numerical * Numerical
                X_new[f'{feat1}*{feat2}'] = X[feat1] * X[feat2]
                
            elif int_type == 'categorical':
                # Categorical & Categorical
                X_new[f'{feat1}&{feat2}'] = X[feat1].astype(str) + '_' + X[feat2].astype(str)
                
            elif int_type == 'mixed':
                # Categorical conditions on numerical
                for cat_val in X[feat1].unique():
                    mask = X[feat1] == cat_val
                    col_name = f'{feat2}_when_{feat1}={cat_val}'
                    X_new[col_name] = X[feat2].where(mask, 0)
        
        return X_new
    
    def fit_transform(self, X: pd.DataFrame, y: np.ndarray, is_regression: bool = False) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(X, y, is_regression)
        return self.transform(X)
    
    def _calculate_feature_importance(
        self, X: pd.DataFrame, y: np.ndarray, is_regression: bool
    ) -> dict:
        """Calculate importance of individual features."""
        importances = {}
        
        for col in X.columns:
            try:
                if SKLEARN_AVAILABLE:
                    X_col = X[col].values.reshape(-1, 1)
                    if is_regression:
                        score = mutual_info_regression(X_col, y, random_state=42)[0]
                    else:
                        score = mutual_info_classif(X_col, y, random_state=42)[0]
                    importances[col] = score
                else:
                    # Fallback: use correlation-based importance
                    if pd.api.types.is_numeric_dtype(X[col]):
                        corr = np.abs(np.corrcoef(X[col], y)[0, 1])
                        importances[col] = corr if not np.isnan(corr) else 0.0
                    else:
                        # For categorical, use basic entropy-based measure
                        importances[col] = 0.1
            except:
                importances[col] = 0.0
                
        # Normalize
        max_imp = max(importances.values()) if importances else 1.0
        if max_imp > 0:
            importances = {k: v/max_imp for k, v in importances.items()}
            
        return importances
    
    def _generate_candidates(
        self, X: pd.DataFrame, feature_importances: dict
    ) -> List[Tuple[str, str, str]]:
        """Generate candidate interaction pairs."""
        candidates = []
        
        # Sort features by importance
        sorted_features = sorted(
            feature_importances.items(), key=lambda x: x[1], reverse=True
        )
        
        # Only consider top features for interactions
        top_features = [f for f, imp in sorted_features[:20] if imp > self.min_importance]
        
        # Generate pairwise combinations
        for feat1, feat2 in combinations(top_features, 2):
            # Determine interaction type
            is_num1 = pd.api.types.is_numeric_dtype(X[feat1])
            is_num2 = pd.api.types.is_numeric_dtype(X[feat2])
            
            if is_num1 and is_num2:
                candidates.append((feat1, feat2, 'multiplicative'))
            elif not is_num1 and not is_num2:
                # Only if cardinality is reasonable
                if X[feat1].nunique() * X[feat2].nunique() < 100:
                    candidates.append((feat1, feat2, 'categorical'))
            else:
                candidates.append((feat1, feat2, 'mixed'))
                
        return candidates
    
    def _score_interactions(
        self, X: pd.DataFrame, y: np.ndarray, 
        candidates: List[Tuple[str, str, str]], is_regression: bool
    ) -> List[dict]:
        """Score candidate interactions."""
        scores = []
        
        for feat1, feat2, int_type in candidates:
            # Create interaction feature
            if int_type == 'multiplicative':
                X_int = (X[feat1] * X[feat2]).values.reshape(-1, 1)
            elif int_type == 'categorical':
                X_int = (X[feat1].astype(str) + '_' + X[feat2].astype(str))
                X_int = pd.get_dummies(X_int).values
            else:  # mixed
                X_int = X[[feat1, feat2]].values
            
            try:
                # Calculate interaction importance
                if SKLEARN_AVAILABLE:
                    if is_regression:
                        score = mutual_info_regression(X_int, y, random_state=42)[0]
                    else:
                        score = mutual_info_classif(X_int, y, random_state=42)[0]
                else:
                    # Fallback: use correlation-based scoring
                    if int_type == 'multiplicative':
                        interaction_vals = X[feat1] * X[feat2]
                        score = abs(np.corrcoef(interaction_vals, y)[0, 1])
                        score = score if not np.isnan(score) else 0.0
                    else:
                        score = 0.1  # Default score for categorical interactions
                    
                scores.append({
                    'features': (feat1, feat2),
                    'type': int_type,
                    'score': score
                })
            except:
                continue
                
        return scores
    
    def _select_top_interactions(
        self, interaction_scores: List[dict], max_interactions: int
    ) -> List[dict]:
        """Select top scoring interactions."""
        # Sort by score
        sorted_interactions = sorted(
            interaction_scores, key=lambda x: x['score'], reverse=True
        )
        
        # Remove redundant interactions
        selected = []
        feature_pairs_used = set()
        
        for interaction in sorted_interactions:
            feat_pair = frozenset(interaction['features'])
            if feat_pair not in feature_pairs_used:
                selected.append(interaction)
                feature_pairs_used.add(feat_pair)
                
                if len(selected) >= max_interactions:
                    break
                    
        return selected
    
    def get_interaction_explanations(self) -> List[str]:
        """Get human-readable explanations of interactions."""
        explanations = []
        
        for interaction in self.interaction_features_:
            feat1, feat2 = interaction['features']
            int_type = interaction['type']
            score = interaction['score']
            
            if int_type == 'multiplicative':
                exp = f"{feat1} × {feat2}: Combined effect of {feat1} and {feat2}"
            elif int_type == 'categorical':
                exp = f"{feat1} & {feat2}: Joint categories of {feat1} and {feat2}"
            else:
                exp = f"{feat2} | {feat1}: {feat2} conditioned on {feat1} values"
                
            explanations.append(f"{exp} (importance: {score:.3f})")
            
        return explanations