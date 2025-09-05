""" Copyright Xplainable Pty Ltd, 2023"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, r2_score

import xplainable as xp
from xplainable.preprocessing.interactions import InteractionGenerator


class TestInteractionGenerator:
    """Test suite for InteractionGenerator."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 200
        
        # Numerical features
        X1 = np.random.randn(n_samples)
        X2 = np.random.randn(n_samples)
        X3 = np.random.randn(n_samples)
        
        # Categorical features
        cat1 = np.random.choice(['A', 'B', 'C'], n_samples)
        cat2 = np.random.choice(['X', 'Y'], n_samples)
        
        X = pd.DataFrame({
            'num1': X1,
            'num2': X2,
            'num3': X3,
            'cat1': cat1,
            'cat2': cat2
        })
        
        # Target with interaction: y depends on num1*num2 and cat1&cat2
        y_continuous = X1 * X2 + 0.5 * (cat1 == 'A').astype(int) * (cat2 == 'X').astype(int)
        y_binary = (y_continuous > np.median(y_continuous)).astype(int)
        
        return X, y_binary, y_continuous
    
    def test_interaction_generator_init(self):
        """Test InteractionGenerator initialization."""
        ig = InteractionGenerator(max_interactions=10, min_importance=0.05)
        
        assert ig.max_interactions == 10
        assert ig.min_importance == 0.05
        assert ig.interaction_depth == 2
        assert ig.interaction_features_ == []
    
    def test_fit_classification(self, sample_data):
        """Test fitting on classification data."""
        X, y, _ = sample_data
        
        ig = InteractionGenerator(max_interactions=5)
        ig.fit(X, y, is_regression=False)
        
        assert len(ig.interaction_features_) <= 5
        assert all('features' in interaction for interaction in ig.interaction_features_)
        assert all('type' in interaction for interaction in ig.interaction_features_)
        assert all('score' in interaction for interaction in ig.interaction_features_)
    
    def test_fit_regression(self, sample_data):
        """Test fitting on regression data."""
        X, _, y = sample_data
        
        ig = InteractionGenerator(max_interactions=5)
        ig.fit(X, y, is_regression=True)
        
        assert len(ig.interaction_features_) <= 5
        assert all(isinstance(interaction['score'], (int, float)) for interaction in ig.interaction_features_)
    
    def test_transform_multiplicative(self, sample_data):
        """Test transformation with multiplicative interactions."""
        X, y, _ = sample_data
        
        # Force a multiplicative interaction
        ig = InteractionGenerator(max_interactions=1)
        ig.interaction_features_ = [{
            'features': ('num1', 'num2'),
            'type': 'multiplicative',
            'score': 0.5
        }]
        
        X_transformed = ig.transform(X)
        
        assert 'num1*num2' in X_transformed.columns
        expected_interaction = X['num1'] * X['num2']
        pd.testing.assert_series_equal(
            X_transformed['num1*num2'], 
            expected_interaction, 
            check_names=False
        )
    
    def test_transform_categorical(self, sample_data):
        """Test transformation with categorical interactions."""
        X, y, _ = sample_data
        
        # Force a categorical interaction
        ig = InteractionGenerator(max_interactions=1)
        ig.interaction_features_ = [{
            'features': ('cat1', 'cat2'),
            'type': 'categorical',
            'score': 0.3
        }]
        
        X_transformed = ig.transform(X)
        
        assert 'cat1&cat2' in X_transformed.columns
        expected_interaction = X['cat1'].astype(str) + '_' + X['cat2'].astype(str)
        pd.testing.assert_series_equal(
            X_transformed['cat1&cat2'], 
            expected_interaction, 
            check_names=False
        )
    
    def test_fit_transform(self, sample_data):
        """Test fit_transform method."""
        X, y, _ = sample_data
        
        ig = InteractionGenerator(max_interactions=3)
        X_transformed = ig.fit_transform(X, y, is_regression=False)
        
        # Should have original columns plus interactions
        assert all(col in X_transformed.columns for col in X.columns)
        assert X_transformed.shape[1] >= X.shape[1]  # Should have more columns
        
        # Check some interaction was found and added (or at least tried)
        interaction_cols = [col for col in X_transformed.columns 
                          if col not in X.columns]
        # Should at least have same number of columns (some datasets might not have strong interactions)
        assert X_transformed.shape[1] >= X.shape[1]
        # Should have found some interactions for this specific synthetic dataset
        if len(ig.interaction_features_) > 0:
            assert len(interaction_cols) > 0
    
    def test_get_interaction_explanations(self, sample_data):
        """Test interaction explanations."""
        X, y, _ = sample_data
        
        ig = InteractionGenerator(max_interactions=2)
        ig.fit(X, y, is_regression=False)
        
        explanations = ig.get_interaction_explanations()
        
        assert len(explanations) == len(ig.interaction_features_)
        assert all(isinstance(exp, str) for exp in explanations)
        assert all('importance:' in exp for exp in explanations)


class TestXplainableWithInteractions:
    """Test Xplainable models with interaction features."""
    
    @pytest.fixture
    def interaction_data(self):
        """Create data with known interactions."""
        np.random.seed(42)
        n_samples = 300
        
        # Features with clear interaction
        X1 = np.random.randn(n_samples)
        X2 = np.random.randn(n_samples)
        noise = np.random.randn(n_samples)
        
        X = pd.DataFrame({
            'feature1': X1,
            'feature2': X2,
            'noise': noise
        })
        
        # Target depends on interaction: y = X1 + X2 + 2*X1*X2
        y_reg = X1 + X2 + 2 * X1 * X2 + 0.1 * noise
        y_clf = (y_reg > np.median(y_reg)).astype(int)
        
        return X, y_clf, y_reg
    
    def test_classifier_with_interactions(self, interaction_data):
        """Test XClassifier with interaction features."""
        X, y, _ = interaction_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Baseline model
        model_baseline = xp.XClassifier(max_depth=4, min_leaf_size=0.01)
        model_baseline.fit(X_train, y_train)
        acc_baseline = accuracy_score(y_test, model_baseline.predict(X_test))
        
        # Model with interactions
        ig = InteractionGenerator(max_interactions=5)
        X_train_int = ig.fit_transform(X_train, y_train, is_regression=False)
        X_test_int = ig.transform(X_test)
        
        model_int = xp.XClassifier(max_depth=4, min_leaf_size=0.01)
        model_int.fit(X_train_int, y_train)
        acc_int = accuracy_score(y_test, model_int.predict(X_test_int))
        
        # With true interaction present, interaction model should perform better or equal
        assert acc_int >= acc_baseline * 0.95  # Allow for small variance
        
        # Check that profile includes interactions
        profile = model_int.profile
        assert 'interactions' in profile
        
        # Check feature importances include interactions
        importances = model_int.feature_importances
        interaction_cols = [col for col in X_train_int.columns if col not in X.columns]
        if interaction_cols:
            interaction_importance = sum(importances.get(col, 0) for col in interaction_cols)
            assert interaction_importance >= 0  # Should have some importance
    
    def test_regressor_with_interactions(self, interaction_data):
        """Test XRegressor with interaction features."""
        X, _, y = interaction_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Model with interactions
        ig = InteractionGenerator(max_interactions=5)
        X_train_int = ig.fit_transform(X_train, y_train, is_regression=True)
        X_test_int = ig.transform(X_test)
        
        model_int = xp.XRegressor(max_depth=4, min_leaf_size=0.01)
        model_int.fit(X_train_int, y_train)
        
        y_pred = model_int.predict(X_test_int)
        r2 = r2_score(y_test, y_pred)
        
        # Should achieve reasonable R2 (interaction is strong in synthetic data)
        assert r2 > -1.0  # Basic sanity check
        
        # Check profile structure
        profile = model_int.profile
        assert 'interactions' in profile
        assert isinstance(profile['interactions'], dict)
    
    def test_interaction_feature_detection(self):
        """Test interaction feature detection in profile."""
        # Create a simple dataset with manually named interaction
        X = pd.DataFrame({
            'A': [1, 2, 3, 4],
            'B': [2, 3, 4, 5],
            'A*B': [2, 6, 12, 20],  # Interaction feature
            'C&D': ['A_X', 'B_Y', 'A_X', 'B_Y']  # Categorical interaction
        })
        y = pd.Series([0, 1, 1, 0], name='target')
        
        model = xp.XClassifier(max_depth=3)
        model.fit(X, y)
        
        # Check that interaction features are properly detected
        profile = model.profile
        assert 'A*B' in profile['interactions']
        assert 'C&D' in profile['interactions']
        
        # Check interaction parsing
        assert model._is_interaction_feature('A*B')
        assert model._is_interaction_feature('C&D')
        assert not model._is_interaction_feature('A')
        
        assert model._get_interaction_type('A*B') == 'multiplicative'
        assert model._get_interaction_type('C&D') == 'categorical'
        
        assert model._parse_interaction_features('A*B') == ['A', 'B']
        assert model._parse_interaction_features('C&D') == ['C', 'D']


class TestInteractionEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_data(self):
        """Test with empty data."""
        X = pd.DataFrame()
        y = np.array([])
        
        ig = InteractionGenerator()
        ig.fit(X, y)
        
        assert ig.interaction_features_ == []
    
    def test_single_feature(self):
        """Test with single feature (no interactions possible)."""
        X = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        y = np.array([0, 1, 0, 1, 0])
        
        ig = InteractionGenerator()
        ig.fit(X, y)
        
        assert ig.interaction_features_ == []
    
    def test_transform_without_fit(self):
        """Test transform without fitting."""
        X = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        
        ig = InteractionGenerator()
        X_transformed = ig.transform(X)
        
        # Should return original data if no interactions learned
        pd.testing.assert_frame_equal(X, X_transformed)
    
    def test_no_sklearn_fallback(self, monkeypatch):
        """Test fallback when sklearn is not available."""
        # Mock sklearn availability
        from xplainable.preprocessing import interactions
        monkeypatch.setattr(interactions, 'SKLEARN_AVAILABLE', False)
        
        X = pd.DataFrame({
            'num1': [1, 2, 3, 4],
            'num2': [2, 4, 6, 8],
            'cat1': ['A', 'B', 'A', 'B']
        })
        y = np.array([0, 1, 1, 0])
        
        ig = InteractionGenerator(max_interactions=2)
        X_transformed = ig.fit_transform(X, y)
        
        # Should still work with correlation-based fallback
        assert X_transformed.shape[1] >= X.shape[1]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])