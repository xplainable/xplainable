"""
Smoke tests for xplainable core functionality.

These tests verify that the main APIs work end-to-end with small public datasets.
They are designed to run quickly (< 60s) and catch major regressions.
"""

import pandas as pd
import numpy as np
import pytest
import json
import os
from datetime import datetime
from sklearn.datasets import load_iris, load_diabetes, load_breast_cancer
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
import xplainable as xp


# Metrics tracking
METRICS_FILE = "tests/performance_metrics.json"

def load_metrics_history():
    """Load historical performance metrics."""
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_metrics(test_name, metrics):
    """Save performance metrics for a test."""
    history = load_metrics_history()
    
    if test_name not in history:
        history[test_name] = []
    
    entry = {
        "timestamp": datetime.now().isoformat(),
        "version": xp.__version__,
        "metrics": metrics
    }
    
    history[test_name].append(entry)
    
    # Keep only last 10 entries per test
    history[test_name] = history[test_name][-10:]
    
    os.makedirs(os.path.dirname(METRICS_FILE), exist_ok=True)
    with open(METRICS_FILE, 'w') as f:
        json.dump(history, f, indent=2)

def _basic_fit_predict_classifier(data_loader, max_time_seconds=30):
    """Helper function to test basic classifier workflow."""
    # Load data
    data = data_loader(as_frame=True)
    X, y = data.data, data.target
    
    # Ensure we have a reasonable amount of data but not too much for speed
    if len(X) > 300:
        X = X.sample(300, random_state=42)
        y = y.loc[X.index]
    
    # Create and fit model
    model = xp.XClassifier(max_depth=4, min_leaf_size=0.01)
    model.fit(X, y)
    
    # Make predictions
    test_X = X.head(10)
    predictions = model.predict(test_X)
    probabilities = model.predict_proba(test_X)
    scores = model.predict_score(test_X)
    
    # Basic assertions
    assert len(predictions) == len(test_X)
    assert len(probabilities) == len(test_X)
    assert len(scores) == len(test_X)
    
    # Check that probabilities are 2D array with correct shape
    if probabilities.ndim == 2:
        assert probabilities.shape[1] == len(np.unique(y))
    
    assert all(isinstance(p, (int, np.integer)) for p in predictions)
    assert all(0 <= s <= 1 for s in scores)
    
    # Calculate performance metrics on full dataset
    full_predictions = model.predict(X)
    full_probabilities = model.predict_proba(X)
    
    # Handle target mapping for metrics calculation
    y_for_metrics = y.copy()
    if hasattr(model.target_map, '__getitem__'):
        # Convert string targets to numeric using target_map
        if y.dtype == 'object':
            reverse_map = {v: k for k, v in model.target_map.items()}
            y_for_metrics = y.map(reverse_map)
    
    accuracy = accuracy_score(y_for_metrics, full_predictions)
    f1 = f1_score(y_for_metrics, full_predictions, average='weighted')
    
    metrics = {
        "accuracy": round(accuracy, 4),
        "f1_weighted": round(f1, 4),
        "n_samples": len(X),
        "n_features": len(X.columns)
    }
    
    return model, metrics


def _basic_fit_predict_regressor(data_loader, max_time_seconds=30):
    """Helper function to test basic regressor workflow."""
    # Load data
    data = data_loader(as_frame=True)
    X, y = data.data, data.target
    
    # Ensure we have a reasonable amount of data but not too much for speed
    if len(X) > 300:
        X = X.sample(300, random_state=42)
        y = y.loc[X.index]
    
    # Create and fit model
    model = xp.XRegressor(max_depth=4, min_leaf_size=0.01)
    model.fit(X, y)
    
    # Make predictions
    test_X = X.head(10)
    predictions = model.predict(test_X)
    
    # Basic assertions
    assert len(predictions) == len(test_X)
    assert all(isinstance(p, (float, np.floating)) for p in predictions)
    
    # Calculate performance metrics on full dataset
    full_predictions = model.predict(X)
    
    rmse = np.sqrt(mean_squared_error(y, full_predictions))
    r2 = r2_score(y, full_predictions)
    
    metrics = {
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "n_samples": len(X),
        "n_features": len(X.columns)
    }
    
    return model, metrics


def test_iris_classification():
    """Test classification on the iris dataset."""
    model, metrics = _basic_fit_predict_classifier(load_iris)
    
    # Test that model has expected attributes
    assert hasattr(model, 'columns')
    assert hasattr(model, 'target_map')
    assert len(model.columns) == 4  # iris has 4 features
    
    # Target map should exist and be a dict-like object
    assert model.target_map is not None
    assert hasattr(model.target_map, '__getitem__')  # dict-like access
    
    # Save metrics
    save_metrics("iris_classification", metrics)
    print(f"Iris Classification Metrics: {metrics}")


def test_breast_cancer_classification():
    """Test classification on the breast cancer dataset."""
    model, metrics = _basic_fit_predict_classifier(load_breast_cancer)
    
    # Test that model has expected attributes
    assert hasattr(model, 'columns')
    assert hasattr(model, 'target_map')
    
    # Target map should exist and be a dict-like object
    assert model.target_map is not None
    assert hasattr(model.target_map, '__getitem__')  # dict-like access
    
    # Save metrics
    save_metrics("breast_cancer_classification", metrics)
    print(f"Breast Cancer Classification Metrics: {metrics}")


def test_diabetes_regression():
    """Test regression on the diabetes dataset."""
    model, metrics = _basic_fit_predict_regressor(load_diabetes)
    
    # Test that model has expected attributes
    assert hasattr(model, 'columns')
    assert len(model.columns) == 10  # diabetes has 10 features
    
    # Save metrics
    save_metrics("diabetes_regression", metrics)
    print(f"Diabetes Regression Metrics: {metrics}")


def test_partitioned_classifier():
    """Test partitioned classifier functionality."""
    # Load data
    data = load_iris(as_frame=True)
    X, y = data.data, data.target
    
    # Add a partition column using proper numpy random API
    np.random.seed(42)
    X['partition'] = np.random.choice(['A', 'B'], size=len(X))
    
    # Create partitioned model
    partitioned_model = xp.PartitionedClassifier(partition_on='partition')
    
    # Create and fit individual models for each partition
    for partition in X['partition'].unique():
        part_data = X[X['partition'] == partition]
        part_X = part_data.drop('partition', axis=1)
        part_y = y.loc[part_data.index]
        
        # Create and fit model for this partition
        model = xp.XClassifier(max_depth=4, min_leaf_size=0.01)
        model.fit(part_X, part_y)
        
        # Add to partitioned model
        partitioned_model.add_partition(model, partition)
    
    # Add a default model for the __dataset__ partition (required for prediction)
    default_model = xp.XClassifier(max_depth=4, min_leaf_size=0.01)
    default_model.fit(X.drop('partition', axis=1), y)
    partitioned_model.add_partition(default_model, '__dataset__')
    
    # Test predictions
    test_X = X.head(10)
    predictions = partitioned_model.predict(test_X)
    
    # Basic assertions
    assert len(predictions) == len(test_X)
    assert hasattr(partitioned_model, 'partitions')
    assert len(partitioned_model.partitions) > 0
    
    # Calculate basic metrics for default model
    full_predictions = default_model.predict(X.drop('partition', axis=1))
    accuracy = accuracy_score(y, full_predictions)
    
    metrics = {
        "accuracy": round(accuracy, 4),
        "n_partitions": len(partitioned_model.partitions),
        "n_samples": len(X),
        "n_features": len(X.columns) - 1  # exclude partition column
    }
    
    save_metrics("partitioned_classifier", metrics)
    print(f"Partitioned Classifier Metrics: {metrics}")


def test_partitioned_regressor():
    """Test partitioned regressor functionality."""
    # Load data
    data = load_diabetes(as_frame=True)
    X, y = data.data, data.target
    
    # Add a partition column using proper numpy random API
    np.random.seed(42)
    X['partition'] = np.random.choice(['A', 'B'], size=len(X))
    
    # Create partitioned model
    partitioned_model = xp.PartitionedRegressor(partition_on='partition')
    
    # Create and fit individual models for each partition
    for partition in X['partition'].unique():
        part_data = X[X['partition'] == partition]
        part_X = part_data.drop('partition', axis=1)
        part_y = y.loc[part_data.index]
        
        # Create and fit model for this partition
        model = xp.XRegressor(max_depth=4, min_leaf_size=0.01)
        model.fit(part_X, part_y)
        
        # Add to partitioned model
        partitioned_model.add_partition(model, partition)
    
    # Test predictions
    test_X = X.head(10)
    predictions = partitioned_model.predict(test_X)
    
    # Basic assertions
    assert len(predictions) == len(test_X)
    assert hasattr(partitioned_model, 'partitions')
    assert len(partitioned_model.partitions) > 0
    
    metrics = {
        "n_partitions": len(partitioned_model.partitions),
        "n_samples": len(X),
        "n_features": len(X.columns) - 1  # exclude partition column
    }
    
    save_metrics("partitioned_regressor", metrics)
    print(f"Partitioned Regressor Metrics: {metrics}")


def test_model_evaluation():
    """Test model evaluation functionality."""
    # Load data
    data = load_iris(as_frame=True)
    X, y = data.data, data.target
    
    # Create and fit model
    model = xp.XClassifier(max_depth=4)
    model.fit(X, y)
    
    # Test evaluation
    evaluation = model.evaluate(X, y)
    
    # Check that evaluation returns expected metrics
    assert isinstance(evaluation, dict)
    
    # Check for key metrics that should be present
    # The actual structure may vary, so we check for common ones
    expected_metrics = ['classification_report', 'confusion_matrix', 'cohen_kappa']
    for metric in expected_metrics:
        assert metric in evaluation, f"Expected metric '{metric}' not found in evaluation results"
    
    # Check that classification report contains accuracy
    if 'classification_report' in evaluation:
        assert 'accuracy' in evaluation['classification_report']
        
        metrics = {
            "accuracy": round(evaluation['classification_report']['accuracy'], 4),
            "has_confusion_matrix": 'confusion_matrix' in evaluation,
            "has_cohen_kappa": 'cohen_kappa' in evaluation
        }
        
        save_metrics("model_evaluation", metrics)
        print(f"Model Evaluation Metrics: {metrics}")


def test_client_graceful_handling():
    """Test that client functionality is handled gracefully when not available."""
    # This should not raise an error even if xplainable-client is not installed
    assert hasattr(xp, 'Client')
    assert hasattr(xp, 'initialise')
    assert hasattr(xp, 'load_dataset')
    assert hasattr(xp, 'list_datasets')
    
    # The client should be None initially
    assert xp.client is None
    
    # No performance metrics for this test as it's just checking imports


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # -s to show print statements 