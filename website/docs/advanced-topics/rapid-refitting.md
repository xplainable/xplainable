---
sidebar_position: 2
---

# Rapid Refitting

:::info Lightning-Fast Model Updates
**Rapid refitting** allows you to update model parameters in milliseconds without retraining from scratch. Perfect for real-time optimization, A/B testing, and parameter tuning scenarios.
:::

## Overview

Rapid refitting is a unique feature of xplainable models that enables instant parameter updates without full retraining. Unlike traditional machine learning models that require complete retraining when parameters change, xplainable models can adjust their activation functions and decision boundaries in real-time.

### Key Benefits

<div className="row">
  <div className="col col--6">
    <div className="card">
      <div className="card__header">
        <h3>⚡ Instant Updates</h3>
      </div>
      <div className="card__body">
        <p>Update model parameters in milliseconds, not minutes or hours.</p>
      </div>
    </div>
  </div>
  <div className="col col--6">
    <div className="card">
      <div className="card__header">
        <h3>🎯 Real-Time Optimization</h3>
      </div>
      <div className="card__body">
        <p>Continuously optimize model performance as new data arrives.</p>
      </div>
    </div>
  </div>
</div>

<div className="row">
  <div className="col col--6">
    <div className="card">
      <div className="card__header">
        <h3>🔬 A/B Testing</h3>
      </div>
      <div className="card__body">
        <p>Test different parameter configurations instantly in production.</p>
      </div>
    </div>
  </div>
  <div className="col col--6">
    <div className="card">
      <div className="card__header">
        <h3>💡 Interactive Tuning</h3>
      </div>
      <div className="card__body">
        <p>Experiment with parameters and see results immediately.</p>
      </div>
    </div>
  </div>
</div>

## How Rapid Refitting Works

Rapid refitting works by separating the **tree structure** (which requires full training) from the **activation function** (which can be updated instantly):

1. **Tree Structure**: The decision tree ensemble is built once during initial training
2. **Activation Function**: Parameters like `weight`, `power_degree`, and `sigmoid_exponent` control how leaf values are combined
3. **Instant Updates**: Changing activation parameters recalculates predictions without rebuilding trees

:::tip Understanding the Architecture
- **Tree Building**: Computationally expensive, done once during `fit()`
- **Activation Function**: Lightweight mathematical transformation applied to leaf values
- **Rapid Refitting**: Updates only the activation function, keeping trees intact
:::

## Basic Usage

### Simple Parameter Update

```python
from xplainable.core.models import XClassifier
import pandas as pd
from sklearn.model_selection import train_test_split

# Load and prepare data
data = pd.read_csv('data.csv')
X = data.drop('target', axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train initial model
model = XClassifier(
    max_depth=5,
    min_info_gain=0.01,
    weight=0.5,
    power_degree=1,
    sigmoid_exponent=1
)
model.fit(X_train, y_train)

# Initial performance
initial_accuracy = model.score(X_test, y_test)
print(f"Initial accuracy: {initial_accuracy:.3f}")

# Rapid refit with new parameters
model.refit(
    weight=0.8,
    power_degree=2,
    sigmoid_exponent=1.5
)

# New performance (calculated instantly)
new_accuracy = model.score(X_test, y_test)
print(f"New accuracy: {new_accuracy:.3f}")
print(f"Improvement: {new_accuracy - initial_accuracy:.3f}")
```

### Parameter Exploration

```python
# Test different parameter combinations
parameter_combinations = [
    {'weight': 0.3, 'power_degree': 1, 'sigmoid_exponent': 1},
    {'weight': 0.5, 'power_degree': 1.5, 'sigmoid_exponent': 1},
    {'weight': 0.7, 'power_degree': 2, 'sigmoid_exponent': 1.2},
    {'weight': 0.9, 'power_degree': 2.5, 'sigmoid_exponent': 1.5},
]

results = []
for i, params in enumerate(parameter_combinations):
    # Rapid refit with new parameters
    model.refit(**params)
    
    # Evaluate performance
    accuracy = model.score(X_test, y_test)
    results.append({
        'combination': i + 1,
        'params': params,
        'accuracy': accuracy
    })
    
    print(f"Combination {i+1}: {accuracy:.3f} - {params}")

# Find best parameters
best_result = max(results, key=lambda x: x['accuracy'])
print(f"\nBest parameters: {best_result['params']}")
print(f"Best accuracy: {best_result['accuracy']:.3f}")

# Apply best parameters
model.refit(**best_result['params'])
```

## Advanced Techniques

### Real-Time Optimization with Scipy

```python
import numpy as np
from scipy.optimize import minimize

def objective_function(params, model, X_val, y_val):
    """Objective function for optimization."""
    weight, power_degree, sigmoid_exponent = params
    
    # Rapid refit with new parameters
    model.refit(
        weight=weight,
        power_degree=power_degree,
        sigmoid_exponent=sigmoid_exponent
    )
    
    # Return negative accuracy (minimize)
    accuracy = model.score(X_val, y_val)
    return -accuracy

# Split training data for validation
X_train_opt, X_val, y_train_opt, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Train model on optimization set
model = XClassifier(max_depth=5, min_info_gain=0.01)
model.fit(X_train_opt, y_train_opt)

# Optimize parameters using rapid refitting
initial_params = [0.5, 1.0, 1.0]  # weight, power_degree, sigmoid_exponent
bounds = [(0.1, 1.0), (0.5, 3.0), (0.5, 2.0)]

print("Starting optimization...")
result = minimize(
    objective_function,
    initial_params,
    args=(model, X_val, y_val),
    bounds=bounds,
    method='L-BFGS-B'
)

# Apply optimized parameters
optimal_weight, optimal_power, optimal_sigmoid = result.x
model.refit(
    weight=optimal_weight,
    power_degree=optimal_power,
    sigmoid_exponent=optimal_sigmoid
)

print(f"Optimized parameters:")
print(f"  Weight: {optimal_weight:.3f}")
print(f"  Power degree: {optimal_power:.3f}")
print(f"  Sigmoid exponent: {optimal_sigmoid:.3f}")
print(f"  Final accuracy: {-result.fun:.3f}")
```

### Bayesian Optimization Integration

```python
from xplainable.core.optimisation import XParamOptimiser

# Create parameter optimizer with rapid refitting focus
optimizer = XParamOptimiser(
    model=XClassifier(),
    X=X_train,
    y=y_train,
    metric='roc_auc',
    cv=5,
    n_iter=50,
    random_state=42
)

# Define parameter space focused on rapid refitting parameters
param_space = {
    'weight': [0.1, 1.0],
    'power_degree': [0.5, 3.0],
    'sigmoid_exponent': [0.5, 2.0],
    # Include some tree parameters for comparison
    'max_depth': [3, 8],
    'min_info_gain': [0.001, 0.1]
}

# Optimize parameters
print("Running Bayesian optimization...")
best_params = optimizer.optimise(param_space)

# Train model with best parameters
model = XClassifier(**best_params)
model.fit(X_train, y_train)

print(f"Best parameters: {best_params}")
print(f"Best CV score: {optimizer.best_score:.3f}")
print(f"Test accuracy: {model.score(X_test, y_test):.3f}")

# Now use rapid refitting to fine-tune activation parameters
rapid_refit_params = {
    'weight': best_params['weight'],
    'power_degree': best_params['power_degree'],
    'sigmoid_exponent': best_params['sigmoid_exponent']
}

# Fine-tune with grid search
weight_values = np.linspace(max(0.1, best_params['weight'] - 0.2), 
                           min(1.0, best_params['weight'] + 0.2), 5)
power_values = np.linspace(max(0.5, best_params['power_degree'] - 0.5), 
                          min(3.0, best_params['power_degree'] + 0.5), 5)

best_score = 0
best_fine_tune_params = rapid_refit_params.copy()

for weight in weight_values:
    for power in power_values:
        model.refit(weight=weight, power_degree=power, 
                   sigmoid_exponent=best_params['sigmoid_exponent'])
        score = model.score(X_test, y_test)
        
        if score > best_score:
            best_score = score
            best_fine_tune_params = {
                'weight': weight,
                'power_degree': power,
                'sigmoid_exponent': best_params['sigmoid_exponent']
            }

print(f"\nFine-tuned parameters: {best_fine_tune_params}")
print(f"Fine-tuned accuracy: {best_score:.3f}")
```

## Regression-Specific Rapid Refitting

### Basic Regression Example

```python
from xplainable.core.models import XRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Train regression model
regressor = XRegressor(
    max_depth=5,
    min_info_gain=0.01,
    weight=0.5,
    power_degree=1,
    prediction_range=[0, 100]  # Regression-specific parameter
)
regressor.fit(X_train, y_train)

# Test different parameter combinations
parameter_grid = {
    'weight': [0.3, 0.5, 0.7, 0.9],
    'power_degree': [1, 1.5, 2, 2.5],
    'prediction_range': [[0, 100], [0, 200], [-50, 150]]
}

best_r2 = -np.inf
best_params = {}

print("Testing regression parameter combinations...")
for weight in parameter_grid['weight']:
    for power_degree in parameter_grid['power_degree']:
        for pred_range in parameter_grid['prediction_range']:
            # Rapid refit
            regressor.refit(
                weight=weight,
                power_degree=power_degree,
                prediction_range=pred_range
            )
            
            # Evaluate
            predictions = regressor.predict(X_test)
            r2 = r2_score(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            
            if r2 > best_r2:
                best_r2 = r2
                best_params = {
                    'weight': weight,
                    'power_degree': power_degree,
                    'prediction_range': pred_range,
                    'r2': r2,
                    'mse': mse
                }

print(f"\nBest R² score: {best_r2:.3f}")
print(f"Best parameters: {best_params}")

# Apply best parameters
regressor.refit(**{k: v for k, v in best_params.items() if k not in ['r2', 'mse']})
```

### Advanced Regression Optimization

```python
def optimize_regression_parameters(regressor, X_train, y_train, X_val, y_val):
    """Advanced regression parameter optimization."""
    
    # Define parameter ranges based on data characteristics
    y_range = y_train.max() - y_train.min()
    y_mean = y_train.mean()
    y_std = y_train.std()
    
    # Adaptive parameter ranges
    prediction_ranges = [
        [y_train.min() - y_std, y_train.max() + y_std],
        [y_train.min() - 2*y_std, y_train.max() + 2*y_std],
        [y_mean - 3*y_std, y_mean + 3*y_std]
    ]
    
    optimization_results = []
    
    # Grid search with adaptive ranges
    for weight in np.linspace(0.1, 1.0, 10):
        for power_degree in np.linspace(0.5, 3.0, 10):
            for pred_range in prediction_ranges:
                regressor.refit(
                    weight=weight,
                    power_degree=power_degree,
                    prediction_range=pred_range
                )
                
                # Evaluate on validation set
                val_predictions = regressor.predict(X_val)
                val_r2 = r2_score(y_val, val_predictions)
                val_mse = mean_squared_error(y_val, val_predictions)
                
                optimization_results.append({
                    'weight': weight,
                    'power_degree': power_degree,
                    'prediction_range': pred_range,
                    'val_r2': val_r2,
                    'val_mse': val_mse
                })
    
    # Find best parameters
    best_result = max(optimization_results, key=lambda x: x['val_r2'])
    
    return best_result, optimization_results

# Usage
best_params, all_results = optimize_regression_parameters(
    regressor, X_train, y_train, X_val, y_val
)

print(f"Best regression parameters:")
print(f"  Weight: {best_params['weight']:.3f}")
print(f"  Power degree: {best_params['power_degree']:.3f}")
print(f"  Prediction range: {best_params['prediction_range']}")
print(f"  Validation R²: {best_params['val_r2']:.3f}")
```

## Production Use Cases

### A/B Testing Framework

```python
class ABTestingFramework:
    def __init__(self, base_model):
        self.base_model = base_model
        self.variants = {}
        self.performance_metrics = {}
    
    def create_variant(self, variant_name, params):
        """Create a new variant with specific parameters."""
        # Clone the base model structure (trees remain the same)
        variant = self.base_model
        variant.refit(**params)
        
        self.variants[variant_name] = {
            'model': variant,
            'params': params,
            'predictions': 0,
            'performance': []
        }
        
        print(f"Created variant '{variant_name}' with params: {params}")
    
    def predict(self, X, variant_name='control'):
        """Make predictions with a specific variant."""
        if variant_name == 'control':
            predictions = self.base_model.predict(X)
        else:
            # Apply variant parameters
            variant_params = self.variants[variant_name]['params']
            self.base_model.refit(**variant_params)
            predictions = self.base_model.predict(X)
            self.variants[variant_name]['predictions'] += len(X)
        
        return predictions
    
    def update_performance(self, variant_name, metric_value):
        """Update performance metrics for a variant."""
        if variant_name == 'control':
            if 'control' not in self.performance_metrics:
                self.performance_metrics['control'] = []
            self.performance_metrics['control'].append(metric_value)
        elif variant_name in self.variants:
            self.variants[variant_name]['performance'].append(metric_value)
    
    def get_performance_summary(self):
        """Get performance summary for all variants."""
        summary = {}
        
        # Control performance
        if 'control' in self.performance_metrics:
            control_perf = self.performance_metrics['control']
            summary['control'] = {
                'mean_performance': np.mean(control_perf),
                'std_performance': np.std(control_perf),
                'sample_count': len(control_perf)
            }
        
        # Variant performance
        for variant_name, variant_data in self.variants.items():
            if variant_data['performance']:
                summary[variant_name] = {
                    'mean_performance': np.mean(variant_data['performance']),
                    'std_performance': np.std(variant_data['performance']),
                    'sample_count': len(variant_data['performance']),
                    'prediction_count': variant_data['predictions']
                }
        
        return summary
    
    def get_best_variant(self):
        """Get the best performing variant."""
        summary = self.get_performance_summary()
        
        best_variant = 'control'
        best_performance = summary.get('control', {}).get('mean_performance', 0)
        
        for variant_name, metrics in summary.items():
            if variant_name != 'control' and metrics['mean_performance'] > best_performance:
                best_performance = metrics['mean_performance']
                best_variant = variant_name
        
        return best_variant, best_performance

# Usage example
ab_framework = ABTestingFramework(model)

# Create variants
ab_framework.create_variant('variant_a', {'weight': 0.7, 'power_degree': 1.5})
ab_framework.create_variant('variant_b', {'weight': 0.9, 'power_degree': 2.0})
ab_framework.create_variant('variant_c', {'weight': 0.5, 'power_degree': 2.5})

# Simulate A/B testing
import random

for i in range(100):  # 100 test rounds
    # Randomly select variant
    variant = random.choice(['control', 'variant_a', 'variant_b', 'variant_c'])
    
    # Make prediction (using a sample from test set)
    sample_idx = random.randint(0, len(X_test) - 1)
    X_sample = X_test.iloc[[sample_idx]]
    y_sample = y_test.iloc[sample_idx]
    
    prediction = ab_framework.predict(X_sample, variant)
    
    # Calculate performance (accuracy for this sample)
    performance = 1 if prediction[0] == y_sample else 0
    ab_framework.update_performance(variant, performance)

# Get results
summary = ab_framework.get_performance_summary()
best_variant, best_performance = ab_framework.get_best_variant()

print("\nA/B Testing Results:")
for variant, metrics in summary.items():
    print(f"{variant}: {metrics['mean_performance']:.3f} ± {metrics['std_performance']:.3f} "
          f"(n={metrics['sample_count']})")

print(f"\nBest variant: {best_variant} with performance {best_performance:.3f}")
```

### Online Learning Simulation

```python
class OnlineLearningSimulator:
    def __init__(self, model, learning_rate=0.01):
        self.model = model
        self.learning_rate = learning_rate
        self.performance_history = []
        self.parameter_history = []
    
    def update_parameters(self, X_batch, y_batch):
        """Update parameters based on batch performance."""
        # Get current parameters
        current_params = {
            'weight': getattr(self.model, 'weight', 0.5),
            'power_degree': getattr(self.model, 'power_degree', 1.0),
            'sigmoid_exponent': getattr(self.model, 'sigmoid_exponent', 1.0)
        }
        
        # Test parameter adjustments
        best_accuracy = self.model.score(X_batch, y_batch)
        best_params = current_params.copy()
        
        # Simple gradient-free optimization
        for param_name in ['weight', 'power_degree', 'sigmoid_exponent']:
            # Try increasing parameter
            test_params = current_params.copy()
            test_params[param_name] += self.learning_rate
            
            # Apply bounds
            if param_name == 'weight' and test_params[param_name] > 1.0:
                test_params[param_name] = 1.0
            
            self.model.refit(**test_params)
            test_accuracy = self.model.score(X_batch, y_batch)
            
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_params = test_params.copy()
            
            # Try decreasing parameter
            test_params = current_params.copy()
            test_params[param_name] -= self.learning_rate
            
            if test_params[param_name] < 0.1:
                test_params[param_name] = 0.1
            
            self.model.refit(**test_params)
            test_accuracy = self.model.score(X_batch, y_batch)
            
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_params = test_params.copy()
        
        # Apply best parameters
        self.model.refit(**best_params)
        
        # Record history
        self.performance_history.append(best_accuracy)
        self.parameter_history.append(best_params.copy())
    
    def simulate_online_learning(self, X, y, batch_size=100):
        """Simulate online learning with streaming data."""
        n_samples = len(X)
        
        print(f"Starting online learning simulation with {n_samples} samples...")
        
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            X_batch = X.iloc[i:end_idx]
            y_batch = y.iloc[i:end_idx]
            
            self.update_parameters(X_batch, y_batch)
            
            batch_num = i // batch_size + 1
            print(f"Batch {batch_num}: Accuracy = {self.performance_history[-1]:.3f}")
        
        return self.performance_history, self.parameter_history

# Usage
online_simulator = OnlineLearningSimulator(model, learning_rate=0.05)

# Simulate online learning
performance_history, parameter_history = online_simulator.simulate_online_learning(
    X_train, y_train, batch_size=100
)

# Plot results
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

# Performance over time
plt.subplot(1, 2, 1)
plt.plot(performance_history)
plt.title('Performance Over Time')
plt.xlabel('Batch')
plt.ylabel('Accuracy')
plt.grid(True)

# Parameter evolution
plt.subplot(1, 2, 2)
weights = [p['weight'] for p in parameter_history]
powers = [p['power_degree'] for p in parameter_history]
plt.plot(weights, label='Weight')
plt.plot(powers, label='Power Degree')
plt.title('Parameter Evolution')
plt.xlabel('Batch')
plt.ylabel('Parameter Value')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"\nFinal parameters:")
print(f"  Weight: {parameter_history[-1]['weight']:.3f}")
print(f"  Power degree: {parameter_history[-1]['power_degree']:.3f}")
print(f"  Sigmoid exponent: {parameter_history[-1]['sigmoid_exponent']:.3f}")
```

## Parameter Sensitivity Analysis

### Comprehensive Sensitivity Analysis

```python
def analyze_parameter_sensitivity(model, X_test, y_test, param_name, param_range):
    """Analyze sensitivity of model performance to parameter changes."""
    original_params = {
        'weight': getattr(model, 'weight', 0.5),
        'power_degree': getattr(model, 'power_degree', 1.0),
        'sigmoid_exponent': getattr(model, 'sigmoid_exponent', 1.0)
    }
    
    param_values = []
    accuracies = []
    
    for param_value in param_range:
        # Update parameter
        test_params = original_params.copy()
        test_params[param_name] = param_value
        
        # Rapid refit
        model.refit(**test_params)
        
        # Evaluate
        accuracy = model.score(X_test, y_test)
        
        param_values.append(param_value)
        accuracies.append(accuracy)
    
    # Restore original parameters
    model.refit(**original_params)
    
    return param_values, accuracies

# Analyze sensitivity for each parameter
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Weight sensitivity
weight_range = np.linspace(0.1, 1.0, 20)
weight_values, weight_accuracies = analyze_parameter_sensitivity(
    model, X_test, y_test, 'weight', weight_range
)
axes[0].plot(weight_values, weight_accuracies, 'b-', linewidth=2)
axes[0].set_title('Weight Sensitivity')
axes[0].set_xlabel('Weight')
axes[0].set_ylabel('Accuracy')
axes[0].grid(True)

# Power degree sensitivity
power_range = np.linspace(0.5, 3.0, 20)
power_values, power_accuracies = analyze_parameter_sensitivity(
    model, X_test, y_test, 'power_degree', power_range
)
axes[1].plot(power_values, power_accuracies, 'r-', linewidth=2)
axes[1].set_title('Power Degree Sensitivity')
axes[1].set_xlabel('Power Degree')
axes[1].set_ylabel('Accuracy')
axes[1].grid(True)

# Sigmoid exponent sensitivity
sigmoid_range = np.linspace(0.5, 2.0, 20)
sigmoid_values, sigmoid_accuracies = analyze_parameter_sensitivity(
    model, X_test, y_test, 'sigmoid_exponent', sigmoid_range
)
axes[2].plot(sigmoid_values, sigmoid_accuracies, 'g-', linewidth=2)
axes[2].set_title('Sigmoid Exponent Sensitivity')
axes[2].set_xlabel('Sigmoid Exponent')
axes[2].set_ylabel('Accuracy')
axes[2].grid(True)

plt.tight_layout()
plt.show()

# Find optimal values from sensitivity analysis
optimal_weight = weight_values[np.argmax(weight_accuracies)]
optimal_power = power_values[np.argmax(power_accuracies)]
optimal_sigmoid = sigmoid_values[np.argmax(sigmoid_accuracies)]

print(f"Optimal parameters from sensitivity analysis:")
print(f"  Weight: {optimal_weight:.3f} (accuracy: {max(weight_accuracies):.3f})")
print(f"  Power degree: {optimal_power:.3f} (accuracy: {max(power_accuracies):.3f})")
print(f"  Sigmoid exponent: {optimal_sigmoid:.3f} (accuracy: {max(sigmoid_accuracies):.3f})")
```

## Performance Benchmarking

### Speed Comparison

```python
import time

def benchmark_rapid_refitting(model, X_train, y_train, X_test, y_test):
    """Compare rapid refitting vs full retraining performance."""
    
    # Full retraining benchmark
    start_time = time.time()
    
    new_model = XClassifier(
        max_depth=model.max_depth,
        min_info_gain=model.min_info_gain,
        weight=0.8,
        power_degree=2.0,
        sigmoid_exponent=1.5
    )
    new_model.fit(X_train, y_train)
    full_retrain_time = time.time() - start_time
    full_retrain_accuracy = new_model.score(X_test, y_test)
    
    # Rapid refitting benchmark
    start_time = time.time()
    
    model.refit(
        weight=0.8,
        power_degree=2.0,
        sigmoid_exponent=1.5
    )
    rapid_refit_time = time.time() - start_time
    rapid_refit_accuracy = model.score(X_test, y_test)
    
    # Multiple rapid refits to show consistency
    refit_times = []
    for _ in range(100):
        start_time = time.time()
        model.refit(
            weight=np.random.uniform(0.1, 1.0),
            power_degree=np.random.uniform(0.5, 3.0),
            sigmoid_exponent=np.random.uniform(0.5, 2.0)
        )
        refit_times.append(time.time() - start_time)
    
    avg_refit_time = np.mean(refit_times)
    
    print(f"Performance Benchmarking Results:")
    print(f"{'='*50}")
    print(f"Full retraining:")
    print(f"  Time: {full_retrain_time:.3f} seconds")
    print(f"  Accuracy: {full_retrain_accuracy:.3f}")
    print(f"\nRapid refitting:")
    print(f"  Time: {rapid_refit_time:.6f} seconds")
    print(f"  Accuracy: {rapid_refit_accuracy:.3f}")
    print(f"  Average time (100 refits): {avg_refit_time:.6f} seconds")
    print(f"\nSpeedup: {full_retrain_time/rapid_refit_time:.0f}x")
    print(f"Accuracy difference: {abs(full_retrain_accuracy - rapid_refit_accuracy):.4f}")
    
    return {
        'full_retrain_time': full_retrain_time,
        'rapid_refit_time': rapid_refit_time,
        'avg_refit_time': avg_refit_time,
        'speedup': full_retrain_time/rapid_refit_time,
        'accuracy_difference': abs(full_retrain_accuracy - rapid_refit_accuracy)
    }

# Run benchmark
benchmark_results = benchmark_rapid_refitting(model, X_train, y_train, X_test, y_test)
```

### Parameter Validation

```python
def safe_refit(model, **params):
    """Safely refit model with parameter validation."""
    
    # Define parameter bounds
    param_bounds = {
        'weight': (0.1, 1.0),
        'power_degree': (0.5, 3.0),
        'sigmoid_exponent': (0.5, 2.0),
        'tail_sensitivity': (0.0, 1.0)
    }
    
    # Validate parameters
    validated_params = {}
    warnings = []
    
    for param_name, param_value in params.items():
        if param_name in param_bounds:
            min_val, max_val = param_bounds[param_name]
            
            if param_value < min_val:
                validated_params[param_name] = min_val
                warnings.append(f"{param_name} clipped from {param_value} to {min_val}")
            elif param_value > max_val:
                validated_params[param_name] = max_val
                warnings.append(f"{param_name} clipped from {param_value} to {max_val}")
            else:
                validated_params[param_name] = param_value
        else:
            validated_params[param_name] = param_value
    
    # Display warnings
    if warnings:
        print("Parameter validation warnings:")
        for warning in warnings:
            print(f"  ⚠️  {warning}")
    
    # Apply validated parameters
    try:
        model.refit(**validated_params)
        print(f"✅ Successfully applied parameters: {validated_params}")
    except Exception as e:
        print(f"❌ Failed to apply parameters: {e}")
        return None
    
    return validated_params

# Usage examples
print("Testing parameter validation:")

# Valid parameters
safe_refit(model, weight=0.7, power_degree=1.5, sigmoid_exponent=1.2)

# Invalid parameters (will be clipped)
safe_refit(model, weight=1.5, power_degree=0.1, sigmoid_exponent=2.5)
```

## Integration with Other Features

### Combining with Partitioned Models

```python
def rapid_refit_partitioned_models(partitioned_models, **refit_params):
    """Apply rapid refitting to all partition models."""
    updated_models = {}
    
    for partition, model in partitioned_models.items():
        try:
            # Apply rapid refitting
            model.refit(**refit_params)
            updated_models[partition] = model
            print(f"✅ Updated parameters for {partition}")
        except Exception as e:
            print(f"❌ Failed to update {partition}: {e}")
            updated_models[partition] = model  # Keep original
    
    return updated_models

# Example usage with partitioned models
partitioned_models = {
    'segment_A': XClassifier().fit(X_train[:100], y_train[:100]),
    'segment_B': XClassifier().fit(X_train[100:200], y_train[100:200]),
    'segment_C': XClassifier().fit(X_train[200:300], y_train[200:300])
}

# Apply rapid refitting to all partitions
updated_partitioned_models = rapid_refit_partitioned_models(
    partitioned_models,
    weight=0.8,
    power_degree=2.0
)
```

### Cloud Integration

```python
def deploy_rapid_refit_model(model, client, model_name):
    """Deploy a model with rapid refitting capabilities to the cloud."""
    
    # Create base model in cloud
    model_id = client.create_model(
        model=model,
        model_name=model_name,
        model_description="Model with rapid refitting capabilities"
    )
    
    # Create versions with different parameter configurations
    parameter_variants = [
        {'weight': 0.5, 'power_degree': 1.0, 'sigmoid_exponent': 1.0},
        {'weight': 0.7, 'power_degree': 1.5, 'sigmoid_exponent': 1.2},
        {'weight': 0.9, 'power_degree': 2.0, 'sigmoid_exponent': 1.5}
    ]
    
    version_ids = {}
    
    for i, params in enumerate(parameter_variants):
        # Rapid refit with new parameters
        model.refit(**params)
        
        # Create version
        version_id = client.add_version(
            model_id=model_id,
            model=model,
            version_name=f"Variant_{i+1}",
            version_description=f"Parameters: {params}"
        )
        
        version_ids[f"variant_{i+1}"] = version_id
        print(f"Created version {version_id} with parameters: {params}")
    
    return model_id, version_ids

# Usage (assuming you have a cloud client)
# model_id, version_ids = deploy_rapid_refit_model(model, client, "Rapid Refit Model")
```

## Best Practices

### When to Use Rapid Refitting

:::tip Ideal Use Cases
1. **Parameter tuning** - Quick exploration of parameter space
2. **Real-time optimization** - Continuous model improvement
3. **A/B testing** - Testing different configurations in production
4. **Interactive analysis** - Immediate feedback during exploration
5. **Online learning** - Adapting to streaming data patterns
:::

### Performance Considerations

```python
# Best practices for rapid refitting
def rapid_refit_best_practices():
    """Demonstrate best practices for rapid refitting."""
    
    print("Rapid Refitting Best Practices:")
    print("=" * 40)
    
    print("\n1. Parameter Bounds:")
    print("   - Weight: 0.1 to 1.0")
    print("   - Power degree: 0.5 to 3.0")
    print("   - Sigmoid exponent: 0.5 to 2.0")
    
    print("\n2. Validation Strategy:")
    print("   - Always validate on held-out data")
    print("   - Use cross-validation for robust estimates")
    print("   - Monitor for overfitting")
    
    print("\n3. Performance Monitoring:")
    print("   - Track parameter changes over time")
    print("   - Monitor model stability")
    print("   - Set up alerts for performance degradation")
    
    print("\n4. Integration Patterns:")
    print("   - Combine with Bayesian optimization")
    print("   - Use in A/B testing frameworks")
    print("   - Integrate with monitoring systems")

rapid_refit_best_practices()
```

## Next Steps

:::note Ready for Advanced Optimization?
- Explore [XEvolutionaryNetwork](./XEvolutionaryNetwork.md) for sophisticated optimization strategies
- Learn about [partitioned models](./partitioned-models.md) for segment-specific rapid refitting
- Check out [custom transformers](./custom-transformers.md) for preprocessing optimization
:::

Rapid refitting is a powerful feature that enables real-time model optimization and experimentation. By separating tree structure from activation parameters, xplainable models can adapt instantly to new requirements while maintaining their transparency and interpretability.
