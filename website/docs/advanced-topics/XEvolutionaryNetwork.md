---
sidebar_position: 4
---

# XEvolutionaryNetwork

:::info Neural Network for Hyperparameter Optimization
**XEvolutionaryNetwork** is a sophisticated multi-layer optimization framework that acts like a "neural network for hyperparameter optimization." It chains together optimization layers to create powerful, automated machine learning pipelines.
:::

## Overview

XEvolutionaryNetwork represents a paradigm shift in automated machine learning. Instead of manually tuning hyperparameters, you build a network of optimization layers that automatically discover the best configurations for your specific problem.

### Key Benefits

<div className="row">
  <div className="col col--6">
    <div className="card">
      <div className="card__header">
        <h3>🧠 Intelligent Optimization</h3>
      </div>
      <div className="card__body">
        <p>Multi-layer architecture that learns optimal hyperparameters automatically.</p>
      </div>
    </div>
  </div>
  <div className="col col--6">
    <div className="card">
      <div className="card__header">
        <h3>🔗 Composable Layers</h3>
      </div>
      <div className="card__body">
        <p>Chain together different optimization strategies for complex problems.</p>
      </div>
    </div>
  </div>
</div>

<div className="row">
  <div className="col col--6">
    <div className="card">
      <div className="card__header">
        <h3>🎯 Adaptive Learning</h3>
      </div>
      <div className="card__body">
        <p>Learns from previous optimizations to improve future performance.</p>
      </div>
    </div>
  </div>
  <div className="col col--6">
    <div className="card">
      <div className="card__header">
        <h3>🚀 Automated Pipelines</h3>
      </div>
      <div className="card__body">
        <p>Create end-to-end automated machine learning workflows.</p>
      </div>
    </div>
  </div>
</div>

## Architecture Overview

XEvolutionaryNetwork consists of specialized layers that work together:

1. **XParamOptimiser**: Bayesian optimization for hyperparameter tuning
2. **Evolve**: Evolutionary algorithms for complex search spaces
3. **Tighten**: Gradient-based fine-tuning for precise optimization
4. **Target**: Goal-oriented optimization with specific objectives
5. **NLP**: Natural language processing for text-based optimization

:::tip Think of it as a Neural Network
- **Layers**: Each optimization strategy is a layer
- **Forward Pass**: Data flows through optimization layers
- **Backpropagation**: Results inform previous layers
- **Training**: The network learns optimal optimization strategies
:::

## Available Layers

### XParamOptimiser Layer

The foundational layer for Bayesian optimization:

```python
from xplainable.core.optimisation import XParamOptimiser, XEvolutionaryNetwork
from xplainable.core.models import XClassifier

# Create XParamOptimiser layer
param_layer = XParamOptimiser(
    model=XClassifier(),
    X=X_train,
    y=y_train,
    metric='roc_auc',
    cv=5,
    n_iter=50,
    random_state=42
)

# Define search space
param_space = {
    'max_depth': [3, 10],
    'min_info_gain': [0.001, 0.1],
    'weight': [0.1, 1.0],
    'power_degree': [0.5, 3.0]
}

# Optimize parameters
best_params = param_layer.optimise(param_space)
print(f"Best parameters: {best_params}")
```

### Evolve Layer

Evolutionary optimization for complex problems:

```python
from xplainable.core.optimisation import Evolve

# Create evolution layer
evolve_layer = Evolve(
    population_size=50,
    generations=100,
    mutation_rate=0.1,
    crossover_rate=0.8,
    selection_method='tournament'
)

# Define complex search space
complex_space = {
    'model_architecture': ['shallow', 'medium', 'deep'],
    'regularization': [0.0, 0.1, 0.2, 0.3],
    'feature_selection': ['none', 'univariate', 'recursive'],
    'preprocessing': ['standard', 'minmax', 'robust']
}

# Evolve solutions
best_solution = evolve_layer.evolve(complex_space, fitness_function)
```

### Tighten Layer

Gradient-based fine-tuning:

```python
from xplainable.core.optimisation import Tighten

# Create tightening layer
tighten_layer = Tighten(
    learning_rate=0.01,
    max_iterations=100,
    tolerance=1e-6,
    optimization_method='adam'
)

# Fine-tune parameters
refined_params = tighten_layer.tighten(initial_params, objective_function)
```

### Target Layer

Goal-oriented optimization:

```python
from xplainable.core.optimisation import Target

# Create target layer
target_layer = Target(
    primary_metric='accuracy',
    secondary_metrics=['precision', 'recall'],
    constraints={'max_depth': 8, 'training_time': 300}
)

# Optimize towards targets
optimized_model = target_layer.optimize(model, X_train, y_train)
```

### NLP Layer

Natural language processing optimization:

```python
from xplainable.core.optimisation import NLP

# Create NLP layer
nlp_layer = NLP(
    text_features=['description', 'comments'],
    embedding_method='tfidf',
    max_features=10000,
    ngram_range=(1, 2)
)

# Optimize text processing
text_optimized_model = nlp_layer.optimize(model, X_text, y)
```

## Basic Usage Examples

### Simple Classification Network

```python
from xplainable.core.optimisation import XEvolutionaryNetwork, XParamOptimiser
from xplainable.core.models import XClassifier
import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv('classification_data.csv')
X = data.drop('target', axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create evolutionary network
network = XEvolutionaryNetwork()

# Add optimization layers
network.add_layer(XParamOptimiser(
    model=XClassifier(),
    X=X_train,
    y=y_train,
    metric='f1_weighted',
    cv=5,
    n_iter=30
))

# Define parameter space
param_space = {
    'max_depth': [3, 8],
    'min_info_gain': [0.001, 0.05],
    'weight': [0.3, 0.9],
    'power_degree': [1.0, 2.5]
}

# Train the network
print("Training XEvolutionaryNetwork...")
network.fit(param_space)

# Get optimized model
best_model = network.get_best_model()
print(f"Best parameters: {network.get_best_params()}")
print(f"Best CV score: {network.get_best_score():.3f}")

# Evaluate on test set
test_accuracy = best_model.score(X_test, y_test)
print(f"Test accuracy: {test_accuracy:.3f}")
```

### Regression Network

```python
from xplainable.core.models import XRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Create regression network
regression_network = XEvolutionaryNetwork()

# Add regression-specific optimization
regression_network.add_layer(XParamOptimiser(
    model=XRegressor(),
    X=X_train,
    y=y_train,
    metric='neg_mean_squared_error',
    cv=5,
    n_iter=40
))

# Regression parameter space
regression_space = {
    'max_depth': [3, 10],
    'min_info_gain': [0.001, 0.1],
    'weight': [0.1, 1.0],
    'power_degree': [0.5, 3.0],
    'prediction_range': [[-100, 100], [0, 200], [-50, 150]]
}

# Train regression network
print("Training regression network...")
regression_network.fit(regression_space)

# Evaluate regression performance
best_regressor = regression_network.get_best_model()
predictions = best_regressor.predict(X_test)

r2 = r2_score(y_test, predictions)
mse = mean_squared_error(y_test, predictions)

print(f"Best regression parameters: {regression_network.get_best_params()}")
print(f"Test R² score: {r2:.3f}")
print(f"Test MSE: {mse:.3f}")
```

## Advanced Layer Configurations

### Multi-Objective Optimization Network

```python
from xplainable.core.optimisation import Target, Evolve

# Create multi-objective network
multi_objective_network = XEvolutionaryNetwork()

# Add target layer for multiple objectives
multi_objective_network.add_layer(Target(
    primary_metric='accuracy',
    secondary_metrics=['precision', 'recall', 'f1_score'],
    weights=[0.4, 0.2, 0.2, 0.2],  # Relative importance
    constraints={
        'max_depth': 8,
        'training_time': 300,
        'model_size': 1000
    }
))

# Add evolutionary layer for complex search
multi_objective_network.add_layer(Evolve(
    population_size=30,
    generations=50,
    mutation_rate=0.15,
    crossover_rate=0.7,
    selection_method='pareto'  # Pareto-optimal selection
))

# Add parameter optimization layer
multi_objective_network.add_layer(XParamOptimiser(
    model=XClassifier(),
    X=X_train,
    y=y_train,
    metric='f1_weighted',
    cv=3,
    n_iter=20
))

# Complex parameter space
complex_space = {
    'max_depth': [3, 12],
    'min_info_gain': [0.001, 0.2],
    'weight': [0.1, 1.0],
    'power_degree': [0.5, 4.0],
    'sigmoid_exponent': [0.5, 2.0],
    'feature_selection_k': [5, 50],
    'preprocessing_method': ['standard', 'minmax', 'robust', 'quantile']
}

# Train multi-objective network
print("Training multi-objective network...")
multi_objective_network.fit(complex_space)

# Get Pareto-optimal solutions
pareto_solutions = multi_objective_network.get_pareto_front()
print(f"Found {len(pareto_solutions)} Pareto-optimal solutions")

for i, solution in enumerate(pareto_solutions[:3]):
    print(f"Solution {i+1}: {solution['metrics']}")
```

### Hierarchical Optimization Network

```python
# Create hierarchical network with progressive refinement
hierarchical_network = XEvolutionaryNetwork()

# Stage 1: Coarse optimization
hierarchical_network.add_layer(XParamOptimiser(
    model=XClassifier(),
    X=X_train,
    y=y_train,
    metric='roc_auc',
    cv=3,
    n_iter=20,
    stage_name='coarse'
))

# Stage 2: Evolutionary refinement
hierarchical_network.add_layer(Evolve(
    population_size=20,
    generations=30,
    mutation_rate=0.1,
    crossover_rate=0.8,
    stage_name='evolve'
))

# Stage 3: Fine-tuning
hierarchical_network.add_layer(Tighten(
    learning_rate=0.005,
    max_iterations=50,
    tolerance=1e-5,
    stage_name='fine_tune'
))

# Progressive parameter spaces
coarse_space = {
    'max_depth': [3, 8],
    'min_info_gain': [0.01, 0.1],
    'weight': [0.3, 0.9]
}

refined_space = {
    'max_depth': [4, 7],  # Narrowed based on coarse results
    'min_info_gain': [0.005, 0.05],
    'weight': [0.4, 0.8],
    'power_degree': [1.0, 2.0]
}

fine_tune_space = {
    'weight': [0.55, 0.75],  # Very narrow range
    'power_degree': [1.2, 1.8],
    'sigmoid_exponent': [0.9, 1.3]
}

# Train hierarchical network
print("Training hierarchical network...")
hierarchical_network.fit([coarse_space, refined_space, fine_tune_space])

# Get results from each stage
coarse_results = hierarchical_network.get_stage_results('coarse')
evolved_results = hierarchical_network.get_stage_results('evolve')
fine_tuned_results = hierarchical_network.get_stage_results('fine_tune')

print(f"Coarse optimization: {coarse_results['best_score']:.3f}")
print(f"Evolutionary refinement: {evolved_results['best_score']:.3f}")
print(f"Fine-tuning: {fine_tuned_results['best_score']:.3f}")
```

## Specialized Optimization Scenarios

### Time Series Optimization

```python
# Create time series specific network
ts_network = XEvolutionaryNetwork()

# Add time series aware optimization
ts_network.add_layer(XParamOptimiser(
    model=XRegressor(),
    X=X_train,
    y=y_train,
    metric='neg_mean_absolute_error',
    cv=TimeSeriesSplit(n_splits=5),  # Time series cross-validation
    n_iter=30
))

# Time series specific parameters
ts_space = {
    'max_depth': [3, 8],
    'min_info_gain': [0.001, 0.05],
    'weight': [0.2, 0.8],
    'power_degree': [0.8, 2.2],
    'prediction_range': [[-1000, 1000], [0, 500], [-200, 800]],
    'seasonal_adjustment': [True, False],
    'trend_removal': ['linear', 'polynomial', 'none']
}

# Train time series network
print("Training time series network...")
ts_network.fit(ts_space)

# Evaluate with time series metrics
best_ts_model = ts_network.get_best_model()
ts_predictions = best_ts_model.predict(X_test)

# Time series specific evaluation
from sklearn.metrics import mean_absolute_percentage_error

mape = mean_absolute_percentage_error(y_test, ts_predictions)
print(f"Time series MAPE: {mape:.3f}")
```

### Imbalanced Data Optimization

```python
# Create network for imbalanced classification
imbalanced_network = XEvolutionaryNetwork()

# Add imbalanced data specific optimization
imbalanced_network.add_layer(XParamOptimiser(
    model=XClassifier(),
    X=X_train,
    y=y_train,
    metric='f1_weighted',  # Better for imbalanced data
    cv=StratifiedKFold(n_splits=5),
    n_iter=35
))

# Add resampling optimization
imbalanced_network.add_layer(Evolve(
    population_size=25,
    generations=40,
    mutation_rate=0.12,
    crossover_rate=0.75,
    fitness_function='balanced_accuracy'
))

# Imbalanced data parameter space
imbalanced_space = {
    'max_depth': [4, 10],
    'min_info_gain': [0.001, 0.08],
    'weight': [0.2, 0.9],
    'power_degree': [1.0, 3.0],
    'class_weight': ['balanced', 'balanced_subsample', None],
    'sampling_strategy': ['over', 'under', 'combined'],
    'sampling_ratio': [0.5, 1.0, 1.5]
}

# Train imbalanced network
print("Training imbalanced data network...")
imbalanced_network.fit(imbalanced_space)

# Evaluate with imbalanced metrics
best_imbalanced_model = imbalanced_network.get_best_model()
imbalanced_predictions = best_imbalanced_model.predict(X_test)

from sklearn.metrics import classification_report, balanced_accuracy_score

balanced_acc = balanced_accuracy_score(y_test, imbalanced_predictions)
print(f"Balanced accuracy: {balanced_acc:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, imbalanced_predictions))
```

### High-Dimensional Data Optimization

```python
# Create network for high-dimensional data
high_dim_network = XEvolutionaryNetwork()

# Add dimensionality reduction optimization
high_dim_network.add_layer(XParamOptimiser(
    model=XClassifier(),
    X=X_train,
    y=y_train,
    metric='roc_auc',
    cv=5,
    n_iter=25
))

# Add feature selection layer
high_dim_network.add_layer(Target(
    primary_metric='accuracy',
    secondary_metrics=['feature_count'],
    weights=[0.8, 0.2],  # Prefer accuracy but consider feature count
    constraints={'max_features': 100}
))

# High-dimensional parameter space
high_dim_space = {
    'max_depth': [3, 6],  # Simpler models for high dimensions
    'min_info_gain': [0.005, 0.05],
    'weight': [0.3, 0.8],
    'power_degree': [1.0, 2.0],
    'feature_selection_method': ['univariate', 'recursive', 'lasso'],
    'feature_selection_k': [10, 50, 100],
    'dimensionality_reduction': ['pca', 'ica', 'none'],
    'n_components': [10, 50, 100]
}

# Train high-dimensional network
print("Training high-dimensional network...")
high_dim_network.fit(high_dim_space)

# Analyze feature importance
best_high_dim_model = high_dim_network.get_best_model()
feature_importance = best_high_dim_model.feature_importance()

print(f"Selected {len(feature_importance)} features")
print(f"Top 5 features: {feature_importance.head().index.tolist()}")
```

## Custom Layer Development

### Creating Custom Optimization Layers

```python
from xplainable.core.optimisation import BaseOptimizationLayer

class CustomGradientLayer(BaseOptimizationLayer):
    """Custom gradient-based optimization layer."""
    
    def __init__(self, learning_rate=0.01, momentum=0.9, max_epochs=100):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.max_epochs = max_epochs
        self.velocity = {}
    
    def optimize(self, param_space, objective_function, X, y):
        """Custom gradient-based optimization."""
        # Initialize parameters
        current_params = self._initialize_params(param_space)
        best_params = current_params.copy()
        best_score = float('-inf')
        
        # Initialize velocity for momentum
        for param in current_params:
            self.velocity[param] = 0
        
        for epoch in range(self.max_epochs):
            # Calculate gradients (numerical approximation)
            gradients = self._calculate_gradients(
                current_params, objective_function, X, y
            )
            
            # Update parameters with momentum
            for param, gradient in gradients.items():
                self.velocity[param] = (self.momentum * self.velocity[param] + 
                                      self.learning_rate * gradient)
                current_params[param] -= self.velocity[param]
                
                # Apply bounds
                current_params[param] = self._apply_bounds(
                    current_params[param], param_space[param]
                )
            
            # Evaluate current parameters
            score = objective_function(current_params, X, y)
            
            if score > best_score:
                best_score = score
                best_params = current_params.copy()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Score = {score:.4f}")
        
        return best_params, best_score
    
    def _calculate_gradients(self, params, objective_function, X, y, epsilon=1e-5):
        """Calculate numerical gradients."""
        gradients = {}
        base_score = objective_function(params, X, y)
        
        for param_name, param_value in params.items():
            # Forward difference
            params_forward = params.copy()
            params_forward[param_name] += epsilon
            forward_score = objective_function(params_forward, X, y)
            
            # Calculate gradient
            gradients[param_name] = (forward_score - base_score) / epsilon
        
        return gradients
    
    def _initialize_params(self, param_space):
        """Initialize parameters randomly within bounds."""
        import random
        params = {}
        
        for param_name, bounds in param_space.items():
            if isinstance(bounds, list) and len(bounds) == 2:
                params[param_name] = random.uniform(bounds[0], bounds[1])
            else:
                params[param_name] = random.choice(bounds)
        
        return params
    
    def _apply_bounds(self, value, bounds):
        """Apply parameter bounds."""
        if isinstance(bounds, list) and len(bounds) == 2:
            return max(bounds[0], min(bounds[1], value))
        return value

# Usage of custom layer
custom_network = XEvolutionaryNetwork()

# Add custom gradient layer
custom_network.add_layer(CustomGradientLayer(
    learning_rate=0.005,
    momentum=0.95,
    max_epochs=50
))

# Add standard optimization layer
custom_network.add_layer(XParamOptimiser(
    model=XClassifier(),
    X=X_train,
    y=y_train,
    metric='accuracy',
    cv=5,
    n_iter=20
))

# Train network with custom layer
print("Training network with custom layer...")
custom_network.fit(param_space)
```

### Ensemble Layer

```python
class EnsembleOptimizationLayer(BaseOptimizationLayer):
    """Ensemble of multiple optimization strategies."""
    
    def __init__(self, strategies=['bayesian', 'evolutionary', 'random']):
        super().__init__()
        self.strategies = strategies
        self.strategy_results = {}
    
    def optimize(self, param_space, objective_function, X, y):
        """Run multiple optimization strategies and ensemble results."""
        all_results = {}
        
        for strategy in self.strategies:
            print(f"Running {strategy} optimization...")
            
            if strategy == 'bayesian':
                optimizer = XParamOptimiser(
                    model=XClassifier(),
                    X=X, y=y,
                    metric='accuracy',
                    cv=3, n_iter=15
                )
                best_params = optimizer.optimise(param_space)
                score = objective_function(best_params, X, y)
                
            elif strategy == 'evolutionary':
                # Simplified evolutionary approach
                best_params, score = self._evolutionary_search(
                    param_space, objective_function, X, y
                )
                
            elif strategy == 'random':
                best_params, score = self._random_search(
                    param_space, objective_function, X, y
                )
            
            all_results[strategy] = {
                'params': best_params,
                'score': score
            }
        
        # Ensemble the results (select best)
        best_strategy = max(all_results, key=lambda x: all_results[x]['score'])
        best_params = all_results[best_strategy]['params']
        best_score = all_results[best_strategy]['score']
        
        print(f"Best strategy: {best_strategy} with score {best_score:.4f}")
        
        return best_params, best_score
    
    def _evolutionary_search(self, param_space, objective_function, X, y):
        """Simplified evolutionary search."""
        import random
        
        population_size = 20
        generations = 30
        
        # Initialize population
        population = []
        for _ in range(population_size):
            individual = self._random_individual(param_space)
            score = objective_function(individual, X, y)
            population.append((individual, score))
        
        # Evolve population
        for generation in range(generations):
            # Selection and crossover
            population.sort(key=lambda x: x[1], reverse=True)
            new_population = population[:population_size//2]  # Keep best half
            
            # Generate offspring
            while len(new_population) < population_size:
                parent1, parent2 = random.sample(population[:10], 2)
                child = self._crossover(parent1[0], parent2[0], param_space)
                child = self._mutate(child, param_space)
                score = objective_function(child, X, y)
                new_population.append((child, score))
            
            population = new_population
        
        # Return best individual
        best_individual = max(population, key=lambda x: x[1])
        return best_individual[0], best_individual[1]
    
    def _random_search(self, param_space, objective_function, X, y):
        """Random search optimization."""
        import random
        
        best_params = None
        best_score = float('-inf')
        
        for _ in range(50):  # 50 random trials
            params = self._random_individual(param_space)
            score = objective_function(params, X, y)
            
            if score > best_score:
                best_score = score
                best_params = params
        
        return best_params, best_score
    
    def _random_individual(self, param_space):
        """Generate random individual."""
        import random
        individual = {}
        
        for param_name, bounds in param_space.items():
            if isinstance(bounds, list) and len(bounds) == 2:
                individual[param_name] = random.uniform(bounds[0], bounds[1])
            else:
                individual[param_name] = random.choice(bounds)
        
        return individual
    
    def _crossover(self, parent1, parent2, param_space):
        """Simple crossover operation."""
        import random
        child = {}
        
        for param_name in param_space:
            if random.random() < 0.5:
                child[param_name] = parent1[param_name]
            else:
                child[param_name] = parent2[param_name]
        
        return child
    
    def _mutate(self, individual, param_space, mutation_rate=0.1):
        """Simple mutation operation."""
        import random
        
        for param_name, bounds in param_space.items():
            if random.random() < mutation_rate:
                if isinstance(bounds, list) and len(bounds) == 2:
                    individual[param_name] = random.uniform(bounds[0], bounds[1])
                else:
                    individual[param_name] = random.choice(bounds)
        
        return individual

# Usage of ensemble layer
ensemble_network = XEvolutionaryNetwork()

# Add ensemble layer
ensemble_network.add_layer(EnsembleOptimizationLayer(
    strategies=['bayesian', 'evolutionary', 'random']
))

# Train ensemble network
print("Training ensemble network...")
ensemble_network.fit(param_space)
```

## Performance Monitoring and Analysis

### Comprehensive Optimization Tracking

```python
class OptimizationTracker:
    """Track optimization progress across all layers."""
    
    def __init__(self):
        self.optimization_history = []
        self.layer_performance = {}
        self.convergence_data = {}
    
    def track_optimization(self, network, param_space):
        """Track optimization progress."""
        # Monitor each layer
        for i, layer in enumerate(network.layers):
            layer_name = f"Layer_{i}_{type(layer).__name__}"
            
            # Track layer performance
            layer_results = layer.optimize(param_space, objective_function, X_train, y_train)
            
            self.layer_performance[layer_name] = {
                'best_params': layer_results[0],
                'best_score': layer_results[1],
                'optimization_time': layer.optimization_time,
                'iterations': layer.n_iterations
            }
            
            # Track convergence
            if hasattr(layer, 'convergence_history'):
                self.convergence_data[layer_name] = layer.convergence_history
    
    def plot_optimization_progress(self):
        """Plot optimization progress for all layers."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Layer performance comparison
        layer_names = list(self.layer_performance.keys())
        scores = [self.layer_performance[name]['best_score'] for name in layer_names]
        
        axes[0, 0].bar(layer_names, scores)
        axes[0, 0].set_title('Layer Performance Comparison')
        axes[0, 0].set_ylabel('Best Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Optimization time comparison
        times = [self.layer_performance[name]['optimization_time'] for name in layer_names]
        axes[0, 1].bar(layer_names, times)
        axes[0, 1].set_title('Optimization Time Comparison')
        axes[0, 1].set_ylabel('Time (seconds)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Convergence curves
        for layer_name, convergence in self.convergence_data.items():
            axes[1, 0].plot(convergence, label=layer_name)
        axes[1, 0].set_title('Convergence Curves')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Parameter distribution
        all_params = []
        for layer_name, data in self.layer_performance.items():
            for param_name, param_value in data['best_params'].items():
                if isinstance(param_value, (int, float)):
                    all_params.append(param_value)
        
        axes[1, 1].hist(all_params, bins=20, alpha=0.7)
        axes[1, 1].set_title('Parameter Value Distribution')
        axes[1, 1].set_xlabel('Parameter Value')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
    
    def generate_optimization_report(self):
        """Generate comprehensive optimization report."""
        print("XEvolutionaryNetwork Optimization Report")
        print("=" * 50)
        
        # Overall performance
        best_layer = max(self.layer_performance.items(), 
                        key=lambda x: x[1]['best_score'])
        
        print(f"\nBest Performing Layer: {best_layer[0]}")
        print(f"Best Score: {best_layer[1]['best_score']:.4f}")
        print(f"Optimization Time: {best_layer[1]['optimization_time']:.2f} seconds")
        
        # Layer comparison
        print(f"\nLayer Performance Summary:")
        for layer_name, data in self.layer_performance.items():
            print(f"  {layer_name}: {data['best_score']:.4f} "
                  f"({data['optimization_time']:.2f}s)")
        
        # Parameter analysis
        print(f"\nParameter Analysis:")
        param_counts = {}
        for layer_name, data in self.layer_performance.items():
            for param_name in data['best_params']:
                param_counts[param_name] = param_counts.get(param_name, 0) + 1
        
        for param_name, count in param_counts.items():
            print(f"  {param_name}: optimized in {count} layers")
        
        return best_layer

# Usage
tracker = OptimizationTracker()

# Create and train network
network = XEvolutionaryNetwork()
network.add_layer(XParamOptimiser(
    model=XClassifier(),
    X=X_train, y=y_train,
    metric='f1_weighted',
    cv=5, n_iter=30
))

# Track optimization
tracker.track_optimization(network, param_space)

# Generate report and plots
best_layer = tracker.generate_optimization_report()
tracker.plot_optimization_progress()
```

### Automated Pipeline Selection

```python
class AutoPipelineSelector:
    """Automatically select best optimization pipeline."""
    
    def __init__(self, problem_type='classification'):
        self.problem_type = problem_type
        self.pipeline_templates = self._create_pipeline_templates()
    
    def _create_pipeline_templates(self):
        """Create predefined pipeline templates."""
        templates = {
            'simple': {
                'layers': [XParamOptimiser],
                'config': {'n_iter': 30, 'cv': 5}
            },
            'evolutionary': {
                'layers': [XParamOptimiser, Evolve],
                'config': {'n_iter': 20, 'cv': 3, 'generations': 30}
            },
            'hierarchical': {
                'layers': [XParamOptimiser, Evolve, Tighten],
                'config': {'n_iter': 15, 'cv': 3, 'generations': 20}
            },
            'multi_objective': {
                'layers': [Target, XParamOptimiser],
                'config': {'n_iter': 25, 'cv': 5}
            }
        }
        return templates
    
    def select_pipeline(self, X, y, time_budget=300):
        """Select best pipeline based on data characteristics."""
        data_characteristics = self._analyze_data(X, y)
        
        # Select pipeline based on characteristics
        if data_characteristics['n_samples'] < 1000:
            pipeline_name = 'simple'
        elif data_characteristics['n_features'] > 100:
            pipeline_name = 'hierarchical'
        elif data_characteristics['class_imbalance'] > 0.8:
            pipeline_name = 'multi_objective'
        else:
            pipeline_name = 'evolutionary'
        
        print(f"Selected pipeline: {pipeline_name}")
        print(f"Data characteristics: {data_characteristics}")
        
        return self._build_pipeline(pipeline_name, X, y, time_budget)
    
    def _analyze_data(self, X, y):
        """Analyze data characteristics."""
        import numpy as np
        from collections import Counter
        
        characteristics = {
            'n_samples': len(X),
            'n_features': X.shape[1],
            'feature_types': self._analyze_feature_types(X),
            'missing_values': X.isnull().sum().sum() / (X.shape[0] * X.shape[1]),
        }
        
        if self.problem_type == 'classification':
            class_counts = Counter(y)
            characteristics['n_classes'] = len(class_counts)
            characteristics['class_imbalance'] = max(class_counts.values()) / min(class_counts.values())
        else:
            characteristics['target_variance'] = np.var(y)
            characteristics['target_range'] = y.max() - y.min()
        
        return characteristics
    
    def _analyze_feature_types(self, X):
        """Analyze feature types."""
        numeric_features = X.select_dtypes(include=[np.number]).shape[1]
        categorical_features = X.select_dtypes(include=['object']).shape[1]
        
        return {
            'numeric': numeric_features,
            'categorical': categorical_features,
            'ratio': numeric_features / (numeric_features + categorical_features)
        }
    
    def _build_pipeline(self, pipeline_name, X, y, time_budget):
        """Build the selected pipeline."""
        template = self.pipeline_templates[pipeline_name]
        network = XEvolutionaryNetwork()
        
        # Add layers based on template
        for layer_class in template['layers']:
            if layer_class == XParamOptimiser:
                network.add_layer(XParamOptimiser(
                    model=XClassifier() if self.problem_type == 'classification' else XRegressor(),
                    X=X, y=y,
                    metric='f1_weighted' if self.problem_type == 'classification' else 'neg_mean_squared_error',
                    cv=template['config']['cv'],
                    n_iter=template['config']['n_iter']
                ))
            elif layer_class == Evolve:
                network.add_layer(Evolve(
                    population_size=20,
                    generations=template['config']['generations'],
                    mutation_rate=0.1,
                    crossover_rate=0.8
                ))
            elif layer_class == Tighten:
                network.add_layer(Tighten(
                    learning_rate=0.01,
                    max_iterations=50,
                    tolerance=1e-6
                ))
            elif layer_class == Target:
                if self.problem_type == 'classification':
                    network.add_layer(Target(
                        primary_metric='f1_weighted',
                        secondary_metrics=['precision', 'recall'],
                        weights=[0.6, 0.2, 0.2]
                    ))
                else:
                    network.add_layer(Target(
                        primary_metric='r2',
                        secondary_metrics=['neg_mean_squared_error'],
                        weights=[0.7, 0.3]
                    ))
        
        return network

# Usage
auto_selector = AutoPipelineSelector(problem_type='classification')

# Automatically select and build pipeline
optimal_network = auto_selector.select_pipeline(X_train, y_train, time_budget=300)

# Train the automatically selected pipeline
print("Training automatically selected pipeline...")
optimal_network.fit(param_space)

# Evaluate results
best_model = optimal_network.get_best_model()
final_accuracy = best_model.score(X_test, y_test)
print(f"Final test accuracy: {final_accuracy:.3f}")
```

## Best Practices

### Network Architecture Design

:::tip Designing Effective Networks
1. **Start Simple**: Begin with basic XParamOptimiser layer
2. **Add Complexity Gradually**: Add layers based on problem requirements
3. **Consider Data Size**: Simpler networks for small datasets
4. **Balance Exploration vs Exploitation**: Mix broad and focused layers
5. **Monitor Performance**: Track each layer's contribution
:::

### Performance Optimization

```python
def optimize_network_performance(network, X, y, param_space):
    """Optimize network performance and efficiency."""
    
    # Performance monitoring
    start_time = time.time()
    
    # Memory efficient training
    if len(X) > 10000:
        # Use smaller CV folds for large datasets
        for layer in network.layers:
            if hasattr(layer, 'cv'):
                layer.cv = 3
    
    # Adaptive parameter space
    if X.shape[1] > 100:
        # Simpler models for high-dimensional data
        param_space['max_depth'] = [3, 6]
    
    # Train network
    network.fit(param_space)
    
    training_time = time.time() - start_time
    
    print(f"Network training completed in {training_time:.2f} seconds")
    
    # Performance analysis
    best_model = network.get_best_model()
    best_params = network.get_best_params()
    
    return {
        'model': best_model,
        'params': best_params,
        'training_time': training_time,
        'network_complexity': len(network.layers)
    }
```

## Integration with Other Features

### Combining with Partitioned Models

```python
def create_partitioned_evolutionary_network(data, partition_column):
    """Create evolutionary networks for each partition."""
    partition_networks = {}
    
    for partition in data[partition_column].unique():
        partition_data = data[data[partition_column] == partition]
        
        if len(partition_data) < 100:
            continue
        
        X_partition = partition_data.drop(['target', partition_column], axis=1)
        y_partition = partition_data['target']
        
        # Create partition-specific network
        partition_network = XEvolutionaryNetwork()
        
        # Customize network based on partition characteristics
        if partition == 'high_value':
            # More sophisticated optimization for high-value partition
            partition_network.add_layer(XParamOptimiser(
                model=XClassifier(),
                X=X_partition, y=y_partition,
                metric='f1_weighted',
                cv=5, n_iter=40
            ))
            partition_network.add_layer(Evolve(
                population_size=30,
                generations=50
            ))
        else:
            # Simpler optimization for other partitions
            partition_network.add_layer(XParamOptimiser(
                model=XClassifier(),
                X=X_partition, y=y_partition,
                metric='accuracy',
                cv=3, n_iter=20
            ))
        
        partition_networks[partition] = partition_network
    
    return partition_networks

# Usage
partitioned_networks = create_partitioned_evolutionary_network(data, 'customer_segment')

# Train all partition networks
for partition, network in partitioned_networks.items():
    print(f"Training network for partition: {partition}")
    network.fit(param_space)
```

### Cloud Integration

```python
def deploy_evolutionary_network_to_cloud(network, client, model_name):
    """Deploy evolutionary network results to cloud."""
    
    # Get best model from network
    best_model = network.get_best_model()
    best_params = network.get_best_params()
    
    # Deploy base model
    model_id = client.create_model(
        model=best_model,
        model_name=model_name,
        model_description=f"Optimized with XEvolutionaryNetwork: {best_params}"
    )
    
    # Deploy alternative configurations from network
    alternative_configs = network.get_pareto_front()[:3]  # Top 3 alternatives
    
    version_ids = []
    for i, config in enumerate(alternative_configs):
        version_id = client.add_version(
            model_id=model_id,
            model=config['model'],
            version_name=f"Alternative_{i+1}",
            version_description=f"Score: {config['score']:.3f}, Params: {config['params']}"
        )
        version_ids.append(version_id)
    
    return model_id, version_ids
```

## Next Steps

:::note Ready for Production?
- Explore [rapid refitting](./rapid-refitting.md) for real-time optimization of network results
- Learn about [partitioned models](./partitioned-models.md) for segment-specific evolutionary networks
- Check out [custom transformers](./custom-transformers.md) for preprocessing optimization layers
:::

XEvolutionaryNetwork represents the cutting edge of automated machine learning, providing a flexible and powerful framework for creating sophisticated optimization pipelines. By combining multiple optimization strategies in a neural network-like architecture, you can achieve superior model performance while maintaining the transparency and interpretability that makes xplainable unique.