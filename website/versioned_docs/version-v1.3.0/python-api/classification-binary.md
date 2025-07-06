---
sidebar_position: 2
---

import BlogPost from "@site/src/components/Cards/BlogPost.jsx";

# Classification – Binary

:::info Transparent Binary Classification
**XClassifier** provides transparent binary classification with real-time explainability. Unlike black-box models, you get instant insights into how predictions are made without needing surrogate models.
:::

## Overview

The `XClassifier` is xplainable's flagship transparent classification model. It uses a novel feature-wise ensemble approach where each feature gets its own decision tree, optimized for maximum information gain while maintaining complete interpretability.

### Key Features

<div className="row">
  <div className="col col--6">
    <div className="card">
      <div className="card__header">
        <h3>🔍 Real-time Explainability</h3>
      </div>
      <div className="card__body">
        <p>Get instant explanations as part of the prediction process - no SHAP or LIME needed.</p>
      </div>
    </div>
  </div>
  <div className="col col--6">
    <div className="card">
      <div className="card__header">
        <h3>⚡ Rapid Refitting</h3>
      </div>
      <div className="card__body">
        <p>Update parameters on individual features without complete retraining.</p>
      </div>
    </div>
  </div>
</div>

<div className="row">
  <div className="col col--6">
    <div className="card">
      <div className="card__header">
        <h3>🎯 Feature-wise Ensemble</h3>
      </div>
      <div className="card__body">
        <p>Each feature gets its own decision tree, providing granular control and transparency.</p>
      </div>
    </div>
  </div>
  <div className="col col--6">
    <div className="card">
      <div className="card__header">
        <h3>📊 Probability Calibration</h3>
      </div>
      <div className="card__body">
        <p>Built-in probability mapping for reliable confidence scores.</p>
      </div>
    </div>
  </div>
</div>

## Quick Start

### GUI Interface

Training an `XClassifier` with the embedded GUI is the fastest way to get started:

```python
import xplainable as xp
import pandas as pd

# Load your data
data = pd.read_csv('data.csv')

# Train your model (opens embedded GUI)
model = xp.classifier(data)
```

:::tip GUI Benefits
The GUI interface provides:
- Interactive hyperparameter tuning
- Real-time performance metrics
- Visual feature importance
- Automatic data preprocessing options
:::

### Python API

For programmatic control, use the Python API:

```python
from xplainable.core.models import XClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# Load and prepare data
data = pd.read_csv('data.csv')
X, y = data.drop('target', axis=1), data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = XClassifier()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# Get explanations
model.explain()
```

## Model Parameters

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_depth` | int | 5 | Maximum depth of decision trees |
| `min_info_gain` | float | 0.01 | Minimum information gain for splits |
| `min_leaf_size` | int | 5 | Minimum samples required for leaf nodes |
| `weight` | float | 0.5 | Activation function weight parameter |
| `power_degree` | int | 1 | Power degree for activation function |
| `sigmoid_exponent` | int | 1 | Sigmoid exponent for activation |

### Advanced Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tail_sensitivity` | float | 0.5 | Weight for divisive leaf nodes |
| `ignore_nan` | bool | True | Handle missing values automatically |
| `map_calibration` | bool | True | Apply probability calibration mapping |

### Example with Parameters

```python
model = XClassifier(
    max_depth=7,
    min_info_gain=0.005,
    min_leaf_size=10,
    weight=0.7,
    power_degree=2,
    sigmoid_exponent=1,
    tail_sensitivity=0.3,
    ignore_nan=True,
    map_calibration=True
)
```

## Model Methods

### Training Methods

```python
# Basic training
model.fit(X_train, y_train)

# With validation data
model.fit(X_train, y_train, validation_data=(X_val, y_val))

# With sample weights
model.fit(X_train, y_train, sample_weight=weights)
```

### Prediction Methods

```python
# Binary predictions
predictions = model.predict(X_test)

# Probability predictions
probabilities = model.predict_proba(X_test)

# Single sample prediction
single_pred = model.predict(X_test.iloc[[0]])
```

### Explanation Methods

```python
# Global explanations
model.explain()

# Feature importance
importance = model.feature_importance()

# Local explanations for specific samples
model.explain(X_test.iloc[[0]])

# Waterfall plot for decision breakdown
model.waterfall(X_test.iloc[[0]])
```

### Model Inspection

```python
# Get model statistics
stats = model.stats()

# View decision trees for each feature
trees = model.trees()

# Get feature contributions
contributions = model.feature_contributions(X_test)
```

## Advanced Usage

### Rapid Refitting

One of xplainable's unique features is the ability to update parameters without complete retraining:

```python
# Initial training
model = XClassifier()
model.fit(X_train, y_train)

# Update parameters rapidly
model.refit(
    max_depth=7,
    weight=0.8,
    features=['feature1', 'feature2']  # Only update specific features
)

# Performance comparison
print(f"Original accuracy: {model.score(X_test, y_test)}")
```

:::tip Rapid Refitting Benefits
- **10-100x faster** than complete retraining
- **Feature-specific updates** for granular control
- **Real-time parameter tuning** in production
- **A/B testing** different configurations
:::

### Partitioned Classification

For datasets with natural segments, use `PartitionedClassifier`:

```python
from xplainable.core.models import PartitionedClassifier, XClassifier

# Create partitioned model
partitioned_model = PartitionedClassifier(partition_on='segment_column')

# Train separate models for each segment
for segment in train['segment_column'].unique():
    segment_data = train[train['segment_column'] == segment]
    X_seg, y_seg = segment_data.drop('target', axis=1), segment_data['target']
    
    # Train model for this segment
    segment_model = XClassifier(
        max_depth=5,
        min_info_gain=0.01
    )
    segment_model.fit(X_seg, y_seg)
    
    # Add to partitioned model
    partitioned_model.add_partition(segment_model, segment)

# Predict with automatic segment routing
predictions = partitioned_model.predict(X_test)
```

### Surrogate Models

Explain black-box models with transparent surrogates:

```python
from xplainable.core.models import XSurrogateClassifier
from sklearn.ensemble import RandomForestClassifier

# Train black-box model
black_box = RandomForestClassifier()
black_box.fit(X_train, y_train)

# Create transparent surrogate
surrogate = XSurrogateClassifier(
    black_box_model=black_box,
    max_depth=5,
    min_info_gain=0.01
)

# Fit surrogate to explain black-box
surrogate.fit(X_train, y_train)

# Get explanations for black-box predictions
surrogate.explain(X_test)
```

## Hyperparameter Optimization

### Automatic Optimization

```python
from xplainable.core.optimisation.bayesian import XParamOptimiser

# Set up optimizer
optimizer = XParamOptimiser(
    n_trials=200,
    n_folds=5,
    early_stopping=40,
    objective='roc_auc'  # or 'f1', 'precision', 'recall', 'accuracy'
)

# Find optimal parameters
best_params = optimizer.optimise(X_train, y_train)

# Train optimized model
model = XClassifier(**best_params)
model.fit(X_train, y_train)
```

### Custom Search Spaces

```python
from hyperopt import hp

# Define custom search space
search_space = {
    'max_depth': hp.choice('max_depth', [3, 4, 5, 6, 7]),
    'min_info_gain': hp.uniform('min_info_gain', 0.001, 0.1),
    'weight': hp.uniform('weight', 0.1, 0.9),
    'power_degree': hp.choice('power_degree', [1, 2, 3])
}

# Optimize with custom space
optimizer = XParamOptimiser(
    n_trials=100,
    search_space=search_space
)
best_params = optimizer.optimise(X_train, y_train)
```

## Performance Metrics

### Built-in Evaluation

```python
# Accuracy score
accuracy = model.score(X_test, y_test)

# Detailed metrics
from xplainable.metrics import classification_metrics
metrics = classification_metrics(y_test, model.predict(X_test))

print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall: {metrics['recall']:.3f}")
print(f"F1-Score: {metrics['f1']:.3f}")
print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
```

### Custom Metrics

```python
from sklearn.metrics import classification_report, confusion_matrix

# Predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# Detailed classification report
print(classification_report(y_test, y_pred))

# Confusion matrix
print(confusion_matrix(y_test, y_pred))
```

## Visualization & Explainability

### Feature Importance

```python
# Global feature importance
importance = model.feature_importance()
print(importance.head())

# Plot feature importance
model.plot_feature_importance()
```

### Decision Explanations

```python
# Explain specific predictions
sample_explanation = model.explain(X_test.iloc[[0]])

# Waterfall plot showing decision breakdown
model.waterfall(X_test.iloc[[0]])

# Feature contribution analysis
contributions = model.feature_contributions(X_test)
```

### Model Visualization

```python
# Visualize decision trees for each feature
model.plot_trees()

# Show model architecture
model.plot_architecture()

# Performance curves
model.plot_performance_curves(X_test, y_test)
```

## Integration Examples

### Scikit-learn Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Create pipeline with xplainable model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', XClassifier())
])

# Fit pipeline
pipeline.fit(X_train, y_train)

# Predict with pipeline
predictions = pipeline.predict(X_test)
```

### Cross-validation

```python
from sklearn.model_selection import cross_val_score

# Cross-validation with XClassifier
scores = cross_val_score(
    XClassifier(), 
    X_train, 
    y_train, 
    cv=5, 
    scoring='roc_auc'
)

print(f"CV ROC-AUC: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
```

## Production Deployment

### Model Persistence

### Cloud Deployment

```python
from xplainable_client import Client

# Initialize client
client = Client(api_key="your-api-key")

# Deploy to cloud
model_id, version_id = client.create_model(
    model=model,
    model_name="Binary Classification Model",
    model_description="Transparent binary classifier",
    x=X_train,
    y=y_train
)

# Deploy as API
deployment = client.deploy(
    model_id=model_id,
    version_id=version_id,
    deployment_name="binary-classifier-api"
)
```

## Best Practices

### Data Preparation

:::tip Data Quality
- **Handle missing values** appropriately (XClassifier can handle NaN automatically)
- **Encode categorical variables** using preprocessing pipeline
- **Scale features** if using distance-based features
- **Remove highly correlated features** for better interpretability
:::

### Model Configuration

```python
# Recommended starting parameters
model = XClassifier(
    max_depth=5,          # Start conservative
    min_info_gain=0.01,   # Prevent overfitting
    min_leaf_size=10,     # Ensure statistical significance
    weight=0.5,           # Balanced activation
    map_calibration=True  # Better probability estimates
)
```

### Performance Monitoring

```python
# Monitor model performance over time
def monitor_model_performance(model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, predictions),
        'roc_auc': roc_auc_score(y_test, probabilities[:, 1]),
        'f1': f1_score(y_test, predictions)
    }
    
    return metrics

# Regular performance checks
performance = monitor_model_performance(model, X_test, y_test)
```

## Common Use Cases

### 🏦 Financial Services
- Credit scoring and risk assessment
- Fraud detection with explainable decisions
- Regulatory compliance (Basel III, GDPR)

### 🏥 Healthcare
- Clinical decision support
- Patient risk stratification
- Medical diagnosis assistance

### 🛒 E-commerce
- Customer churn prediction
- Product recommendation systems
- Marketing campaign optimization

### 🏭 Manufacturing
- Quality control and defect detection
- Predictive maintenance
- Process optimization

## Troubleshooting

### Common Issues

<details>
<summary><strong>Model not fitting properly</strong></summary>

**Possible causes:**
- Insufficient data for the complexity
- Highly imbalanced classes
- Poor feature quality

**Solutions:**
- Reduce `max_depth` or increase `min_leaf_size`
- Use class weights or resampling
- Improve feature engineering
</details>

<details>
<summary><strong>Poor probability calibration</strong></summary>

**Solutions:**
- Ensure `map_calibration=True`
- Use larger training dataset
- Consider probability calibration post-processing
</details>

<details>
<summary><strong>Slow training performance</strong></summary>

**Solutions:**
- Reduce `max_depth` parameter
- Increase `min_info_gain` threshold
- Use feature selection to reduce dimensionality
</details>

## Next Steps

:::note Ready to Explore?
- Try [multiclass classification](./classification-multiclass.md) for multi-label problems
- Explore [regression models](./regression.md) for continuous targets
- Learn about [preprocessing pipelines](./preprocessing.md) for data preparation
- Check out [advanced topics](../advanced-topics/) for optimization techniques
:::

&nbsp;

<!-- 
<BlogPost 
    imgUrl="https://images.unsplash.com/photo-1556155092-490a1ba16284?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=4140&q=80" 
    tag="Classification" 
    title="CUSTOMER CHURN PREDICTION" 
    description="Build a transparent customer churn model with real-time explanations." 
/>

<BlogPost 
    imgUrl="https://images.unsplash.com/photo-1556155092-490a1ba16284?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=4140&q=80" 
    tag="Finance" 
    title="CREDIT RISK ASSESSMENT" 
    description="Transparent credit scoring with regulatory compliance features." 
/> -->