---
sidebar_position: 3
---

# Regression

:::info Transparent Regression
**XRegressor** provides transparent regression modeling with real-time explainability. Unlike black-box models, you get instant insights into how predictions are made for continuous target variables.
:::

## Overview

The `XRegressor` is xplainable's transparent regression model that uses the same feature-wise ensemble approach as `XClassifier`, but optimized for continuous target variables. It provides complete transparency while maintaining competitive performance with traditional regression models.

### Key Features

<div className="row">
  <div className="col col--6">
    <div className="card">
      <div className="card__header">
        <h3>🔍 Real-time Explainability</h3>
      </div>
      <div className="card__body">
        <p>Get instant explanations for regression predictions without needing post-hoc analysis.</p>
      </div>
    </div>
  </div>
  <div className="col col--6">
    <div className="card">
      <div className="card__header">
        <h3>⚡ Rapid Refitting</h3>
      </div>
      <div className="card__body">
        <p>Update model parameters on individual features without complete retraining.</p>
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
        <p>Each feature contributes through its own decision tree, providing granular insights.</p>
      </div>
    </div>
  </div>
  <div className="col col--6">
    <div className="card">
      <div className="card__header">
        <h3>📊 Prediction Bounds</h3>
      </div>
      <div className="card__body">
        <p>Built-in prediction range constraints for realistic and bounded outputs.</p>
      </div>
    </div>
  </div>
</div>

## Quick Start

### GUI Interface

Training an `XRegressor` with the embedded GUI:

```python
import xplainable as xp
import pandas as pd

# Load your data
data = pd.read_csv('data.csv')

# Train your model (opens embedded GUI)
model = xp.regressor(data)
```

:::tip GUI Benefits
The regression GUI provides:
- Interactive hyperparameter tuning
- Real-time performance metrics (RMSE, MAE, R²)
- Visual prediction vs actual plots
- Feature importance analysis
:::

### Python API

For programmatic control:

```python
from xplainable.core.models import XRegressor
from sklearn.model_selection import train_test_split
import pandas as pd

# Load and prepare data
data = pd.read_csv('data.csv')
X, y = data.drop('target', axis=1), data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = XRegressor()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

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

### Regression-specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prediction_range` | tuple | None | Min/max bounds for predictions |
| `tail_sensitivity` | float | 0.5 | Weight for divisive leaf nodes |
| `ignore_nan` | bool | True | Handle missing values automatically |

### Example with Parameters

```python
model = XRegressor(
    max_depth=7,
    min_info_gain=0.005,
    min_leaf_size=10,
    weight=0.7,
    power_degree=2,
    sigmoid_exponent=1,
    prediction_range=(0, 1000),  # Bound predictions between 0 and 1000
    tail_sensitivity=0.3,
    ignore_nan=True
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
# Predictions
predictions = model.predict(X_test)

# Single sample prediction
single_pred = model.predict(X_test.iloc[[0]])

# Prediction intervals (if supported)
pred_intervals = model.predict_intervals(X_test, confidence=0.95)
```

### Explanation Methods

```python
# Global explanations
model.explain()

# Feature importance
importance = model.feature_importance()

# Local explanations for specific samples
model.explain(X_test.iloc[[0]])

# Waterfall plot for prediction breakdown
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

# Model performance metrics
metrics = model.score(X_test, y_test)
```

## Advanced Usage

### Rapid Refitting

Update model parameters without complete retraining:

```python
# Initial training
model = XRegressor()
model.fit(X_train, y_train)

# Update parameters rapidly
model.refit(
    max_depth=7,
    weight=0.8,
    features=['feature1', 'feature2']  # Only update specific features
)

# Performance comparison
print(f"Original RMSE: {model.score(X_test, y_test)}")
```

:::tip Rapid Refitting for Regression
- **Feature-specific tuning** for different variable types
- **Real-time model adjustment** based on new data patterns
- **A/B testing** different parameter configurations
- **Production model updates** without service interruption
:::

### Partitioned Regression

For datasets with natural segments:

```python
from xplainable.core.models import PartitionedRegressor, XRegressor

# Create partitioned model
partitioned_model = PartitionedRegressor(partition_on='segment_column')

# Train separate models for each segment
for segment in train['segment_column'].unique():
    segment_data = train[train['segment_column'] == segment]
    X_seg, y_seg = segment_data.drop('target', axis=1), segment_data['target']
    
    # Train model for this segment
    segment_model = XRegressor(
        max_depth=5,
        min_info_gain=0.01,
        prediction_range=(0, 1000)
    )
    segment_model.fit(X_seg, y_seg)
    
    # Add to partitioned model
    partitioned_model.add_partition(segment_model, segment)

# Predict with automatic segment routing
predictions = partitioned_model.predict(X_test)
```

### Surrogate Regression

Explain black-box regression models:

```python
from xplainable.core.models import XSurrogateRegressor
from sklearn.ensemble import RandomForestRegressor

# Train black-box model
black_box = RandomForestRegressor()
black_box.fit(X_train, y_train)

# Create transparent surrogate
surrogate = XSurrogateRegressor(
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

# Set up optimizer for regression
optimizer = XParamOptimiser(
    n_trials=200,
    n_folds=5,
    early_stopping=40,
    objective='rmse'  # or 'mae', 'r2'
)

# Find optimal parameters
best_params = optimizer.optimise(X_train, y_train)

# Train optimized model
model = XRegressor(**best_params)
model.fit(X_train, y_train)
```

### Advanced Optimization with Tighten

XRegressor supports the unique "Tighten" algorithm for leaf boosting:

```python
from xplainable.core.optimisation import Tighten

# Apply tighten algorithm
tighten = Tighten(
    model=model,
    X=X_train,
    y=y_train,
    iterations=10,
    learning_rate=0.1
)

# Optimize model
optimized_model = tighten.fit()

# Compare performance
print(f"Original RMSE: {model.score(X_test, y_test)}")
print(f"Tightened RMSE: {optimized_model.score(X_test, y_test)}")
```

## Performance Metrics

### Built-in Evaluation

```python
# R² score (default)
r2_score = model.score(X_test, y_test)

# Detailed metrics
from xplainable.metrics import regression_metrics
metrics = regression_metrics(y_test, model.predict(X_test))

print(f"R² Score: {metrics['r2']:.3f}")
print(f"RMSE: {metrics['rmse']:.3f}")
print(f"MAE: {metrics['mae']:.3f}")
print(f"MAPE: {metrics['mape']:.3f}")
```

### Custom Metrics

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Predictions
y_pred = model.predict(X_test)

# Calculate metrics
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.3f}")
print(f"MAE: {mae:.3f}")
print(f"R² Score: {r2:.3f}")
```

## Visualization & Explainability

### Prediction Analysis

```python
# Prediction vs actual plot
model.plot_predictions(X_test, y_test)

# Residual analysis
model.plot_residuals(X_test, y_test)

# Feature importance
model.plot_feature_importance()
```

### Feature Contributions

```python
# Global feature importance
importance = model.feature_importance()
print(importance.head())

# Feature contributions for specific predictions
contributions = model.feature_contributions(X_test)

# Waterfall plot for individual predictions
model.waterfall(X_test.iloc[[0]])
```

### Model Diagnostics

```python
# Model diagnostic plots
model.plot_diagnostics(X_test, y_test)

# Learning curves
model.plot_learning_curves(X_train, y_train)

# Prediction intervals
model.plot_prediction_intervals(X_test, y_test)
```

## Integration Examples

### Scikit-learn Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Create pipeline with xplainable model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', XRegressor())
])

# Fit pipeline
pipeline.fit(X_train, y_train)

# Predict with pipeline
predictions = pipeline.predict(X_test)
```

### Cross-validation

```python
from sklearn.model_selection import cross_val_score

# Cross-validation with XRegressor
scores = cross_val_score(
    XRegressor(), 
    X_train, 
    y_train, 
    cv=5, 
    scoring='neg_mean_squared_error'
)

print(f"CV RMSE: {(-scores).mean():.3f} (+/- {scores.std() * 2:.3f})")
```

## Production Deployment

### Cloud Deployment

```python
from xplainable_client import Client

# Initialize client
client = Client(api_key="your-api-key")

# Deploy to cloud
model_id, version_id = client.create_model(
    model=model,
    model_name="House Price Prediction",
    model_description="Transparent regression model for house prices",
    x=X_train,
    y=y_train
)

# Deploy as API
deployment = client.deploy(
    model_id=model_id,
    version_id=version_id,
    deployment_name="house-price-api"
)
```

## Best Practices

### Data Preparation

:::tip Regression Data Quality
- **Handle outliers** carefully (they significantly impact regression)
- **Scale features** appropriately for better convergence
- **Check for multicollinearity** between features
- **Validate prediction ranges** are realistic
:::

### Model Configuration

```python
# Recommended starting parameters for regression
model = XRegressor(
    max_depth=5,              # Start conservative
    min_info_gain=0.01,       # Prevent overfitting
    min_leaf_size=10,         # Ensure statistical significance
    weight=0.5,               # Balanced activation
    prediction_range=(0, 100) # Set realistic bounds
)
```

### Performance Monitoring

```python
# Monitor regression model performance
def monitor_regression_performance(model, X_test, y_test):
    predictions = model.predict(X_test)
    
    metrics = {
        'rmse': mean_squared_error(y_test, predictions, squared=False),
        'mae': mean_absolute_error(y_test, predictions),
        'r2': r2_score(y_test, predictions)
    }
    
    return metrics

# Regular performance checks
performance = monitor_regression_performance(model, X_test, y_test)
```

## Common Use Cases

### 🏠 Real Estate
- House price prediction with market explanations
- Property valuation with transparent factors
- Rental price estimation

### 💰 Finance
- Credit risk scoring with regulatory compliance
- Portfolio optimization with explainable returns
- Loan amount determination

### 🏭 Manufacturing
- Quality control with process explanations
- Predictive maintenance scheduling
- Production optimization

### 📊 Business Analytics
- Sales forecasting with driver analysis
- Customer lifetime value prediction
- Revenue optimization

## Troubleshooting

### Common Issues

<details>
<summary><strong>Poor prediction accuracy</strong></summary>

**Possible causes:**
- Insufficient model complexity
- Poor feature engineering
- Outliers in target variable

**Solutions:**
- Increase `max_depth` or decrease `min_info_gain`
- Add feature interactions or transformations
- Handle outliers before training
</details>

<details>
<summary><strong>Predictions outside expected range</strong></summary>

**Solutions:**
- Set `prediction_range` parameter
- Check for data leakage
- Validate input data quality
</details>

<details>
<summary><strong>Model overfitting</strong></summary>

**Solutions:**
- Increase `min_leaf_size` parameter
- Decrease `max_depth`
- Use cross-validation for parameter tuning
- Add regularization through `min_info_gain`
</details>

## Next Steps

:::note Ready to Explore More?
- Try [binary classification](./classification-binary.md) for categorical targets
- Explore [multiclass classification](./classification-multiclass.md) for multiple categories
- Learn about [preprocessing pipelines](./preprocessing.md) for data preparation
- Check out [advanced topics](../advanced-topics/) for optimization techniques
:::

## Complete Example: House Price Prediction

```python
import xplainable as xp
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load data
data = pd.read_csv('house_prices.csv')

# Prepare features and target
X = data.drop('price', axis=1)
y = data['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create preprocessing pipeline
pipeline = xp.XPipeline()
pipeline.add_transformer(xp.FillMissing(strategy='median'))
pipeline.add_transformer(xp.OneHotEncode(columns=['neighborhood', 'house_type']))
pipeline.add_transformer(xp.LogTransform(columns=['lot_size']))
pipeline.add_transformer(xp.MinMaxScale())

# Fit pipeline
pipeline.fit(X_train)

# Transform data
X_train_processed = pipeline.transform(X_train)
X_test_processed = pipeline.transform(X_test)

# Train model with realistic price bounds
model = xp.XRegressor(
    max_depth=6,
    min_info_gain=0.005,
    prediction_range=(50000, 2000000),  # Realistic house price range
    weight=0.7
)

model.fit(X_train_processed, y_train)

# Make predictions
y_pred = model.predict(X_test_processed)

# Evaluate performance
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: ${rmse:,.2f}")
print(f"R² Score: {r2:.3f}")

# Get explanations
model.explain()

# Feature importance
importance = model.feature_importance()
print("\nTop 5 Most Important Features:")
print(importance.head())
```