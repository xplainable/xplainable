---
sidebar_position: 1
---

import BlogPost from "../../src/components/Cards/BlogPost.jsx";

# Partitioned Models

:::info Multi-Segment Modeling
**Partitioned models** enable training separate transparent models on different data segments, then combining them for improved accuracy and deeper insights. Perfect for datasets with natural groupings or heterogeneous patterns.
:::

## Overview

Partitioned models are a powerful technique for handling datasets where different segments exhibit distinct patterns. Instead of training one model on all data, partitioned models train specialized models for each segment, then intelligently route predictions to the appropriate model.

### Key Benefits

<div className="row">
  <div className="col col--6">
    <div className="card">
      <div className="card__header">
        <h3>🎯 Specialized Models</h3>
      </div>
      <div className="card__body">
        <p>Each segment gets a model optimized for its specific patterns and characteristics.</p>
      </div>
    </div>
  </div>
  <div className="col col--6">
    <div className="card">
      <div className="card__header">
        <h3>📈 Improved Accuracy</h3>
      </div>
      <div className="card__body">
        <p>Often outperforms single models by capturing segment-specific relationships.</p>
      </div>
    </div>
  </div>
</div>

<div className="row">
  <div className="col col--6">
    <div className="card">
      <div className="card__header">
        <h3>🔍 Deeper Insights</h3>
      </div>
      <div className="card__body">
        <p>Understand how different segments behave and what drives their outcomes.</p>
      </div>
    </div>
  </div>
  <div className="col col--6">
    <div className="card">
      <div className="card__header">
        <h3>🛡️ Robust Fallback</h3>
      </div>
      <div className="card__body">
        <p>Automatic fallback to a default model for unknown or new segments.</p>
      </div>
    </div>
  </div>
</div>

## How Partitioned Models Work

Partitioned models are **not ensemble models**. Instead of combining predictions from multiple models, they:

1. **Route data** to the appropriate model based on segment values
2. **Train specialized models** on homogeneous data subsets
3. **Maintain transparency** - each prediction comes from a single, explainable model
4. **Provide fallback** - handle unknown segments gracefully

:::tip When to Use Partitioned Models
- **Geographic segmentation** - Different regions have different patterns
- **Customer segments** - B2B vs B2C, different industries, etc.
- **Time-based segments** - Seasonal models, weekday vs weekend
- **Product categories** - Different products have different drivers
- **Heterogeneous data** - Mixed populations with distinct characteristics
:::

## Basic Implementation

### Classification Example

```python
from xplainable.core.models import XClassifier
import pandas as pd
from sklearn.model_selection import train_test_split

# Load your data
data = pd.read_csv('customer_data.csv')
train, test = train_test_split(data, test_size=0.2, random_state=42)

# Create partitioned model dictionary
partitioned_models = {}
partition_column = 'customer_segment'

# Train models for each segment
for segment in train[partition_column].unique():
    print(f"Training model for segment: {segment}")
    
    # Get segment data
    segment_data = train[train[partition_column] == segment]
    X_segment = segment_data.drop(['target', partition_column], axis=1)
    y_segment = segment_data['target']
    
    # Train specialized model
    model = XClassifier(
        max_depth=6,
        min_info_gain=0.01,
        weight=0.7
    )
    model.fit(X_segment, y_segment)
    
    # Store model
    partitioned_models[segment] = model

# Create default model for unknown segments
default_model = XClassifier(max_depth=5, min_info_gain=0.02)
default_model.fit(
    train.drop(['target', partition_column], axis=1),
    train['target']
)
partitioned_models['__dataset__'] = default_model

# Prediction function
def predict_partitioned(X, partition_values, models):
    """Make predictions using partitioned models."""
    predictions = []
    
    for i, partition_value in enumerate(partition_values):
        # Select appropriate model
        if partition_value in models:
            model = models[partition_value]
        else:
            model = models['__dataset__']
        
        # Make prediction
        pred = model.predict(X.iloc[[i]])
        predictions.append(pred[0])
    
    return predictions

# Make predictions
X_test = test.drop(['target', partition_column], axis=1)
y_test = test['target']
partition_test_values = test[partition_column].values

predictions = predict_partitioned(X_test, partition_test_values, partitioned_models)

# Evaluate
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions)
print(f"Partitioned model accuracy: {accuracy:.3f}")
```

### Regression Example

```python
from xplainable.core.models import XRegressor
import pandas as pd

# Load sales data with regional segments
data = pd.read_csv('sales_data.csv')

# Create partitioned regressor dictionary
partitioned_regressors = {}
partition_column = 'region'

# Train region-specific models
for region in data[partition_column].unique():
    region_data = data[data[partition_column] == region]
    X_region = region_data.drop(['sales', partition_column], axis=1)
    y_region = region_data['sales']
    
    # Customize model for region characteristics
    if region == 'urban':
        model = XRegressor(max_depth=7, min_info_gain=0.005)  # More complex
    else:
        model = XRegressor(max_depth=5, min_info_gain=0.02)   # Simpler
    
    model.fit(X_region, y_region)
    partitioned_regressors[region] = model

# Create default model
default_regressor = XRegressor(max_depth=6, min_info_gain=0.01)
default_regressor.fit(
    data.drop(['sales', partition_column], axis=1),
    data['sales']
)
partitioned_regressors['__dataset__'] = default_regressor

# Make predictions
def predict_partitioned_regression(X, partition_values, models):
    """Make regression predictions using partitioned models."""
    predictions = []
    
    for i, partition_value in enumerate(partition_values):
        model = models.get(partition_value, models['__dataset__'])
        pred = model.predict(X.iloc[[i]])
        predictions.append(pred[0])
    
    return predictions
```

## Advanced Partitioning Strategies

### Hierarchical Partitioning

```python
def create_hierarchical_partitions(data, hierarchy_columns):
    """Create hierarchical partitioned models."""
    partitioned_models = {}
    
    # Create hierarchical segments
    for level, column in enumerate(hierarchy_columns):
        if level == 0:
            # Top level partitioning
            for segment in data[column].unique():
                segment_data = data[data[column] == segment]
                
                model = XClassifier(
                    max_depth=6 - level,  # Reduce complexity at deeper levels
                    min_info_gain=0.01 * (level + 1)
                )
                
                X_segment = segment_data.drop(['target'] + hierarchy_columns, axis=1)
                y_segment = segment_data['target']
                model.fit(X_segment, y_segment)
                
                partitioned_models[segment] = model
        else:
            # Nested partitioning
            for parent_segment in data[hierarchy_columns[level-1]].unique():
                parent_data = data[data[hierarchy_columns[level-1]] == parent_segment]
                
                for child_segment in parent_data[column].unique():
                    segment_key = f"{parent_segment}_{child_segment}"
                    segment_data = parent_data[parent_data[column] == child_segment]
                    
                    if len(segment_data) > 50:  # Minimum segment size
                        model = XClassifier(
                            max_depth=6 - level,
                            min_info_gain=0.01 * (level + 1)
                        )
                        
                        X_segment = segment_data.drop(['target'] + hierarchy_columns, axis=1)
                        y_segment = segment_data['target']
                        model.fit(X_segment, y_segment)
                        
                        partitioned_models[segment_key] = model
    
    return partitioned_models

# Usage
hierarchy_columns = ['region', 'customer_type', 'product_category']
hierarchical_models = create_hierarchical_partitions(data, hierarchy_columns)
```

### Dynamic Partitioning

```python
def create_dynamic_partitions(data, partition_column, min_segment_size=100):
    """Create partitions dynamically based on data characteristics."""
    partitioned_models = {}
    
    # Analyze segment characteristics
    segment_stats = data.groupby(partition_column).agg({
        'target': ['count', 'mean', 'std'],
        data.columns[0]: 'count'  # Use first feature column for size
    }).round(3)
    
    print("Segment Analysis:")
    print(segment_stats)
    
    # Create models based on segment characteristics
    for segment in data[partition_column].unique():
        segment_data = data[data[partition_column] == segment]
        
        if len(segment_data) < min_segment_size:
            print(f"Skipping {segment}: insufficient data ({len(segment_data)} samples)")
            continue
        
        # Adjust model complexity based on segment size and variance
        segment_size = len(segment_data)
        target_variance = segment_data['target'].std()
        
        # More complex models for larger, more variable segments
        if segment_size > 500 and target_variance > 0.3:
            model_params = {'max_depth': 8, 'min_info_gain': 0.005}
        elif segment_size > 200:
            model_params = {'max_depth': 6, 'min_info_gain': 0.01}
        else:
            model_params = {'max_depth': 4, 'min_info_gain': 0.02}
        
        model = XClassifier(**model_params)
        
        X_segment = segment_data.drop(['target', partition_column], axis=1)
        y_segment = segment_data['target']
        model.fit(X_segment, y_segment)
        
        partitioned_models[segment] = model
        print(f"Created model for {segment}: {len(segment_data)} samples, "
              f"depth={model_params['max_depth']}")
    
    return partitioned_models
```

## Comprehensive Use Cases

### Geographic Segmentation

```python
def create_geographic_partitions(data):
    """Create geographically partitioned models."""
    geographic_models = {}
    
    # Define geographic segments
    geographic_mapping = {
        'North America': ['US', 'CA', 'MX'],
        'Europe': ['UK', 'DE', 'FR', 'IT', 'ES'],
        'Asia Pacific': ['JP', 'CN', 'IN', 'AU', 'SG'],
        'Latin America': ['BR', 'AR', 'CL', 'CO'],
        'Other': []  # Catch-all for other countries
    }
    
    # Create region column
    def map_country_to_region(country):
        for region, countries in geographic_mapping.items():
            if country in countries:
                return region
        return 'Other'
    
    data['region'] = data['country'].apply(map_country_to_region)
    
    # Train region-specific models
    for region in data['region'].unique():
        region_data = data[data['region'] == region]
        
        if len(region_data) < 50:
            continue
        
        # Adjust for regional characteristics
        if region == 'North America':
            # More complex model for mature market
            model = XClassifier(max_depth=7, min_info_gain=0.005, weight=0.8)
        elif region == 'Asia Pacific':
            # High variance market
            model = XClassifier(max_depth=6, min_info_gain=0.01, weight=0.6)
        else:
            # Standard model
            model = XClassifier(max_depth=5, min_info_gain=0.02, weight=0.7)
        
        X_region = region_data.drop(['target', 'region', 'country'], axis=1)
        y_region = region_data['target']
        model.fit(X_region, y_region)
        
        geographic_models[region] = model
    
    return geographic_models
```

### B2B vs B2C Segmentation

```python
def create_business_type_partitions(data):
    """Create B2B and B2C specific models."""
    business_models = {}
    
    # Define business type segments
    for business_type in ['B2B', 'B2C']:
        business_data = data[data['business_type'] == business_type]
        
        if business_type == 'B2B':
            # B2B models: Focus on relationship and contract features
            model = XClassifier(
                max_depth=8,  # More complex relationships
                min_info_gain=0.005,
                weight=0.9,  # Higher confidence in predictions
                features=['contract_length', 'account_size', 'industry', 
                         'decision_maker_level', 'previous_purchases']
            )
        else:
            # B2C models: Focus on individual behavior
            model = XClassifier(
                max_depth=6,  # Simpler individual patterns
                min_info_gain=0.01,
                weight=0.7,
                features=['age', 'income', 'purchase_history', 
                         'channel_preference', 'seasonal_behavior']
            )
        
        X_business = business_data.drop(['target', 'business_type'], axis=1)
        y_business = business_data['target']
        model.fit(X_business, y_business)
        
        business_models[business_type] = model
    
    return business_models
```

### Temporal Segmentation

```python
def create_temporal_partitions(data, date_column):
    """Create time-based partitioned models."""
    temporal_models = {}
    
    # Convert date column
    data[date_column] = pd.to_datetime(data[date_column])
    
    # Create temporal segments
    data['month'] = data[date_column].dt.month
    data['quarter'] = data[date_column].dt.quarter
    data['day_of_week'] = data[date_column].dt.dayofweek
    data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
    
    # Seasonal models
    seasonal_mapping = {
        'Spring': [3, 4, 5],
        'Summer': [6, 7, 8],
        'Fall': [9, 10, 11],
        'Winter': [12, 1, 2]
    }
    
    def get_season(month):
        for season, months in seasonal_mapping.items():
            if month in months:
                return season
        return 'Unknown'
    
    data['season'] = data['month'].apply(get_season)
    
    # Train seasonal models
    for season in data['season'].unique():
        season_data = data[data['season'] == season]
        
        if len(season_data) < 100:
            continue
        
        # Adjust for seasonal characteristics
        if season in ['Winter', 'Summer']:
            # More extreme seasonal patterns
            model = XClassifier(max_depth=7, min_info_gain=0.005)
        else:
            # Moderate seasonal patterns
            model = XClassifier(max_depth=5, min_info_gain=0.01)
        
        X_season = season_data.drop(['target', 'season', 'month', 'quarter', 
                                   'day_of_week', 'is_weekend', date_column], axis=1)
        y_season = season_data['target']
        model.fit(X_season, y_season)
        
        temporal_models[season] = model
    
    return temporal_models
```

## Performance Optimization

### Parallel Training

```python
import multiprocessing as mp
from functools import partial

def train_partition_model(args):
    """Train a single partition model."""
    segment, segment_data, model_params = args
    
    model = XClassifier(**model_params)
    X_segment = segment_data.drop(['target', 'partition_column'], axis=1)
    y_segment = segment_data['target']
    model.fit(X_segment, y_segment)
    
    return segment, model

def train_partitioned_models_parallel(data, partition_column, model_params=None):
    """Train partitioned models in parallel."""
    if model_params is None:
        model_params = {'max_depth': 6, 'min_info_gain': 0.01}
    
    # Prepare data for parallel processing
    training_args = []
    for segment in data[partition_column].unique():
        segment_data = data[data[partition_column] == segment].copy()
        segment_data['partition_column'] = segment_data[partition_column]
        training_args.append((segment, segment_data, model_params))
    
    # Train models in parallel
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(train_partition_model, training_args)
    
    # Collect results
    partitioned_models = dict(results)
    
    return partitioned_models
```

### Memory-Efficient Training

```python
def train_memory_efficient_partitions(data, partition_column, batch_size=1000):
    """Train partitioned models with memory efficiency."""
    partitioned_models = {}
    
    # Get unique segments
    segments = data[partition_column].unique()
    
    for segment in segments:
        print(f"Training model for segment: {segment}")
        
        # Create model
        model = XClassifier(max_depth=6, min_info_gain=0.01)
        
        # Get segment data in batches
        segment_mask = data[partition_column] == segment
        segment_data = data[segment_mask]
        
        if len(segment_data) <= batch_size:
            # Small segment - train normally
            X_segment = segment_data.drop(['target', partition_column], axis=1)
            y_segment = segment_data['target']
            model.fit(X_segment, y_segment)
        else:
            # Large segment - use batch training
            X_segment = segment_data.drop(['target', partition_column], axis=1)
            y_segment = segment_data['target']
            
            # Train on first batch
            first_batch = X_segment.iloc[:batch_size]
            first_batch_y = y_segment.iloc[:batch_size]
            model.fit(first_batch, first_batch_y)
            
            # Note: For true incremental learning, you would need
            # to implement partial_fit or use online learning techniques
            print(f"Trained on {len(segment_data)} samples for {segment}")
        
        partitioned_models[segment] = model
    
    return partitioned_models
```

## Model Comparison and Analysis

### Performance Comparison

```python
def compare_partitioned_vs_single_model(data, partition_column, test_size=0.2):
    """Compare partitioned models vs single model performance."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    
    # Split data
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
    
    # Train single model
    print("Training single model...")
    single_model = XClassifier(max_depth=6, min_info_gain=0.01)
    X_train_single = train_data.drop(['target', partition_column], axis=1)
    y_train_single = train_data['target']
    single_model.fit(X_train_single, y_train_single)
    
    # Train partitioned models
    print("Training partitioned models...")
    partitioned_models = {}
    for segment in train_data[partition_column].unique():
        segment_data = train_data[train_data[partition_column] == segment]
        
        if len(segment_data) < 20:  # Skip small segments
            continue
        
        model = XClassifier(max_depth=6, min_info_gain=0.01)
        X_segment = segment_data.drop(['target', partition_column], axis=1)
        y_segment = segment_data['target']
        model.fit(X_segment, y_segment)
        
        partitioned_models[segment] = model
    
    # Create default model for unknown segments
    default_model = XClassifier(max_depth=5, min_info_gain=0.02)
    default_model.fit(X_train_single, y_train_single)
    partitioned_models['__dataset__'] = default_model
    
    # Test single model
    X_test = test_data.drop(['target', partition_column], axis=1)
    y_test = test_data['target']
    single_predictions = single_model.predict(X_test)
    single_accuracy = accuracy_score(y_test, single_predictions)
    
    # Test partitioned models
    partitioned_predictions = []
    for i, row in test_data.iterrows():
        segment = row[partition_column]
        model = partitioned_models.get(segment, partitioned_models['__dataset__'])
        
        X_row = row.drop(['target', partition_column]).to_frame().T
        pred = model.predict(X_row)[0]
        partitioned_predictions.append(pred)
    
    partitioned_accuracy = accuracy_score(y_test, partitioned_predictions)
    
    # Results
    print(f"\nPerformance Comparison:")
    print(f"Single Model Accuracy: {single_accuracy:.4f}")
    print(f"Partitioned Models Accuracy: {partitioned_accuracy:.4f}")
    print(f"Improvement: {partitioned_accuracy - single_accuracy:.4f}")
    
    # Detailed analysis by segment
    print(f"\nSegment-wise Performance:")
    for segment in test_data[partition_column].unique():
        segment_test = test_data[test_data[partition_column] == segment]
        if len(segment_test) == 0:
            continue
        
        X_segment_test = segment_test.drop(['target', partition_column], axis=1)
        y_segment_test = segment_test['target']
        
        # Single model on segment
        single_segment_pred = single_model.predict(X_segment_test)
        single_segment_acc = accuracy_score(y_segment_test, single_segment_pred)
        
        # Partitioned model on segment
        if segment in partitioned_models:
            partition_model = partitioned_models[segment]
            partition_segment_pred = partition_model.predict(X_segment_test)
            partition_segment_acc = accuracy_score(y_segment_test, partition_segment_pred)
        else:
            partition_segment_acc = single_segment_acc  # Use default
        
        print(f"  {segment}: Single={single_segment_acc:.3f}, "
              f"Partitioned={partition_segment_acc:.3f}, "
              f"Samples={len(segment_test)}")
    
    return {
        'single_model': single_model,
        'partitioned_models': partitioned_models,
        'single_accuracy': single_accuracy,
        'partitioned_accuracy': partitioned_accuracy,
        'improvement': partitioned_accuracy - single_accuracy
    }
```

### Feature Importance Analysis

```python
def analyze_partition_feature_importance(partitioned_models, feature_names):
    """Analyze feature importance across partitions."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Collect feature importance from all models
    importance_data = {}
    
    for partition, model in partitioned_models.items():
        if partition == '__dataset__':
            continue
        
        try:
            importance = model.feature_importance()
            importance_data[partition] = importance
        except:
            print(f"Could not get feature importance for {partition}")
    
    # Create DataFrame for analysis
    importance_df = pd.DataFrame(importance_data).fillna(0)
    
    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(importance_df, annot=True, cmap='YlOrRd', fmt='.3f')
    plt.title('Feature Importance Across Partitions')
    plt.xlabel('Partitions')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()
    
    # Find partition-specific important features
    print("Top Features by Partition:")
    for partition in importance_df.columns:
        top_features = importance_df[partition].nlargest(3)
        print(f"\n{partition}:")
        for feature, importance in top_features.items():
            print(f"  {feature}: {importance:.3f}")
    
    # Find features with high variance across partitions
    feature_variance = importance_df.var(axis=1).sort_values(ascending=False)
    print(f"\nFeatures with Highest Variance Across Partitions:")
    for feature, variance in feature_variance.head(5).items():
        print(f"  {feature}: {variance:.3f}")
    
    return importance_df
```

## Best Practices

### Partition Selection

:::tip Choosing Good Partitions
1. **Business Logic**: Partitions should make business sense
2. **Sufficient Data**: Each partition needs enough samples (typically 100+)
3. **Distinct Patterns**: Segments should have different relationships
4. **Stability**: Partition values should be consistent over time
5. **Interpretability**: Partitions should be explainable to stakeholders
:::

### Model Monitoring

```python
def monitor_partition_performance(partitioned_models, new_data, partition_column):
    """Monitor performance of partitioned models over time."""
    monitoring_results = {}
    
    for segment in new_data[partition_column].unique():
        segment_data = new_data[new_data[partition_column] == segment]
        
        if segment in partitioned_models:
            model = partitioned_models[segment]
            X_segment = segment_data.drop(['target', partition_column], axis=1)
            y_segment = segment_data['target']
            
            # Calculate performance metrics
            predictions = model.predict(X_segment)
            accuracy = accuracy_score(y_segment, predictions)
            
            monitoring_results[segment] = {
                'accuracy': accuracy,
                'sample_count': len(segment_data),
                'model_exists': True
            }
        else:
            monitoring_results[segment] = {
                'accuracy': None,
                'sample_count': len(segment_data),
                'model_exists': False
            }
    
    # Alert for performance degradation
    for segment, results in monitoring_results.items():
        if results['model_exists'] and results['accuracy'] < 0.7:
            print(f"⚠️  Performance alert for {segment}: {results['accuracy']:.3f}")
        elif not results['model_exists'] and results['sample_count'] > 50:
            print(f"🔄 New segment detected: {segment} ({results['sample_count']} samples)")
    
    return monitoring_results
```

## Integration with Other Features

### Combining with Rapid Refitting

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

# Usage
updated_partitioned_models = rapid_refit_partitioned_models(
    partitioned_models,
    weight=0.8,
    power_degree=2.0
)
```

### Cloud Integration

```python
def deploy_partitioned_models_to_cloud(partitioned_models, client, model_name):
    """Deploy partitioned models to xplainable cloud."""
    deployment_results = {}
    
    for partition, model in partitioned_models.items():
        try:
            # Create cloud model
            model_id = client.create_model(
                model=model,
                model_name=f"{model_name}_{partition}",
                model_description=f"Partition model for {partition}"
            )
            
            deployment_results[partition] = {
                'model_id': model_id,
                'status': 'deployed'
            }
            
            print(f"✅ Deployed {partition} model: {model_id}")
            
        except Exception as e:
            print(f"❌ Failed to deploy {partition}: {e}")
            deployment_results[partition] = {
                'model_id': None,
                'status': 'failed',
                'error': str(e)
            }
    
    return deployment_results
```

## Next Steps

:::note Ready for More Advanced Topics?
- Explore [rapid refitting](./rapid-refitting.md) for real-time partition optimization
- Learn about [XEvolutionaryNetwork](./XEvolutionaryNetwork.md) for advanced optimization
- Check out [custom transformers](./custom-transformers.md) for partition-specific preprocessing
:::

Partitioned models are a powerful technique for handling heterogeneous data while maintaining the transparency and interpretability that makes xplainable unique. By training specialized models for different segments, you can achieve better performance and gain deeper insights into your data's underlying patterns.