---
sidebar_position: 5
---

# Preprocessing

:::info Data Preprocessing Pipeline
**XPipeline** provides a comprehensive preprocessing framework with 15+ transformers designed for transparent machine learning workflows. Build reusable, interpretable preprocessing pipelines that integrate seamlessly with xplainable models.
:::

## Overview

Xplainable's preprocessing system is built around the `XPipeline` class, which chains together multiple transformers to create comprehensive data preprocessing workflows. Unlike traditional preprocessing approaches, xplainable's system is designed with transparency and interpretability in mind.

### Key Features

<div className="row">
  <div className="col col--6">
    <div className="card">
      <div className="card__header">
        <h3>🔗 Pipeline Chaining</h3>
      </div>
      <div className="card__body">
        <p>Chain multiple transformers together in a single pipeline for reproducible preprocessing.</p>
      </div>
    </div>
  </div>
  <div className="col col--6">
    <div className="card">
      <div className="card__header">
        <h3>🎯 Feature-level Control</h3>
      </div>
      <div className="card__body">
        <p>Apply transformations at both feature and dataset levels with granular control.</p>
      </div>
    </div>
  </div>
</div>

<div className="row">
  <div className="col col--6">
    <div className="card">
      <div className="card__header">
        <h3>💾 Pipeline Persistence</h3>
      </div>
      <div className="card__body">
        <p>Save and load preprocessing pipelines for consistent data transformation across environments.</p>
      </div>
    </div>
  </div>
  <div className="col col--6">
    <div className="card">
      <div className="card__header">
        <h3>🔍 Transparent Transformations</h3>
      </div>
      <div className="card__body">
        <p>All transformations are interpretable and can be inspected for better understanding.</p>
      </div>
    </div>
  </div>
</div>

## Quick Start

### Basic Pipeline

```python
import xplainable as xp
import pandas as pd

# Load data
data = pd.read_csv('data.csv')

# Create pipeline
pipeline = xp.XPipeline()

# Add transformers
pipeline.add_transformer(xp.FillMissing())
pipeline.add_transformer(xp.OneHotEncode())
pipeline.add_transformer(xp.MinMaxScale())

# Fit and transform
pipeline.fit(data)
transformed_data = pipeline.transform(data)
```

### GUI Interface

```python
import xplainable as xp

# Load data
data = pd.read_csv('data.csv')

# Open preprocessing GUI
preprocessor = xp.Preprocessor(data)
```

:::tip GUI Benefits
The preprocessing GUI provides:
- Interactive transformer selection
- Real-time data preview
- Visual impact assessment
- Automatic pipeline generation
:::

## XPipeline Class

### Basic Usage

```python
from xplainable.preprocessing import XPipeline

# Create pipeline
pipeline = XPipeline()

# Add transformers in sequence
pipeline.add_transformer(transformer1)
pipeline.add_transformer(transformer2)
pipeline.add_transformer(transformer3)

# Fit pipeline to data
pipeline.fit(X_train)

# Transform data
X_train_transformed = pipeline.transform(X_train)
X_test_transformed = pipeline.transform(X_test)
```

### Pipeline Methods

| Method | Description | Usage |
|--------|-------------|-------|
| `add_transformer()` | Add transformer to pipeline | `pipeline.add_transformer(transformer)` |
| `fit()` | Fit pipeline to training data | `pipeline.fit(X_train)` |
| `transform()` | Transform data using fitted pipeline | `X_transformed = pipeline.transform(X)` |
| `fit_transform()` | Fit and transform in one step | `X_transformed = pipeline.fit_transform(X)` |
| `inverse_transform()` | Reverse transformations where possible | `X_original = pipeline.inverse_transform(X)` |

### Pipeline Inspection

```python
# View pipeline steps
print(pipeline.steps)

# Get transformer at specific step
transformer = pipeline.get_transformer(step_index=0)

# View pipeline summary
pipeline.summary()

# Get feature names after transformation
feature_names = pipeline.get_feature_names()
```

## Available Transformers

### Dataset-level Transformers

#### DropCols
Remove unwanted columns from the dataset.

```python
from xplainable.preprocessing import DropCols

# Drop specific columns
drop_cols = DropCols(columns=['unwanted_col1', 'unwanted_col2'])

# Drop columns by pattern
drop_cols = DropCols(pattern='temp_*')

# Drop columns with high missing values
drop_cols = DropCols(missing_threshold=0.5)
```

#### DropNaNs
Remove rows with missing values.

```python
from xplainable.preprocessing import DropNaNs

# Drop rows with any missing values
drop_nans = DropNaNs()

# Drop rows with missing values in specific columns
drop_nans = DropNaNs(columns=['important_col1', 'important_col2'])

# Drop rows with missing values above threshold
drop_nans = DropNaNs(threshold=0.3)
```

#### FillMissing
Fill missing values with various strategies.

```python
from xplainable.preprocessing import FillMissing

# Fill with mean/mode (automatic detection)
fill_missing = FillMissing()

# Fill with specific strategy
fill_missing = FillMissing(strategy='mean')  # 'mean', 'median', 'mode', 'constant'

# Fill with constant value
fill_missing = FillMissing(strategy='constant', fill_value=0)

# Fill specific columns
fill_missing = FillMissing(columns=['col1', 'col2'], strategy='median')
```

#### Operation
Apply mathematical operations to create new features.

```python
from xplainable.preprocessing import Operation

# Create new feature from existing ones
operation = Operation(
    new_column='total_amount',
    operation='col1 + col2'
)

# Multiple operations
operation = Operation(
    operations={
        'total_amount': 'price * quantity',
        'price_per_unit': 'total_cost / quantity',
        'discount_rate': '(original_price - final_price) / original_price'
    }
)
```

### Numeric Transformers

#### MinMaxScale
Scale numeric features to a specified range.

```python
from xplainable.preprocessing import MinMaxScale

# Scale to [0, 1] range
scaler = MinMaxScale()

# Scale to custom range
scaler = MinMaxScale(feature_range=(-1, 1))

# Scale specific columns
scaler = MinMaxScale(columns=['feature1', 'feature2'])
```

#### LogTransform
Apply logarithmic transformation to numeric features.

```python
from xplainable.preprocessing import LogTransform

# Natural log transformation
log_transform = LogTransform()

# Log base 10
log_transform = LogTransform(base=10)

# Handle zero values
log_transform = LogTransform(handle_zeros=True, zero_replacement=1e-8)
```

#### Clip
Clip values to specified bounds.

```python
from xplainable.preprocessing import Clip

# Clip outliers using percentiles
clip = Clip(lower_percentile=5, upper_percentile=95)

# Clip to specific values
clip = Clip(lower_bound=0, upper_bound=100)

# Clip specific columns
clip = Clip(columns=['feature1', 'feature2'], lower_bound=0)
```

### Categorical Transformers

#### DetectCategories
Automatically detect and mark categorical columns.

```python
from xplainable.preprocessing import DetectCategories

# Auto-detect categorical columns
detect_cats = DetectCategories()

# Set threshold for categorical detection
detect_cats = DetectCategories(threshold=10)  # Columns with ≤10 unique values

# Force specific columns as categorical
detect_cats = DetectCategories(force_categorical=['status', 'category'])
```

#### OneHotEncode
Convert categorical variables to one-hot encoded format.

```python
from xplainable.preprocessing import OneHotEncode

# One-hot encode all categorical columns
one_hot = OneHotEncode()

# Encode specific columns
one_hot = OneHotEncode(columns=['category', 'status'])

# Handle unknown categories
one_hot = OneHotEncode(handle_unknown='ignore')

# Drop first category to avoid multicollinearity
one_hot = OneHotEncode(drop_first=True)
```

#### LabelEncode
Convert categorical variables to numeric labels.

```python
from xplainable.preprocessing import LabelEncode

# Label encode categorical columns
label_encode = LabelEncode()

# Encode specific columns
label_encode = LabelEncode(columns=['category', 'status'])

# Custom label mapping
label_encode = LabelEncode(
    mapping={'category': {'A': 0, 'B': 1, 'C': 2}}
)
```

### Time Series Transformers

#### DateTimeExtract
Extract features from datetime columns.

```python
from xplainable.preprocessing import DateTimeExtract

# Extract common datetime features
datetime_extract = DateTimeExtract(
    column='timestamp',
    features=['year', 'month', 'day', 'hour', 'dayofweek']
)

# Extract custom features
datetime_extract = DateTimeExtract(
    column='date',
    features=['quarter', 'is_weekend', 'is_month_end']
)
```

#### RollingOperation
Apply rolling window operations.

```python
from xplainable.preprocessing import RollingOperation

# Rolling mean
rolling_mean = RollingOperation(
    column='sales',
    operation='mean',
    window=7,
    new_column='sales_7day_avg'
)

# Multiple rolling operations
rolling_ops = RollingOperation(
    operations={
        'sales_mean_7d': {'column': 'sales', 'operation': 'mean', 'window': 7},
        'sales_std_7d': {'column': 'sales', 'operation': 'std', 'window': 7}
    }
)
```

#### GroupbyShift
Create lag features grouped by categories.

```python
from xplainable.preprocessing import GroupbyShift

# Create lag features
groupby_shift = GroupbyShift(
    group_by='customer_id',
    column='purchase_amount',
    periods=[1, 2, 3],
    new_columns=['prev_purchase', 'prev_purchase_2', 'prev_purchase_3']
)
```

## Advanced Usage

### Custom Transformers

Create your own transformers by inheriting from `XBaseTransformer`:

```python
from xplainable.preprocessing import XBaseTransformer

class CustomTransformer(XBaseTransformer):
    def __init__(self, parameter1=None, parameter2=None):
        super().__init__()
        self.parameter1 = parameter1
        self.parameter2 = parameter2
    
    def fit(self, X, y=None):
        # Fit logic here
        self.fitted_attribute_ = X.mean()  # Example
        return self
    
    def transform(self, X):
        # Transform logic here
        X_transformed = X.copy()
        # Apply your transformation
        return X_transformed
    
    def inverse_transform(self, X):
        # Reverse transformation if possible
        return X

# Use custom transformer
custom_transformer = CustomTransformer(parameter1=0.5)
pipeline.add_transformer(custom_transformer)
```

### Conditional Transformations

Apply transformations based on conditions:

```python
from xplainable.preprocessing import ConditionalTransformer

# Apply transformation only to specific data subsets
conditional = ConditionalTransformer(
    condition=lambda x: x['category'] == 'A',
    transformer=MinMaxScale(),
    columns=['feature1', 'feature2']
)
```

### Feature-level Transformations

Apply different transformations to different features:

```python
# Create feature-specific pipeline
pipeline = XPipeline()

# Numeric features
pipeline.add_transformer(MinMaxScale(columns=['numeric_col1', 'numeric_col2']))
pipeline.add_transformer(LogTransform(columns=['skewed_col']))

# Categorical features
pipeline.add_transformer(OneHotEncode(columns=['cat_col1', 'cat_col2']))
pipeline.add_transformer(LabelEncode(columns=['ordinal_col']))

# Text features (if applicable)
pipeline.add_transformer(TextVectorizer(columns=['text_col']))
```

## Pipeline Persistence

### Save and Load Pipelines

```python
import joblib

# Save pipeline
joblib.dump(pipeline, 'preprocessing_pipeline.pkl')

# Load pipeline
loaded_pipeline = joblib.load('preprocessing_pipeline.pkl')

# Use loaded pipeline
X_transformed = loaded_pipeline.transform(X_new)
```

### Cloud Integration

```python
from xplainable_client import Client

# Initialize client
client = Client(api_key="your-api-key")

# Save pipeline to cloud
preprocessor_id = client.create_preprocessor(
    preprocessor=pipeline,
    preprocessor_name="Customer Data Pipeline",
    preprocessor_description="Standard preprocessing for customer data"
)

# Load pipeline from cloud
cloud_pipeline = client.load_preprocessor(preprocessor_id)
```

## Data Quality Checks

### Built-in Quality Scanning

```python
from xplainable.quality import DataQualityScanner

# Scan data quality
scanner = DataQualityScanner()
quality_report = scanner.scan(data)

# View quality issues
print(quality_report.summary())

# Get recommendations
recommendations = quality_report.get_recommendations()
```

### Custom Quality Checks

```python
# Add custom quality checks to pipeline
from xplainable.preprocessing import QualityCheck

quality_check = QualityCheck(
    checks=[
        'missing_values',
        'duplicate_rows',
        'outliers',
        'data_types'
    ]
)

pipeline.add_transformer(quality_check)
```

## Integration Examples

### With XClassifier

```python
import xplainable as xp
from xplainable.preprocessing import XPipeline

# Create preprocessing pipeline
pipeline = XPipeline()
pipeline.add_transformer(xp.FillMissing())
pipeline.add_transformer(xp.OneHotEncode())
pipeline.add_transformer(xp.MinMaxScale())

# Fit pipeline
pipeline.fit(X_train)

# Transform data
X_train_processed = pipeline.transform(X_train)
X_test_processed = pipeline.transform(X_test)

# Train model on processed data
model = xp.XClassifier()
model.fit(X_train_processed, y_train)

# Predict
predictions = model.predict(X_test_processed)
```

### With Scikit-learn

```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# Create combined pipeline
combined_pipeline = Pipeline([
    ('preprocessing', pipeline),
    ('classifier', xp.XClassifier())
])

# Cross-validation
scores = cross_val_score(combined_pipeline, X, y, cv=5)
```

## Best Practices

### Pipeline Design

:::tip Design Principles
1. **Order matters** - Apply transformations in logical sequence
2. **Fit on training data only** - Avoid data leakage
3. **Handle missing values early** - Before other transformations
4. **Scale after encoding** - Categorical encoding first, then scaling
5. **Document transformations** - Keep track of what each step does
:::

### Recommended Order

```python
# Recommended transformation order
pipeline = XPipeline()

# 1. Data cleaning
pipeline.add_transformer(DropCols(columns=['irrelevant_col']))
pipeline.add_transformer(DropNaNs(threshold=0.8))

# 2. Missing value handling
pipeline.add_transformer(FillMissing(strategy='mean'))

# 3. Feature engineering
pipeline.add_transformer(Operation(operations={'new_feature': 'col1 * col2'}))
pipeline.add_transformer(DateTimeExtract(column='date'))

# 4. Categorical encoding
pipeline.add_transformer(DetectCategories())
pipeline.add_transformer(OneHotEncode())

# 5. Numeric transformations
pipeline.add_transformer(LogTransform(columns=['skewed_col']))
pipeline.add_transformer(Clip(lower_percentile=1, upper_percentile=99))

# 6. Scaling (last step)
pipeline.add_transformer(MinMaxScale())
```

### Performance Optimization

```python
# Cache intermediate results for large datasets
pipeline = XPipeline(cache_intermediate=True)

# Parallel processing for independent transformations
pipeline = XPipeline(n_jobs=-1)

# Memory-efficient transformations
pipeline = XPipeline(memory_efficient=True)
```

## Common Use Cases

### 🏦 Financial Data
```python
# Financial data preprocessing
financial_pipeline = XPipeline()
financial_pipeline.add_transformer(FillMissing(strategy='median'))
financial_pipeline.add_transformer(LogTransform(columns=['income', 'loan_amount']))
financial_pipeline.add_transformer(Clip(lower_percentile=1, upper_percentile=99))
financial_pipeline.add_transformer(OneHotEncode(columns=['employment_type']))
financial_pipeline.add_transformer(MinMaxScale())
```

### 🛒 E-commerce Data
```python
# E-commerce preprocessing
ecommerce_pipeline = XPipeline()
ecommerce_pipeline.add_transformer(DateTimeExtract(column='purchase_date'))
ecommerce_pipeline.add_transformer(RollingOperation(
    column='purchase_amount',
    operation='mean',
    window=30
))
ecommerce_pipeline.add_transformer(GroupbyShift(
    group_by='customer_id',
    column='purchase_amount',
    periods=[1, 2, 3]
))
ecommerce_pipeline.add_transformer(OneHotEncode(columns=['category']))
```

### 🏥 Healthcare Data
```python
# Healthcare preprocessing
healthcare_pipeline = XPipeline()
healthcare_pipeline.add_transformer(FillMissing(strategy='mode'))
healthcare_pipeline.add_transformer(DetectCategories(threshold=5))
healthcare_pipeline.add_transformer(LabelEncode(columns=['diagnosis']))
healthcare_pipeline.add_transformer(MinMaxScale(columns=['age', 'weight', 'height']))
```

## Troubleshooting

### Common Issues

<details>
<summary><strong>Pipeline fails during transform</strong></summary>

**Possible causes:**
- New categories in test data not seen during training
- Missing columns in new data
- Data type mismatches

**Solutions:**
- Use `handle_unknown='ignore'` in encoders
- Ensure consistent column names
- Check data types before transformation
</details>

<details>
<summary><strong>Memory issues with large datasets</strong></summary>

**Solutions:**
- Enable `memory_efficient=True` in pipeline
- Process data in chunks
- Use sparse matrices for one-hot encoding
- Remove unnecessary columns early
</details>

<details>
<summary><strong>Inverse transformation not working</strong></summary>

**Note:**
- Not all transformations are reversible
- Some transformations lose information (e.g., clipping)
- Check transformer documentation for inverse support
</details>

## Next Steps

:::note Ready to Build Models?
Now that you understand preprocessing, explore:
- [Binary Classification](./classification-binary.md) with preprocessed data
- [Regression](./regression.md) for continuous targets
- [Advanced Topics](../advanced-topics/) for optimization techniques
- [Cloud Integration](../getting-started/cloud-integration.md) for pipeline deployment
:::

## Example: Complete Preprocessing Workflow

```python
import xplainable as xp
import pandas as pd

# Load data
data = pd.read_csv('customer_data.csv')

# Create comprehensive preprocessing pipeline
pipeline = xp.XPipeline()

# Data cleaning
pipeline.add_transformer(xp.DropCols(columns=['id', 'timestamp']))
pipeline.add_transformer(xp.DropNaNs(threshold=0.5))

# Missing value handling
pipeline.add_transformer(xp.FillMissing(
    strategy='mean',
    columns=['age', 'income']
))
pipeline.add_transformer(xp.FillMissing(
    strategy='mode',
    columns=['category', 'region']
))

# Feature engineering
pipeline.add_transformer(xp.Operation(
    operations={
        'income_per_age': 'income / age',
        'is_high_income': 'income > 50000'
    }
))

# Categorical encoding
pipeline.add_transformer(xp.DetectCategories())
pipeline.add_transformer(xp.OneHotEncode(drop_first=True))

# Numeric transformations
pipeline.add_transformer(xp.LogTransform(columns=['income']))
pipeline.add_transformer(xp.Clip(lower_percentile=1, upper_percentile=99))

# Final scaling
pipeline.add_transformer(xp.MinMaxScale())

# Fit and transform
X, y = data.drop('target', axis=1), data['target']
pipeline.fit(X)
X_processed = pipeline.transform(X)

# Train model
model = xp.XClassifier()
model.fit(X_processed, y)

print(f"Pipeline successfully created with {len(pipeline.steps)} steps")
print(f"Final feature count: {X_processed.shape[1]}")
```