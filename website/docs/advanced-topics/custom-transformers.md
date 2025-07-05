---
sidebar_position: 3
---

# Custom Transformers

:::info Build Your Own Preprocessing Pipeline
**Custom transformers** allow you to create specialized preprocessing components that integrate seamlessly with xplainable models. Build domain-specific transformations while maintaining full transparency.
:::

## Overview

Custom transformers extend xplainable's preprocessing capabilities by allowing you to create specialized transformations tailored to your specific domain and data requirements. They maintain the same transparency and interpretability principles as core xplainable components.

### Key Benefits

<div className="row">
  <div className="col col--6">
    <div className="card">
      <div className="card__header">
        <h3>🔧 Domain-Specific</h3>
      </div>
      <div className="card__body">
        <p>Create transformations specific to your industry or use case.</p>
      </div>
    </div>
  </div>
  <div className="col col--6">
    <div className="card">
      <div className="card__header">
        <h3>🔗 Seamless Integration</h3>
      </div>
      <div className="card__body">
        <p>Integrate perfectly with xplainable models and pipelines.</p>
      </div>
    </div>
  </div>
</div>

## Creating Custom Transformers

### Basic Structure

```python
from xplainable.core.preprocessing import XBaseTransformer
import pandas as pd
import numpy as np

class CustomTransformer(XBaseTransformer):
    """Template for custom transformers."""
    
    def __init__(self, param1=None, param2=None):
        super().__init__()
        self.param1 = param1
        self.param2 = param2
        self.fitted_attributes = {}
    
    def fit(self, X, y=None):
        """Fit the transformer to the data."""
        # Your fitting logic here
        self.fitted_attributes['feature_names'] = X.columns.tolist()
        return self
    
    def transform(self, X):
        """Transform the data."""
        # Your transformation logic here
        X_transformed = X.copy()
        return X_transformed
    
    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
```

### Advanced Example: Financial Ratio Transformer

```python
class FinancialRatioTransformer(XBaseTransformer):
    """Create financial ratios from raw financial data."""
    
    def __init__(self, ratios_to_create=['liquidity', 'profitability', 'leverage']):
        super().__init__()
        self.ratios_to_create = ratios_to_create
        self.ratio_definitions = {}
    
    def fit(self, X, y=None):
        """Fit the transformer."""
        # Define ratio calculations
        self.ratio_definitions = {
            'liquidity': {
                'current_ratio': ('current_assets', 'current_liabilities'),
                'quick_ratio': ('quick_assets', 'current_liabilities'),
                'cash_ratio': ('cash', 'current_liabilities')
            },
            'profitability': {
                'gross_margin': ('gross_profit', 'revenue'),
                'operating_margin': ('operating_income', 'revenue'),
                'net_margin': ('net_income', 'revenue'),
                'roa': ('net_income', 'total_assets'),
                'roe': ('net_income', 'shareholders_equity')
            },
            'leverage': {
                'debt_to_equity': ('total_debt', 'shareholders_equity'),
                'debt_to_assets': ('total_debt', 'total_assets'),
                'interest_coverage': ('operating_income', 'interest_expense')
            }
        }
        
        # Validate required columns exist
        required_columns = set()
        for ratio_type in self.ratios_to_create:
            for ratio_name, (num, den) in self.ratio_definitions[ratio_type].items():
                required_columns.update([num, den])
        
        missing_columns = required_columns - set(X.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        return self
    
    def transform(self, X):
        """Transform data by adding financial ratios."""
        X_transformed = X.copy()
        
        for ratio_type in self.ratios_to_create:
            for ratio_name, (numerator, denominator) in self.ratio_definitions[ratio_type].items():
                # Calculate ratio with safe division
                ratio_values = np.where(
                    X[denominator] != 0,
                    X[numerator] / X[denominator],
                    np.nan
                )
                
                X_transformed[f'{ratio_name}'] = ratio_values
        
        return X_transformed
```

## Integration Examples

### Complete Pipeline with Custom Transformers

```python
from xplainable.core.models import XClassifier
from xplainable.core.preprocessing import XPreprocessor
from sklearn.model_selection import train_test_split

# Create custom transformer
financial_transformer = FinancialRatioTransformer(
    ratios_to_create=['liquidity', 'profitability', 'leverage']
)

# Create preprocessing pipeline
preprocessor = XPreprocessor([
    financial_transformer,
    ('scaler', 'standard'),
    ('selector', 'univariate')
])

# Load financial data
data = pd.read_csv('financial_data.csv')
X = data.drop('default', axis=1)
y = data['default']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit preprocessing pipeline
preprocessor.fit(X_train, y_train)

# Transform data
X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Train model
model = XClassifier(max_depth=6, min_info_gain=0.01)
model.fit(X_train_processed, y_train)

# Evaluate
accuracy = model.score(X_test_processed, y_test)
print(f"Model accuracy: {accuracy:.3f}")

# Analyze feature importance including custom ratios
feature_importance = model.feature_importance()
print("\nTop 10 most important features:")
print(feature_importance.head(10))
```

## Next Steps

:::note Ready for Advanced Features?
- Explore [rapid refitting](./rapid-refitting.md) for real-time model updates
- Learn about [partitioned models](./partitioned-models.md) for segment-specific modeling
- Check out [XEvolutionaryNetwork](./XEvolutionaryNetwork.md) for advanced optimization
:::

Custom transformers provide the flexibility to create domain-specific preprocessing while maintaining xplainable's core principles of transparency and interpretability.
