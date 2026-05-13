
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![Contributors](https://img.shields.io/badge/Contributors-3-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

<div align="center">
<img src="https://raw.githubusercontent.com/xplainable/xplainable/main/docs/assets/logo/xplainable-logo.png">
<h1 align="center">xplainable</h1>
<h3 align="center">Real-time explainable machine learning for business optimisation</h3>

[![Python](https://img.shields.io/pypi/pyversions/xplainable)](https://pypi.org/project/xplainable/)
[![PyPi](https://img.shields.io/pypi/v/xplainable?color=blue)](https://pypi.org/project/xplainable/)
[![Downloads](https://static.pepy.tech/badge/xplainable)](https://pepy.tech/project/xplainable)

**Xplainable** makes tabular machine learning transparent, fair, and actionable.
</div>

## Why Xplainable?

In machine learning, there has long been a trade-off between accuracy and
explainability. Libraries like [Shap](https://github.com/slundberg/shap) and [Lime](https://github.com/marcotcr/lime) estimate model decisions after the fact, but they're slow and add complexity.

**xplainable** takes a different approach: models that are explainable *by design*. Our algorithms match the performance of black-box models like [XGBoost](https://github.com/dmlc/xgboost) and [LightGBM](https://github.com/microsoft/LightGBM) while providing complete transparency in real-time — no surrogate models, no approximations.

Every prediction comes with per-feature contribution scores that explain *why* the model made that decision. These contributions are exact (not estimates) and can be used to drive business actions like retention campaigns, risk routing, and cost optimisation.

## Installation

```bash
pip install xplainable
```

For preprocessing pipelines (spec-driven, JSON-serializable):
```bash
pip install xplainable-preprocessing
```

For cloud model management, deployment, and collaboration:
```bash
pip install xplainable-client
```

## Quick Start

```python
import xplainable as xp
from xplainable.core.models import XClassifier
from xplainable.core.optimisation.bayesian import XParamOptimiser
from sklearn.model_selection import train_test_split

# Load and split data
data = xp.load_dataset('titanic')
X, y = data.drop(columns=['Survived']), data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Optimise hyperparameters
opt = XParamOptimiser()
params = opt.optimise(X_train, y_train)

# Train
model = XClassifier(**params)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Explain — interactive feature importances and contribution plots
model.explain()
```

## Models

| Model | Class |
|:------|:------|
| Binary Classification | `XClassifier` |
| Regression | `XRegressor` |
| Partitioned Classification | `PartitionedClassifier` |
| Partitioned Regression | `PartitionedRegressor` |

## Key Features

### Explainability — Built In, Not Bolted On

Every xplainable model provides:

- **Feature importances** — which features matter most
- **Partition contributions** — how each feature value range shifts the prediction
- **Per-instance explanations** — why this specific prediction was made

```python
# Global explanation
model.explain()

# Per-instance contributions
contributions = model._transform(X_test)

# Model profile (all partition details)
profile = model.profile
```

### Preprocessing with `xplainable-preprocessing`

Spec-driven, JSON-serializable pipelines that can be versioned, previewed, and persisted to Xplainable Cloud.

```python
from xplainable_preprocessing import PipelineSpec, StepSpec, compile_spec

spec = PipelineSpec(steps=[
    StepSpec(
        id="lowercase",
        type="TextCleanTransformer",
        columns=["country", "category"],
        params={"operations": ["lowercase"]},
    ),
    StepSpec(
        id="fill_missing",
        type="FillMissingTransformer",
        params={"strategy": "median"},
    ),
    StepSpec(
        id="drop_ids",
        type="DropColumnsTransformer",
        params={"columns": ["customer_id", "order_id"]},
    ),
])

pipeline = compile_spec(spec)
df_transformed = pipeline.fit_transform(df)
```

Available transformers: `TextCleanTransformer`, `DropColumnsTransformer`, `FillMissingTransformer`, `TypeCastTransformer`, `CategoryCondenseTransformer`, `ExpressionTransformer`, `DateTimeExtractTransformer`, `RenameColumnsTransformer`, `GroupByAggTransformer`, `GroupedLagTransformer`, `RollingAggTransformer`, plus all standard sklearn transformers (StandardScaler, OneHotEncoder, etc.)

### Hyperparameter Optimisation

Bayesian optimisation finds the best parameters automatically.

```python
from xplainable.core.optimisation.bayesian import XParamOptimiser

opt = XParamOptimiser(metric='roc-auc')
params = opt.optimise(X_train, y_train)

model = XClassifier(**params)
model.fit(X_train, y_train)
```

### Rapid Refitting

Fine-tune model parameters on individual features without retraining from scratch.

```python
model.update_feature_params(
    features=['Age'],
    max_depth=6,
    min_info_gain=0.01,
    min_leaf_size=0.03,
    weight=0.05,
    power_degree=1,
    sigmoid_exponent=1,
    x=X_train,
    y=y_train
)
```

### Contribution-Driven Optimisation

Use the model's per-feature contributions to identify controllable business levers and calculate the expected value of interventions — derived from the data, not assumed.

```python
# Get per-feature contributions
contributions = model._transform(X_test)

# Model profile shows partition boundaries and scores
profile = model.profile

# For controllable features, compute counterfactual lever effects:
# "How much would churn drop if we moved this customer to the best partition?"
best_score = min(p['score'] for p in profile['numeric']['orders_count'])
lever_effect = current_contribution - best_score
```

See the [Shopify Customer Churn](examples/Shopify_Customer_Churn.ipynb) notebook for a complete example.

## Xplainable Cloud

Deploy models, persist preprocessing pipelines, and collaborate with your team through the Xplainable Cloud platform.

```python
from xplainable_client.client.client import XplainableClient

client = XplainableClient(
    api_key="your-api-key",
    hostname="https://platform.xplainable.io"
)

# Persist preprocessing
client.preprocessing.create_preprocessor(
    name="My Preprocessor",
    description="Feature transforms for churn model",
    spec=preprocessing_spec.model_dump(),
    sample_df=df,
)

# Persist model
client.models.create_model(
    model=model,
    model_name="Churn Prediction",
    model_description="Customer churn classifier",
    x=X_train, y=y_train
)

# Deploy
deployment = client.deployments.deploy(model_version_id=version_id)
```

## Examples

| Notebook | Type | Description |
|:---------|:-----|:------------|
| [Shopify Customer Churn](examples/Shopify_Customer_Churn.ipynb) | Classification | Churn prediction with contribution-driven retention optimisation |
| [Shopify Order Returns](examples/Shopify_Order_Returns.ipynb) | Classification | Return prediction with intervention routing |
| [Telco Churn](examples/Telco_Churn.ipynb) | Classification | IBM Telco customer churn |
| [HELOC Credit Risk](examples/HELOC_Credit_Risk.ipynb) | Classification | Credit risk assessment |
| [Lead Scoring](examples/Lead_Scoring_Prediction.ipynb) | Classification | Lead conversion prediction |
| [House Prices](examples/House_Prices_Regression.ipynb) | Regression | Property price prediction |
| [Concrete Strength](examples/Concrete_Compressive_Strength.ipynb) | Regression | Material strength prediction |
| [Power Plant Output](examples/Power_Plant_Energy_Output.ipynb) | Regression | Energy output prediction |

## Documentation

- [General Documentation](https://docs.xplainable.io)
- [API Documentation](https://xplainable.readthedocs.io)

## Contributing

We welcome contributions. If you're interested, reach out at contact@xplainable.io.

<div align="center">
<br>
<strong>Made with care in Australia</strong>
<br>
<hr>
&copy; xplainable pty ltd
</div>
