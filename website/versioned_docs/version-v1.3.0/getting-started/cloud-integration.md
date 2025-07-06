---
sidebar_position: 3
---

# Cloud Integration

:::info Xplainable Cloud
**Xplainable Cloud** provides enterprise-grade model deployment, collaboration, and production management capabilities through the separate `xplainable-client` package.
:::

## Overview

The xplainable ecosystem includes two packages:

1. **`xplainable`** - Core transparent ML package (open source)
2. **`xplainable-client`** - Cloud integration package (separate install)

This separation allows you to use the core xplainable features without cloud dependencies, while providing full cloud capabilities when needed.

## Installation

```bash
pip install xplainable-client
```

:::tip Cloud Package
The cloud client is completely separate from the core xplainable package. Install both for full functionality.
:::

## Quick Start

### Basic Setup

```python
import os
from xplainable_client import Client

# Initialize the client
client = Client(api_key=os.environ['XP_API_KEY'])
```

### With Custom Configuration

```python
client = Client(
    api_key=os.environ['XP_API_KEY'],
    hostname='https://api.xplainable.io',  # Default
    org_id=None,  # Optional organization ID
    team_id=None  # Optional team ID
)
```

## Core Features

### 🗂️ Dataset Management

<div className="row">
  <div className="col col--12">
    <div className="card">
      <div className="card__header">
        <h3>Public Datasets</h3>
      </div>
      <div className="card__body">
        <p>Access curated datasets for learning and experimentation.</p>
        
```python
# List available public datasets
datasets = client.list_datasets()
print(f"Available datasets: {len(datasets)}")

# Load a specific dataset
data = client.load_dataset('titanic')
print(f"Dataset shape: {data.shape}")

# Load with custom parameters
data = client.load_dataset(
    'titanic',
    include_target=True,
    preprocessing=True
)
```
      </div>
    </div>
  </div>
</div>

### 🤖 Model Management

<div className="row">
  <div className="col col--6">
    <div className="card">
      <div className="card__header">
        <h3>Model Creation</h3>
      </div>
      <div className="card__body">
        
```python
# Train a local model
import xplainable as xp
model = xp.XClassifier()
model.fit(X_train, y_train)

# Deploy to cloud
model_id, version_id = client.create_model(
    model=model,
    model_name="Customer Churn Model",
    model_description="Predicts customer churn",
    x=X_train,
    y=y_train
)
```
      </div>
    </div>
  </div>
  <div className="col col--6">
    <div className="card">
      <div className="card__header">
        <h3>Model Loading</h3>
      </div>
      <div className="card__body">
        
```python
# Load classifier from cloud
model = client.load_classifier(
    model_id="your-model-id",
    version_id="latest"
)

# Load regressor from cloud
model = client.load_regressor(
    model_id="your-model-id",
    version_id="v1.0"
)
```
      </div>
    </div>
  </div>
</div>

### 📊 Preprocessing Management

```python
# Create preprocessing pipeline
preprocessor = xp.XPipeline()
preprocessor.add_transformer(xp.FillMissing())
preprocessor.add_transformer(xp.OneHotEncode())

# Save to cloud
preprocessor_id = client.create_preprocessor(
    preprocessor=preprocessor,
    preprocessor_name="Standard Pipeline",
    preprocessor_description="Fill missing + one-hot encoding"
)

# Load from cloud
preprocessor = client.load_preprocessor(preprocessor_id)
```

### 🚀 Model Deployment

<div className="row">
  <div className="col col--12">
    <div className="card">
      <div className="card__header">
        <h3>Production Deployment</h3>
      </div>
      <div className="card__body">
        <p>Deploy models as REST APIs with one command.</p>
        
```python
# Deploy model to production
deployment = client.deploy(
    model_id="your-model-id",
    version_id="latest",
    deployment_name="churn-prediction-api",
    description="Customer churn prediction endpoint"
)

print(f"API URL: {deployment['api_url']}")
print(f"Status: {deployment['status']}")
```
      </div>
    </div>
  </div>
</div>

### 🔍 Model Inference

```python
# Make predictions via API
predictions = client.predict(
    model_id="your-model-id",
    data=X_test,
    return_explanations=True
)

# Get explanations
explanations = client.explain(
    model_id="your-model-id",
    data=X_test,
    explanation_type="global"
)
```

## Advanced Features

### 📈 Model Monitoring

```python
# Get model performance metrics
metrics = client.get_model_metrics(
    model_id="your-model-id",
    version_id="latest",
    time_range="7d"
)

# Set up alerts
client.create_alert(
    model_id="your-model-id",
    metric="accuracy",
    threshold=0.85,
    condition="below"
)
```

### 🔄 Model Versioning

```python
# Add new version to existing model
version_id = client.add_version(
    model_id="existing-model-id",
    model=updated_model,
    version_name="v2.0",
    description="Improved accuracy with new features"
)

# Compare versions
comparison = client.compare_versions(
    model_id="your-model-id",
    version_a="v1.0",
    version_b="v2.0"
)
```

### 👥 Team Collaboration

```python
# Share model with team
client.share_model(
    model_id="your-model-id",
    team_id="your-team-id",
    permissions=["read", "predict"]
)

# List team models
team_models = client.list_team_models(team_id="your-team-id")
```

## AI Assistant Integration

:::tip AI-Powered Insights
The cloud client includes AI assistant capabilities for automated insights and explanations.
:::

```python
# Get AI insights about your model
insights = client.get_ai_insights(
    model_id="your-model-id",
    data=X_test,
    question="What are the key drivers of churn?"
)

# Generate automated report
report = client.generate_report(
    model_id="your-model-id",
    report_type="performance",
    include_explanations=True
)
```

## Complete Workflow Example

Here's a complete example showing the full workflow from training to deployment:

```python
import xplainable as xp
from xplainable_client import Client
import pandas as pd
import os

# Initialize client
client = Client(api_key=os.environ['XP_API_KEY'])

# Load data from cloud
data = client.load_dataset('customer_churn')
X, y = data.drop('churn', axis=1), data['churn']

# Create and train model locally
model = xp.XClassifier(
    max_depth=5,
    min_info_gain=0.01,
    weight=0.5
)
model.fit(X, y)

# Create preprocessing pipeline
preprocessor = xp.XPipeline()
preprocessor.add_transformer(xp.FillMissing())
preprocessor.add_transformer(xp.OneHotEncode())
preprocessor.fit(X)

# Deploy preprocessing to cloud
preprocessor_id = client.create_preprocessor(
    preprocessor=preprocessor,
    preprocessor_name="Churn Preprocessing",
    preprocessor_description="Standard preprocessing for churn model"
)

# Deploy model to cloud
model_id, version_id = client.create_model(
    model=model,
    model_name="Customer Churn Predictor",
    model_description="Transparent model for predicting customer churn",
    x=X,
    y=y,
    preprocessor_id=preprocessor_id
)

# Deploy to production
deployment = client.deploy(
    model_id=model_id,
    version_id=version_id,
    deployment_name="churn-api",
    description="Production churn prediction API"
)

print(f"🚀 Model deployed successfully!")
print(f"📊 Model ID: {model_id}")
print(f"🔗 API URL: {deployment['api_url']}")
```

## Security & Authentication

### API Key Management

```python
# Set API key via environment variable (recommended)
export XP_API_KEY="your-api-key-here"

# Or set programmatically (not recommended for production)
client = Client(api_key="your-api-key-here")
```

### Organization & Team Management

```python
# Initialize with organization context
client = Client(
    api_key=os.environ['XP_API_KEY'],
    org_id="your-org-id",
    team_id="your-team-id"
)

# Switch context
client.set_organization("different-org-id")
client.set_team("different-team-id")
```

## Error Handling

```python
from xplainable_client import Client, XplainableClientError

try:
    client = Client(api_key="invalid-key")
    model = client.load_classifier("non-existent-model")
except XplainableClientError as e:
    print(f"Client error: {e}")
    # Handle specific error cases
    if "authentication" in str(e).lower():
        print("Check your API key")
    elif "not found" in str(e).lower():
        print("Model doesn't exist")
```

## Best Practices

### 🔐 Security

- **Never hardcode API keys** - Use environment variables
- **Use organization/team contexts** for proper access control
- **Regularly rotate API keys** for security
- **Monitor API usage** through the dashboard

### 📊 Performance

- **Cache model objects** to avoid repeated downloads
- **Use batch predictions** for multiple samples
- **Monitor deployment metrics** for performance insights
- **Version models systematically** for reproducibility

### 🤝 Collaboration

- **Use descriptive model names** and descriptions
- **Tag models** with relevant metadata
- **Share models appropriately** with team permissions
- **Document model assumptions** and limitations

## Migration Guide

### From Internal Client (Pre-v1.2.9)

If you were using the internal client, here's how to migrate:

```python
# OLD (Internal client - deprecated)
import xplainable as xp
xp.initialise(api_key="your-key")
model = xp.load_model("model-id")

# NEW (External client)
from xplainable_client import Client
client = Client(api_key="your-key")
model = client.load_classifier("model-id")
```

### Key Changes

| Feature | Old (Internal) | New (External) |
|---------|---------------|----------------|
| **Import** | `import xplainable as xp` | `from xplainable_client import Client` |
| **Initialize** | `xp.initialise()` | `Client(api_key=...)` |
| **Load Model** | `xp.load_model()` | `client.load_classifier()` |
| **Deploy** | `xp.deploy()` | `client.deploy()` |

## Support

Need help with cloud integration?

- 📚 **Documentation**: Comprehensive API reference
- 💬 **Community**: Join our user community
- 🔧 **Support**: Enterprise support available
- 🐛 **Issues**: Report bugs and feature requests

:::note Next Steps
Ready to deploy your first model? Check out our [tutorials](../tutorials/) for complete examples, or explore the [Python API documentation](../python-api/) for detailed technical information.
::: 