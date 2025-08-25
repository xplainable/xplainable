# Cloud Integration

The xplainable package integrates with **Xplainable Cloud** through the separate `xplainable-client` package, which provides cloud-based model management, deployment, and collaboration features.

## Installation

The cloud client is available as a separate package:

```bash
pip install xplainable-client
```

## Quick Start

```python
import os
from xplainable_client import Client

# Initialize the client
client = Client(api_key=os.environ['XP_API_KEY'])

# Optional parameters
client = Client(
    api_key=os.environ['XP_API_KEY'],
    hostname='https://platform.xplainable.io',  # Default
    org_id=None,  # Optional organization ID
    team_id=None  # Optional team ID
)
```

## Core Functionality

### Dataset Management

```python
# List available public datasets
datasets = client.list_datasets()

# Load a public dataset
df = client.load_dataset('iris')  # Replace with dataset name
```

### Model Management

```python
# Create a new model
model_id, version_id = client.create_model(
    model=trained_model,  # Your trained XClassifier/XRegressor
    model_name="My Model",
    model_description="Model description",
    x=X_train,  # Training features
    y=y_train   # Training targets
)

# Add a new version to existing model
version_id = client.add_version(
    model=updated_model,
    x=X_train,
    y=y_train
)

# List all models
models = client.list_models()

# Load a model from cloud
loaded_model = client.load_classifier(model_id, version_id)
# or
loaded_model = client.load_regressor(model_id, version_id)
```

### Preprocessing Management

```python
from xplainable.preprocessing import XPipeline

# Create a preprocessor
preprocessor_id, version_id = client.create_preprocessor(
    preprocessor_name="My Preprocessor",
    preprocessor_description="Preprocessing pipeline",
    pipeline=my_pipeline,  # Your XPipeline
    df=raw_data
)

# Load a preprocessor
loaded_pipeline = client.load_preprocessor(preprocessor_id, version_id)

# List preprocessors
preprocessors = client.list_preprocessors()
```

### Model Deployment

```python
# Deploy a model version
deployment = client.deploy(model_version_id=version_id)

# Activate deployment
client.activate_deployment(deployment['deployment_id'])

# Generate API key for deployment
deploy_key = client.generate_deploy_key(
    deployment_id=deployment['deployment_id'],
    description="Production API key",
    days_until_expiry=90
)

# List deployments
deployments = client.list_deployments()
```

### Collections and Scenarios

```python
# Create a collection for organizing scenarios
collection_id = client.create_collection(
    model_id=model_id,
    name="Test Scenarios",
    description="Collection of test scenarios"
)

# Get model collections
collections = client.get_model_collections(model_id)

# Get team collections
team_collections = client.get_team_collections()
```

### Inference

```python
# Make predictions on a file
predictions = client.predict(
    filename="data.csv",
    model_id=model_id,
    version_id=version_id,
    threshold=0.5,  # For classification
    delimiter=","
)
```

## Advanced Features

### Model Lifecycle Management

```python
# Set active version
client.set_active_version(model_id, version_id)

# Archive/restore models
client.archive_model(model_id)
client.restore_archived_model(model_id)

# Publish/unpublish versions
client.publish_model_version(model_id, version_id)
client.unpublish_model_version(model_id, version_id)

# Update model metadata
client.update_model_name(model_id, "New Model Name")
client.update_model_description(model_id, "Updated description")
```

### Deployment Security

```python
# Manage IP restrictions
client.add_allowed_ip_address(deployment_id)
client.list_allowed_ip_addresses(deployment_id)
client.delete_allowed_ip_address(deployment_id, ip_id)

# Activate/deactivate IP blocking
client.activate_deployment_ip_blocking(deployment_id)
client.deactivate_deployment_ip_blocking(deployment_id)

# Manage deploy keys
deploy_keys = client.list_deploy_keys(deployment_id)
client.revoke_deploy_key(deployment_id, deploy_key_id)
client.revoke_all_deploy_keys(deployment_id)
```

### Preprocessing Pipeline Management

```python
# Create preprocessor versions
version_id = client.create_preprocessor_version(
    preprocessor_id=preprocessor_id,
    pipeline=updated_pipeline,
    df=new_data
)

# Link preprocessor to model
client.link_model_preprocessor(
    model_id=model_id,
    version_id=model_version_id,
    preprocessor_id=preprocessor_id,
    preprocessor_version_id=preprocessor_version_id
)

# Check signature compatibility
compatibility = client.check_signature(
    preprocessor_id=preprocessor_id,
    preprocessor_version_id=preprocessor_version_id,
    model_id=model_id,
    model_version_id=model_version_id
)
```

## Error Handling

The client provides informative error messages for common issues:

```python
try:
    model_id, version_id = client.create_model(...)
except ValueError as e:
    print(f"Model creation failed: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Best Practices

1. **API Key Security**: Store API keys in environment variables
2. **Error Handling**: Always wrap client calls in try-except blocks
3. **Version Management**: Use meaningful descriptions for model versions
4. **Resource Cleanup**: Deactivate unused deployments to save resources
5. **Team Collaboration**: Use collections to organize scenarios and models

## Supported Model Types

The client supports all xplainable model types:
- `XClassifier` - Binary classification models
- `XRegressor` - Regression models  
- `PartitionedClassifier` - Multi-partition classification
- `PartitionedRegressor` - Multi-partition regression

## Integration with Main Package

The client seamlessly integrates with the main xplainable package:

```python
import xplainable as xp
from xplainable_client import Client

# Train locally
model = xp.XClassifier()
model.fit(X_train, y_train)

# Deploy to cloud
client = Client(api_key=os.environ['XP_API_KEY'])
model_id, version_id = client.create_model(
    model=model,
    model_name="Local Model",
    model_description="Trained locally, deployed to cloud",
    x=X_train,
    y=y_train
)
```

## Limitations

- The client requires an active internet connection
- Some features require specific subscription tiers
- Model size limits may apply based on your plan
- API rate limits may apply for high-volume usage

For detailed API reference and advanced usage, visit the [Xplainable Cloud documentation](https://docs.xplainable.io). 