# Migration Guide: Internal Client to External Client

This guide helps you migrate from the internal client (deprecated in v1.2.9) to the external `xplainable-client` package.

## Overview

Starting with xplainable v1.2.9, cloud functionality has been moved to a separate `xplainable-client` package. This change provides:

- Better separation of concerns
- Independent versioning for cloud features
- Reduced dependencies for users who don't need cloud functionality
- More flexible deployment options

## Installation

### Before (Internal Client)
```bash
pip install xplainable  # Cloud functionality included
```

### After (External Client)
```bash
pip install xplainable  # Core functionality only
pip install xplainable-client  # Cloud functionality
```

## Import Changes

### Before (Internal Client)
```python
import xplainable as xp

# Initialize cloud functionality
xp.initialise(api_key=os.environ['XP_API_KEY'])

# Access client functions directly
models = xp.list_models()
datasets = xp.list_datasets()
```

### After (External Client)
```python
import xplainable as xp
from xplainable_client import Client

# Initialize cloud client
client = Client(api_key=os.environ['XP_API_KEY'])

# Access client functions through client instance
models = client.list_models()
datasets = client.list_datasets()
```

## Function Mapping

| Old Function | New Function |
|:-------------|:-------------|
| `xp.initialise()` | `Client(api_key=...)` |
| `xp.list_datasets()` | `client.list_datasets()` |
| `xp.load_dataset()` | `client.load_dataset()` |
| `xp.list_models()` | `client.list_models()` |
| `xp.load_classifier()` | `client.load_classifier()` |
| `xp.load_regressor()` | `client.load_regressor()` |
| `xp.create_model()` | `client.create_model()` |
| `xp.deploy()` | `client.deploy()` |

## Common Migration Patterns

### Model Creation and Deployment

#### Before
```python
import xplainable as xp

xp.initialise(api_key=os.environ['XP_API_KEY'])

# Create model
model_id, version_id = xp.create_model(
    model=trained_model,
    model_name="My Model",
    model_description="Description",
    x=X_train,
    y=y_train
)

# Deploy
deployment = xp.deploy(model_version_id=version_id)
```

#### After
```python
import xplainable as xp
from xplainable_client import Client

client = Client(api_key=os.environ['XP_API_KEY'])

# Create model
model_id, version_id = client.create_model(
    model=trained_model,
    model_name="My Model",
    model_description="Description",
    x=X_train,
    y=y_train
)

# Deploy
deployment = client.deploy(model_version_id=version_id)
```

### Dataset Operations

#### Before
```python
import xplainable as xp

xp.initialise(api_key=os.environ['XP_API_KEY'])

# List and load datasets
datasets = xp.list_datasets()
df = xp.load_dataset('iris')
```

#### After
```python
from xplainable_client import Client

client = Client(api_key=os.environ['XP_API_KEY'])

# List and load datasets
datasets = client.list_datasets()
df = client.load_dataset('iris')
```

### Preprocessing

#### Before
```python
import xplainable as xp

xp.initialise(api_key=os.environ['XP_API_KEY'])

# Create preprocessor
preprocessor_id, version_id = xp.create_preprocessor(
    preprocessor_name="My Preprocessor",
    preprocessor_description="Description",
    pipeline=my_pipeline,
    df=data
)
```

#### After
```python
from xplainable_client import Client

client = Client(api_key=os.environ['XP_API_KEY'])

# Create preprocessor
preprocessor_id, version_id = client.create_preprocessor(
    preprocessor_name="My Preprocessor",
    preprocessor_description="Description",
    pipeline=my_pipeline,
    df=data
)
```

## Breaking Changes

### 1. Initialization
- `xp.initialise()` no longer exists
- Use `Client(api_key=...)` instead

### 2. Function Access
- Cloud functions are no longer available directly on the `xp` module
- All cloud functions must be accessed through a `Client` instance

### 3. Import Structure
- `from xplainable import Client` no longer works
- Use `from xplainable_client import Client`

### 4. Dependencies
- Cloud functionality requires explicit installation of `xplainable-client`
- The main `xplainable` package no longer includes cloud dependencies

## Error Handling

### Old Import Errors
If you see errors like:
```python
AttributeError: module 'xplainable' has no attribute 'initialise'
AttributeError: module 'xplainable' has no attribute 'list_datasets'
```

This indicates you're using the old internal client API. Follow the migration steps above.

### Missing Client Package
If you see:
```python
ModuleNotFoundError: No module named 'xplainable_client'
```

Install the external client:
```bash
pip install xplainable-client
```

## Gradual Migration Strategy

For large codebases, consider this gradual migration approach:

### Step 1: Install External Client
```bash
pip install xplainable-client
```

### Step 2: Create Wrapper Functions
```python
# migration_helpers.py
from xplainable_client import Client
import os

# Initialize client once
_client = Client(api_key=os.environ['XP_API_KEY'])

# Create wrapper functions for backward compatibility
def list_datasets():
    return _client.list_datasets()

def load_dataset(name):
    return _client.load_dataset(name)

def create_model(*args, **kwargs):
    return _client.create_model(*args, **kwargs)

# Add more wrappers as needed
```

### Step 3: Update Imports Gradually
```python
# Instead of: import xplainable as xp
from migration_helpers import list_datasets, load_dataset, create_model

# Use functions as before
datasets = list_datasets()
df = load_dataset('iris')
```

### Step 4: Complete Migration
Once all code is updated, remove wrapper functions and use the client directly.

## Testing Your Migration

Create a simple test script to verify your migration:

```python
from xplainable_client import Client
import os

# Test client initialization
try:
    client = Client(api_key=os.environ['XP_API_KEY'])
    print("✅ Client initialization successful")
except Exception as e:
    print(f"❌ Client initialization failed: {e}")

# Test basic functionality
try:
    datasets = client.list_datasets()
    print(f"✅ Listed {len(datasets)} datasets")
except Exception as e:
    print(f"❌ Dataset listing failed: {e}")

# Test dataset loading
try:
    df = client.load_dataset('iris')
    print(f"✅ Loaded dataset with {len(df)} rows")
except Exception as e:
    print(f"❌ Dataset loading failed: {e}")

print("Migration test complete!")
```

## Support

If you encounter issues during migration:

1. Check that you have the latest versions:
   ```bash
   pip install --upgrade xplainable xplainable-client
   ```

2. Verify your API key is set correctly:
   ```python
   import os
   print(f"API Key set: {'XP_API_KEY' in os.environ}")
   ```

3. Review the [Cloud Integration Guide](cloud-integration.md) for detailed API documentation

4. Contact support if you need assistance with complex migration scenarios

## Frequently Asked Questions

### Q: Why was the client moved to a separate package?
A: This separation provides better modularity, reduces dependencies for users who don't need cloud functionality, and allows independent versioning of cloud features.

### Q: Do I need to change my API key?
A: No, your existing API key will continue to work with the external client.

### Q: Will the old internal client continue to work?
A: The internal client was removed in v1.2.9. You must migrate to the external client to use cloud functionality.

### Q: Can I use both packages together?
A: Yes, the external client is designed to work seamlessly with the main xplainable package.

### Q: Are there any feature differences?
A: The external client provides the same functionality as the internal client, with improved error handling and additional features. 