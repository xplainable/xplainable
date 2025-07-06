---
sidebar_position: 2
---

# Installation

## Quick Start

The fastest way to get started with xplainable is through PyPI:

```bash
pip install xplainable
```

:::tip Installation Complete!
That's it! You now have the core xplainable package installed and ready to use for transparent machine learning.
:::

## Installation Options

### Core Package

The core package includes all essential features for transparent machine learning:

```bash
pip install xplainable
```

**Includes:**
- ✅ XClassifier and XRegressor models
- ✅ Preprocessing pipeline and transformers
- ✅ Hyperparameter optimization
- ✅ Model explainability and visualization
- ✅ Partitioned models and surrogate models

### GUI Features

For interactive Jupyter notebook GUIs, install with the GUI extras:

```bash
pip install xplainable[gui]
```

**Additional features:**
- 🎯 Interactive model training interfaces
- 📊 Visual preprocessing tools
- 🔧 GUI-based hyperparameter tuning
- 📈 Interactive explanations and plots

### Advanced Plotting

For enhanced visualization capabilities:

```bash
pip install xplainable[plotting]
```

**Additional features:**
- 📊 Advanced Altair-based visualizations
- 🎨 Custom plot themes and styling
- 📈 Interactive explanation plots
- 🔍 Enhanced model inspection tools

### Cloud Integration

For cloud deployment and collaboration features:

```bash
pip install xplainable-client
```

:::info Cloud Package
The cloud client is a **separate package** that provides integration with Xplainable Cloud for model deployment, collaboration, and production management.
:::

**Cloud features:**
- ☁️ Model deployment and management
- 👥 Team collaboration
- 🔄 Model versioning
- 📊 Production monitoring
- 🔐 Secure API deployments

## Complete Installation

For all features, install both packages:

```bash
pip install xplainable[gui,plotting]
pip install xplainable-client
```

## Environment Setup

### Recommended Environment

<div className="row">
  <div className="col col--6">
    <div className="card">
      <div className="card__header">
        <h3>🐍 Python Version</h3>
      </div>
      <div className="card__body">
        <p><strong>Python 3.8 - 3.11</strong></p>
        <p>Python 3.8 recommended for GUI features due to ipywidgets compatibility.</p>
      </div>
    </div>
  </div>
  <div className="col col--6">
    <div className="card">
      <div className="card__header">
        <h3>💻 Environment</h3>
      </div>
      <div className="card__body">
        <p><strong>Virtual Environment</strong></p>
        <p>Always use virtual environments to avoid package conflicts.</p>
      </div>
    </div>
  </div>
</div>

### Setting Up Virtual Environment

#### Using venv (Recommended)

```bash
# Create virtual environment
python -m venv xplainable-env

# Activate environment
# On Windows:
xplainable-env\Scripts\activate
# On macOS/Linux:
source xplainable-env/bin/activate

# Install xplainable
pip install xplainable[gui,plotting]
pip install xplainable-client
```

#### Using conda

```bash
# Create conda environment
conda create -n xplainable-env python=3.8

# Activate environment
conda activate xplainable-env

# Install xplainable
pip install xplainable[gui,plotting]
pip install xplainable-client
```

## Jupyter Notebook Setup

### Installation

If you don't have Jupyter installed:

```bash
pip install jupyter
```

### Widget Extensions

For GUI features to work properly in Jupyter:

```bash
# Install and enable widget extensions
jupyter nbextension enable --py widgetsnbextension
```

### JupyterLab Setup

For JupyterLab users:

```bash
pip install jupyterlab
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

## Known Issues & Solutions

### Widget Rendering Issues

If widgets don't render properly:

```bash
# Reinstall ipywidgets
pip uninstall ipywidgets
pip install ipywidgets==7.6.5

# Clear notebook cache
jupyter notebook --clear-cache
```

### Import Errors

If you encounter import errors:

```bash
# Upgrade pip and reinstall
pip install --upgrade pip
pip install --force-reinstall xplainable
```

## Verification

### Test Core Installation

```python
import xplainable as xp
print(f"Xplainable version: {xp.__version__}")

# Test basic functionality
from xplainable.core.models import XClassifier
model = XClassifier()
print("✅ Core installation successful!")
```

### Test GUI Installation

```python
import xplainable as xp

# This should work without errors if GUI is installed
try:
    # Test GUI components
    from xplainable.gui import classifier
    print("✅ GUI installation successful!")
except ImportError as e:
    print(f"❌ GUI installation failed: {e}")
    print("Install with: pip install xplainable[gui]")
```

### Test Cloud Client

```python
try:
    from xplainable_client import Client
    print("✅ Cloud client installation successful!")
except ImportError as e:
    print(f"❌ Cloud client not installed: {e}")
    print("Install with: pip install xplainable-client")
```

## Docker Setup

For containerized environments:

```dockerfile
FROM python:3.8-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install xplainable
RUN pip install xplainable[gui,plotting] xplainable-client

# Set working directory
WORKDIR /app

# Copy your code
COPY . .

# Expose Jupyter port
EXPOSE 8888

# Start Jupyter
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
```

## Troubleshooting

### Common Issues

<details>
<summary><strong>ModuleNotFoundError: No module named 'xplainable'</strong></summary>

**Solution:**
- Check that you're in the correct virtual environment
- Reinstall: `pip install xplainable`
- Verify installation: `pip list | grep xplainable`
</details>

<details>
<summary><strong>Widgets not displaying in Jupyter</strong></summary>

**Solution:**
- Ensure you have the GUI extras: `pip install xplainable[gui]`
- Install widget extensions: `jupyter nbextension enable --py widgetsnbextension`
- Restart Jupyter kernel
- Use Python 3.8 for best compatibility
</details>

<details>
<summary><strong>Cloud client import errors</strong></summary>

**Solution:**
- Install cloud client: `pip install xplainable-client`
- Check that both packages are in the same environment
- Verify installation: `pip list | grep xplainable`
</details>

## Next Steps

:::note Ready to Build?
Now that you have xplainable installed, check out our [Python API documentation](../python-api/classification-binary.md) or jump straight into our [tutorials](../tutorials/) for hands-on examples.
:::

### Quick Start Example

```python
import xplainable as xp
from xplainable.core.models import XClassifier

# Load sample data
data = xp.load_dataset('titanic')
X, y = data.drop('Survived', axis=1), data['Survived']

# Train a transparent model
model = XClassifier()
model.fit(X, y)

# Get explanations
model.explain()
```

### Cloud Integration Example

```python
from xplainable_client import Client
import os

# Initialize cloud client
client = Client(api_key=os.environ['XP_API_KEY'])

# Deploy your model
model_id, version_id = client.create_model(
    model=model,
    model_name="My First Model",
    model_description="Transparent Titanic survival model",
    x=X,
    y=y
)
```

## Support

Need help with installation?

- 📚 **Documentation**: Check our comprehensive guides
- 💬 **Community**: Join our user community
- 🐛 **Issues**: Report bugs on GitHub
- 📧 **Enterprise**: Contact us for enterprise support
