
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![Contributors](https://img.shields.io/badge/Contributors-2-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

<div align="center">
<img src="https://raw.githubusercontent.com/xplainable/xplainable/main/docs/assets/logo/xplainable-logo.png">
<h1 align="center">xplainable</h1>
<h3 align="center">Real-time explainable machine learning for business optimisation</h3>
    
[![Python](https://img.shields.io/pypi/pyversions/xplainable)](https://pypi.org/project/xplainable/)
[![PyPi](https://img.shields.io/pypi/v/xplainable?color=blue)](https://pypi.org/project/xplainable/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://github.com/xplainable/xplainable/blob/dev/LICENSE)
[![Downloads](https://static.pepy.tech/badge/xplainable)](https://pepy.tech/project/xplainable)
    
**Xplainable** leverages explainable machine learning for fully transparent predictions and advanced data optimisation in production systems.
</div>


## Installation

You can install ``xplainable`` with:

```
pip install xplainable
```

Vist our [Documentation](https://xplainable.readthedocs.io) for extra support.

## Getting Started

**Basic Example**

```python
import xplainable as xp
import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv('data.csv')
train, test = train_test_split(data, test_size=0.2)

# Train a model
model = xp.classifier(train)
```

## Why Was Xplainable Created?
In machine learning, there has long been a trade-off between accuracy and 
explainability. This drawback has led to the creation of explainable ML
libraries such as ``Shap`` and ``LIME`` which make estimations of model decision
processes. These can be incredibly time-expensive and often present steep
learning curves making them challenging to implement effectively in production
environments.

To solve this problem, we created ``xplainable``. **xplainable** presents a
suite of novel machine learning algorithms specifically designed to match the
performance of popular black box models like ``XGBoost`` and ``LightGBM`` while
providing complete transparency, all in real-time.

## Simple Interface
You can interface with xplainable either through a typical Pythonic API, or
using a notebook-embedded GUI in your Jupyter Notebook.

## Models
Xplainable has each of the fundamental tabular models used in data science
teams. They are fast, accurate, and easy to use.

| Model | Python API| Jupyter GUI |
|:------|:------:|:------:|
| Regression | ✅ | ✅ |
| Binary Classification | ✅ | ✅ |
| Multi-Class Classification | ✅ | 🔜 |

## Features
Xplainable helps to streamline development processes by making model tuning
and deployment simpler than you can imagine.

### Preprocessing
We built a comprehensive suite of preprocessing transformers for rapid and
reproducible data preprocessing.

| Feature | Python API| Jupyter GUI |
|:------|:------:|:------:|
| Data Health Checks | ✅ | ✅ |
| Transformers Library | ✅ | ✅ |
| Preprocessing Pipelines | ✅ | ✅ |
| Pipeline Persistance | ✅ | ✅ |

```python
pp = xp.Preprocessor()

pp.preprocess(train)
```
<img src="https://raw.githubusercontent.com/xplainable/xplainable/main/docs/assets/gifs/preprocessing.gif">


### Modelling
Xplainable models can be developed, optimised, and re-optimised using Pythonic
APIs or the embedded GUI.

| Feature | Python API| Jupyter GUI |
|:------|:------:|:------:|
| Classic Vanilla Data Science APIs | ✅ | - |
| AutoML | ✅ | ✅ |
| Hyperparameter Optimisation | ✅ | ✅ |
| Partitioned Models | ✅ | ✅ |
| **Rapid Refitting** (novel to xplainable) | ✅ | ✅ |
| Model Persistance | ✅ | ✅ |

```python
model = xp.classifier(train)
```
<img src="https://raw.githubusercontent.com/xplainable/xplainable/main/docs/assets/gifs/gui_classifier.gif">

### Rapid Refitting
Fine tune your models by refitting model parameters on the fly, even on
individual features.

<img src="https://raw.githubusercontent.com/xplainable/xplainable/main/docs/assets/gifs/recalibrate.gif">

### Explainability
Models are explainable and real-time, right out of the box, without having to fit
surrogate models such as ``Shap`` or ``LIME``.

| Feature | Python API| Jupyter GUI |
|:------|:------:|:------:|
| Global Explainers | ✅ | ✅ |
| Regional Explainers | ✅ | ✅ |
| Local Explainers | ✅ | ✅ |
| Real-time Explainability | ✅ | ✅ |

```python
model.explain()
```

<img src="https://raw.githubusercontent.com/xplainable/xplainable/main/docs/assets/gifs/explain.gif">

### Action & Optimisation
We leverage the explainability of our models to provide real-time
recommendations on how to optimise predicted outcomes at a local and global
level.

| Feature |  |
|:------|:------:|
| Automated Local Prediction Optimisation | ✅ |
| Automated Global Decision Optimisation | 🔜 |

### Deployment
Xplainable brings transparency to API deployments, and it's easy. By the time
your finger leaves the mouse, your model is on a secure server and ready to go.

| Feature | Python API| Xplainable Cloud |
|:------|:------:|:------:|
| < 1 Second API Deployments | ✅ | ✅ |
| Explainability-Enabled API Deployments | ✅ | ✅ |
| A/B Testing | - | 🔜 |
| Champion Challenger Models (MAB) | - | 🔜 |

### #FairML
We promote fair and ethical use of technology for all machine learning tasks.
To help encourage this, we're working on additional bias detection and fairness
testing classes to ensure that everything you deploy is safe, fair, and
compliant.

| Feature | Python API| Xplainable Cloud |
|:------|:------:|:------:|
| Bias Identification | ✅ | ✅ |
| Automated Bias Detection | 🔜 | 🔜 |
| Fairness Testing | 🔜 | 🔜 |

## Xplainable Cloud
This Python package is free and open-source. To add more value to data teams
within organisations, we also created Xplainable Cloud that brings your models
to a collaborative environment.

```python
import xplainable as xp

xp.initialise()
```

<img src="https://raw.githubusercontent.com/xplainable/xplainable/main/docs/assets/gifs/initialise.gif">

## Contributors
We'd love to welcome contributors to xplainable to keep driving forward more
transparent and actionable machine learning. We're working on our contributor
docs at the moment, but if you're interested in contributing, please flick us a
message at contact@xplainable.io.




<div align="center">
<br></br>
<br></br>
Thanks for trying xplainable!
<br></br>
<strong>Made with ❤️ in Australia</strong>
<br></br>
<hr>
&copy; copyright xplainable pty ltd
</div>

