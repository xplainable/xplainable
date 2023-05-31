
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-2-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

<div align="center">
<img src="https://github.com/xplainable/xplainable/blob/dev/docs/assets/logo/xplainable-logo.png">
<h1 align="center">Xplainable</h1>
<h3 align="center">Explainable machine learning for business optimisation</h3>
    
[![Python](https://img.shields.io/pypi/pyversions/xplainable)](https://pypi.org/project/xplainable/)
[![PyPi](https://img.shields.io/pypi/v/xplainable?color=blue)](https://pypi.org/project/xplainable/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://github.com/xplainable/xplainable/blob/dev/LICENSE)
[![Downloads](https://static.pepy.tech/badge/xplainable)](https://pepy.tech/project/xplainable)
    
**Xplainable** leverages explainable machine learning for fully transparent predictions and advanced data optimisation in production systems.
</div>


## Installation

You can install ``xplainable`` with:

```python
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
<img src="https://github.com/xplainable/xplainable/blob/dev/docs/assets/gifs/gui_classifier.gif">

```python
model.explain()
```

## Overview
**xplainable** bridges the gap between data scientists, analysts, developers,
and business domain experts by providing a simple Python API and web interface 
to manage machine learning systems at both a technical and managerial level. It
achieves this by providing a set of tools that allow users to:

- Quickly preprocess data and generate features
- Train and evaluate machine learning models
- Visualise and interpret model performance
- Explain model predictions
- Deploy models to production as REST APIs in seconds
- Collaborate with other users on model development and evaluation
- Save and load preprocessing pipelines and models across your team or
organisation
- Share model profiles with other users via xplainable cloud

At the core of xplainable is a set of novel explainable machine learning
algorithms designed to provide similar performance to black box models while
maintaining complete transparency.

These docs contain details about how xplainable works and how to get the most
out of it.

## Who is Xplainable For?
xplainable is for anyone who wants to build machine learning models that can be
easily understood and explained. Experienced professionals, novices, students,
and hobbyists can all use the package given appropriate data. The only
requirement is a basic understanding of Python and machine learning at a
conceptual level.

The users who will get the most out of xplainable include:

- Data scientists
- Data analysts
- Data engineers
- Developers
- Business domain experts

The ``xplainable`` package, combined with the web application, is designed to be
used by individuals and teams within data-centric organisations.

## Skill Requirements
Anyone with a basic understanding of Python and machine learning at a conceptual
level can use xplainable. The package is intuitive and easy to use by design,
and the web application provides a simple interface for managing models and
deployments.

### Experienced Users

Experienced users can use xplainable like any other open-source machine learning
package. The package provides a simple API for training and evaluating models
with the added benefit of novel model-tuning methods and advanced explainability
tools.

These users can still benefit from the GUI tools of xplainable by streamlining
the process of training and evaluating models, but they can also go as low-level
into the code as they require for complete control over the model development
process.

### Novice Users

Novice users can use xplainable to learn about machine learning and experiment
with models and datasets using AutoML. The package provides a simple embedded
GUI for training and evaluating models without having to write much code or
understand the underlying algorithms.

xplainable also gives users with little to no experience with machine learning
the ability to deploy models to production as REST APIs in seconds, which
significantly reduces the barrier to entry of adding tangible value to
data-centric organisations.

As novice users become more experienced, they can start to interface with
xplainable at a lower level and start to use the more advanced features of the
package.