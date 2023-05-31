
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-2-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

<div align="center">
<img src="https://github.com/xplainable/xplainable/blob/dev/docs/assets/logo/xplainable-logo.png">
<h1 align="center">Xplainable</h1>
<h3 align="center">Explainable machine learning for business optimisation</h3>
    
[![Python](https://img.shields.io/pypi/pyversions/xplainable)](https://pypi.org/project/xplainable/)
[![PyPi](https://img.shields.io/pypi/v/xplainable?color=blue)](https://pypi.org/project/statsforecast/)
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
x, y = data.drop('target', axis=1), data['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Train a model
model = xp.classifier(train)
```
<img src="https://github.com/xplainable/xplainable/blob/dev/docs/assets/gifs/gui_classifier.gif">

