---
sidebar_position: 4
---

# Regression

Xplainable provides a streamlined approach to training regression models, both through a GUI and programmatically using the Python API.

XRegressor is a feature-wise ensemble of decision trees, providing transparency and powerful predictive capabilities for regression problems. It's designed as an alternative to black-box models, offering real-time explainability:

:::info
**Key Features:** Custom algorithm, variable step function for each feature, real-time score explanation.

**Performance:** While powerful, XRegressor alone may be a weak predictor. Enhancements include using the optimise_tail_sensitivity method and fitting an XEvolutionaryNetwork.

**Customizability:** Allows granular model tuning with update_feature_params for specific features.
:::

## Using the GUI

Training an `XRegressor` model with xplainable's GUI simplifies the process significantly. Here's an example of how to do it:

```python
import xplainable as xp
import pandas as pd
import os

# Initialise your session
xp.initialise(api_key=os.environ['XP_API_KEY'])

# Load your data
data = pd.read_csv('data.csv')

# Train your model with GUI
model = xp.regressor(data)
```

&nbsp;

## Using the Python API
For more traditional programming, xplainable allows training of regression models using the Python API, similar to other machine learning libraries:

```python
from xplainable.core.models import XRegressor
from sklearn.model_selection import train_test_split
import pandas as pd

# Load and split data
data = pd.read_csv('data.csv')
x, y = data.drop('target', axis=1), data['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Train and optimise the model
model = XRegressor()
model.fit(x_train, y_train)
model.optimise_tail_sensitivity(x_train, y_train)

# Predict on test data
y_pred = model.predict(x_test)

```