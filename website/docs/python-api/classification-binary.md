---
sidebar_position: 2
---

import BlogPost from "../../src/components/Cards/BlogPost.jsx";

# Classification – Binary

## Creating a Binary Classification model

Training an `XClassifier` model with the embedded xplainable GUI is easy. Run the following lines of code, and you can configure and optimise your model within the GUI to minimise the amount of code you need to write.

### Example – GUI

```python
import xplainable as xp
import pandas as pd
import os

# Initialise your session
xp.initialise(api_key=os.environ['XP_API_KEY'])

# Load your data
data = pd.read_csv('data.csv')

# Train your model (this will open an embedded gui)
model = xp.classifier(data)

```
&nbsp;

## Using the Python API
You can also train an xplainable classification model programmatically. This works in a very similar way to other popular machine learning libraries.

You can import the `XClassifier` class and train a model as follows:

### Example – XClassifier()
```python
from xplainable.core.models import XClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# Load your data
data = pd.read_csv('data.csv')
x, y = data.drop('target', axis=1), data['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Train your model
model = XClassifier()
model.fit(x_train, y_train)

# Predict on the test set
y_pred = model.predict(x_test)
```

### Example – PartitionedClassifier()

```python
from xplainable.core.models import PartitionedClassifier
from xplainable.core.models import XClassifier
import pandas as pd
from sklearn.model_selection import train_test_split

# Load your data
data = pd.read_csv('data.csv')
train, test = train_test_split(data, test_size=0.2)

# Train your model (this will open an embedded gui)
partitioned_model = PartitionedClassifier(partition_on='partition_column')

# Iterate over the unique values in the partition column
for partition in train['partition_column'].unique():
    part = train[train['partition_column'] == partition]
    x_train, y_train = part.drop('target', axis=1), part['target']
    
    model = XClassifier()
    model.fit(x_train, y_train)
    
    partitioned_model.add_partition(model, partition)

x_test, y_test = test.drop('target', axis=1), test['target']

y_pred = partitioned_model.predict(x_test)
```
&nbsp;

## Hyperparameter Optimisation
You can optimise `XClassifier` models automatically using the embedded GUI or 
programmatically using the Python API. The speed of hyperparameter optimisation 
with xplainable is much faster than traditional methods due to the concept of rapid 
refits first introduced by xplainable. You can find documentation on rapid refits 
in the advanced_concepts/rapid_refitting section.

The hyperparameter optimisation process uses a class called `XParamOptimiser` 
which is based on Bayesian optimisation using the Hyperopt library. Xplainable’s 
wrapper has pre-configured optimisation objectives and an easy way to set the 
search space for each parameter. You can find more details in the 
`XParamOptimiser` docs.

```Python
from xplainable.core.models import XClassifier
from xplainable.core.optimisation.bayesian import XParamOptimiser
from sklearn.model_selection import train_test_split
import pandas as pd

# Load your data
data = pd.read_csv('data.csv')
x, y = data.drop('target', axis=1), data['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Find optimised params
optimiser = XParamOptimiser(n_trials=200, n_folds=5, early_stopping=40)
params = optimiser.optimise(x_train, y_train)

# Train your optimised model
model = XClassifier(**params)
model.fit(x_train, y_train)
```

&nbsp;
<!-- 
<BlogPost 
    imgUrl="https://images.unsplash.com/photo-1556155092-490a1ba16284?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=4140&q=80" 
    tag="Tech" 
    title="ANALYSING CUSTOMER CHURN & RETENTION" 
    description="A walkthrough of customer churn drivers of a large telco in pursuit of a better retention strategy." 
/>

<BlogPost 
    imgUrl="https://images.unsplash.com/photo-1556155092-490a1ba16284?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=4140&q=80" 
    tag="Tech" 
    title="ANALYSING CUSTOMER CHURN & RETENTION" 
    description="A walkthrough of customer churn drivers of a large telco in pursuit of a better retention strategy." 
/> -->