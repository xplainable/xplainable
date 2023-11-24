---
sidebar_position: 1
---

# Preprocessing

Xplainable offers a preprocessing module that allows you to build reproducible 
preprocessing pipelines. The module aims to rapidly develop and deploy pipelines 
in production environments and play friendly with ipywidgets.

The preprocessing module is built on the `XPipeline` class from xplainable and 
is used similarly to the scikit-learn Pipeline class. All transformers in the 
pipeline are expected to have a fit and transform method, along with an inverse_transform method.

To create custom transformers, you can inherit from the `XBaseTransformer` 
class. You can render these custom transformers in the embedded xplainable GUI, 
which allows you to build pipelines without writing any code. You can find 
documentation on how to embed them in the GUI in the advanced_concepts/custom_transformers 
section.

## Using the GUI

Xplainable's GUI simplifies the creation of preprocessing pipelines. Here's an example of its usage:

```python
import xplainable as xp
import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv('data.csv')
train, test = train_test_split(data, test_size=0.2, random_state=42)

# Instantiate the preprocessor object
pp = xp.Preprocessor()

# Open the GUI and build pipeline
pp.preprocess(train)

# Apply the pipeline on new data
test_transformed = pp.transform(test)
```

&nbsp;

## Using the Python API
You can develop preprocessing pipelines using the Python API with `XPipeline`. 
The following example shows how to build a pipeline.

### Example

```Python
from xplainable.preprocessing import transformers as xtf
from xplainable.preprocessing.pipeline import XPipeline
from sklearn.model_selection import train_test_split
import pandas as pd

# Load data
data = pd.read_csv('data.csv')
train, test = train_test_split(data, test_size=0.2, random_state=42)

# Instantiate a pipeline
pipeline = XPipeline()

# Add stages for specific features
pipeline.add_stages([
    {"feature": "age", "transformer": xtf.Clip(lower=18, upper=99)},
    {"feature": "balance", "transformer": xtf.LogTransform()}
])

# add stages on multiple features
pipeline.add_stages([
    {"transformer": xtf.FillMissing({'job': 'mode', 'age': 'mean'})},
    {"transformer": xtf.DropCols(columns=['duration', 'campaign'])}
])

# Share a single transformer across multiple features.
# Note this can only be applied when no fit method is required.
upper_case = xtf.ChangeCase(case='upper')

pipeline.add_stages([
    {"feature": "job", "transformer": upper_case},
    {"feature": "month", "transformer": upper_case}
])

# Fit and transform the data
train_transformed = pipeline.fit_transform(train)

# Apply transformations on new data
test_transformed = pipeline.transform(test)

# Inverse transform (only applies to configured features)
test_inv_transformed = pipeline.inverse_transform(test_transform)

```