Regression
=========================

Using the GUI
-------------------------------
Training an ``XRegressor`` model with the embedded xplainable GUI is easy. Run
the following lines of code, and you can configure and optimise your model
within the GUI to minimise the amount of code you need to write.

Examples
~~~~~~~~~~~~~~~~~~~~~~~

**GUI**
::
   
      import xplainable as xp
      import pandas as pd
      import os
      
      # Initialise your session
      xp.initialise(api_key=os.environ['XP_API_KEY'])

      # Load your data
      data = pd.read_csv('data.csv')

      # Train your model (this will open an embedded gui)
      model = xp.regressor(data)

Using the Python API
-------------------------------
You can also train an xplainable regression model programmatically. This
works in a very similar way to other popular machine learning libraries.

You can import the ``XRegressor`` class and train a model as follows:

Examples
~~~~~~~~~~~~~~~~~~~~~~~

**XRegressor**
::
      
      from xplainable.core.models import XRegressor
      from sklearn.model_selection import train_test_split
      import pandas as pd

      # Load data
      data = pd.read_csv('data.csv')
      x, y = data.drop('target', axis=1), data['target']
      x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

      # Train model
      model = XRegressor()
      model.fit(x_train, y_train)

      # Optimise the model
      model.optimise_tail_sensitivity(x_train, y_train)
      
      # <-- Add XEvolutionaryNetwork here -->

      # Predict on the test set
      y_pred = model.predict(x_test)


**PartitionedRegressor**
::
      
      from xplainable.core.models import PartitionedClassifier
      from xpainable.core.models import XClassifier
      import pandas as pd
      from sklearn.model_selection import train_test_split
      
      # Load your data
      data = pd.read_csv('data.csv')
      train, test = train_test_split(data, test_size=0.2)

      # Train your model (this will open an embedded gui)
      partitioned_model = PartitionedClassifier(partition_on='partition_column')

      # Iterate over the unique values in the partition column
      for partition in train['partition_column'].unique():
            # Get the data for the partition
            part = train[train['partition_column'] == partition]
            x_train, y_train = part.drop('target', axis=1), part['target']
            
            # Fit the embedded model
            model = XClassifier()
            model.fit(x, y)

            # Optimise the model
            model.optimise_tail_sensitivity(x_train, y_train)

            # <-- Add XEvolutionaryNetwork here -->

            # Add the model to the partitioned model
            partitioned_model.add_partition(model, partition)
      
      # Prepare the test data
      x_test, y_test = test.drop('target', axis=1), test['target']

      # Predict on the partitioned model
      y_pred = partitioned_model.predict(x_test)

Classes – Regressors
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: xplainable.core.ml.regression
    :members:
    :inherited-members:
    :undoc-members:
    :show-inheritance:

Classes – Regression Optimisation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Regression Optimisers can optimise ``XRegressor`` model weights to a
specific metric. They are used on top of pre-trained models and can be a
powerful tool for optimising models for maximum predictive power while
maintaining complete transparency.

**Example:**
::

      from xplainable.core.optimisation.genetic import XEvolutionaryNetwork
      from xplainable.core.optimisation.layers import Tighten, Evolve
      import pandas as pd
      from sklearn.model_selection import train_test_split

      # Load your data
      data = pd.read_csv('data.csv')
      x, y = data.drop('target', axis=1), data['target']
      x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

      # Train your model
      model = XRegressor()
      model.fit(x_train, y_train)
      model.optimise_tail_sensitivity(x_train, y_train)

      # Create an optimiser
      optimiser = XEvolutionaryNetwork(model)
      optimiser.fit(x_train, y_train)

      # Add a layers to tighten the model
      optimiser.add_layer(Tighten())
      optimiser.add_layer(Evolve())
      optimiser.add_layer(Evolve())
      optimiser.add_layer(Tighten())

      # Optimise the model weights in place
      optimiser.optimise()

      # Predict on the test set
      y_pred = model.predict(x_test)


.. automodule:: xplainable.core.optimisation.genetic
    :members:
    :inherited-members:
    :undoc-members:
    :show-inheritance:

.. automodule:: xplainable.core.optimisation.layers
    :members:
    :inherited-members:
    :undoc-members:
    :show-inheritance: