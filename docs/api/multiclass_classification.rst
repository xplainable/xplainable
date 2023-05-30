Classification – Multi-Class
================================

Important!
-----------------------------
Multi-Class is still being developed and is yet to be available in a 
release version of xplainable. Please check back soon for updates.

The following documentation is a preview of the functionality that will be
available in an upcoming release of xplainable.

Using the GUI
-------------------------------
Training a classification model with the embedded xplainable GUI is easy.
Run the following lines of code, and you can configure and optimise
your model within the GUI to minimise the amount of code you need to
write.

Example – GUI
~~~~~~~~~~~~~~~~~~~~~~~
::
   
      import xplainable as xp
      import pandas as pd
      
      # Initialise your session
      xp.initialise()

      # Load your data
      data = pd.read_csv('data.csv')

      # Train your model (this will open an embedded gui)
      model = xp.multiclass_classifier(data)

Using the Python API
------------------------
You can also train a multi-class classification model programmatically. This
works in a very similar way to other popular machine learning libraries.

You can import the ``XMultiClassifier`` class and train a model as follows:

Example – XMultiClassifier()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
::
      
      from xplainable.core.models import XMultiClassifier
      from sklearn.model_selection import train_test_split
      import pandas as pd

      # Load your data
      data = pd.read_csv('data.csv')
      x, y = data.drop('target', axis=1), data['target']
      x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

      # Train your model
      model = XMultiClassifier()
      model.fit(x_train, y_train)

      # Predict on the test set
      y_pred = model.predict(x_test)

Example – PartitionedMultiClassifier()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
::
      
      from xplainable.core.models import PartitionedMultiClassifier
      from xpainable.core.models import XMultiClassifier
      import pandas as pd
      from sklearn.model_selection import train_test_split
      
      # Load your data
      data = pd.read_csv('data.csv')
      train, test = train_test_split(data, test_size=0.2)

      # Train your model (this will open an embedded gui)
      partitioned_model = PartitionedMultiClassifier(partition_on='partition_column')

      # Iterate over the unique values in the partition column
      for partition in train['partition_column'].unique():
            # Get the data for the partition
            part = train[train['partition_column'] == partition]
            x_train, y_train = part.drop('target', axis=1), part['target']
            
            # Fit the embedded model
            model = XMultiClassifier()
            model.fit(x, y)

            # Add the model to the partitioned model
            partitioned_model.add_partition(model, partition)
      
      # Prepare the test data
      x_test, y_test = test.drop('target', axis=1), test['target']

      # Predict on the partitioned model
      y_pred = partitioned_model.predict(x_test)

