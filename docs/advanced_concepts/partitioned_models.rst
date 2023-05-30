Partitioned Models
=========================

What are Partitioned Models?
-------------------------------
Partitioned models are a way of training xplainable models split into multiple
parts. Each part is a trained model on a subset of the data. The partitioned
model combines these models and often yields more accurate results than training
a single model on all the data. Partitioning the models can also provide a more
in-depth explanation of the model and overall data.

Individual partitions can be stand-alone models, or the partitioned-model object
can facilitate predictions across all partitions when the X data contains mixed
data across partitions.
   
It is important to note that partitioned models are not the same as an ensemble
of models. Partitioned models are trained on different data subsets and combined
to form a single model. It is also important to note that partitioned models,
like stand-alone models, are more likely to perform well when there is
sufficient data.

Example
---------------
You can train a partitioned model by creating a model object based from the
``BasePartition`` class and adding models to it. Each model has an associated
key (name) used to reference the model. These keys derive from the unique values
in a column identified as the *partition_on* column. So, when you pass X values
to the predict method of a partitioned model, the correct embedded model is used
for each respective row::

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
            model.fit(x_train, y_train)

            # Add the model to the partitioned model
            partitioned_model.add_partition(model, partition)
      
      # Prepare the test data
      x_test, y_test = test.drop('target', axis=1), test['target']

      # Predict on the partitioned model
      y_pred = partitioned_model.predict(x_test)


You can access the individual models in the partitioned model by calling the
``.partitions`` attribute. This attribute will return a dictionary of the
models, where the keys are the unique values in the *partition_on* column.

Every partitioned model will contain a default partition called ``__dataset__``.
This partition is the default model used when the *partition_on* column is not
present in the X data or if you pass an unknown partition to the predict method.
It is trained across the entire dataset and acts as a fallback.::

      # Access the default model
      model = partitioned_model.partitions['__dataset__']

