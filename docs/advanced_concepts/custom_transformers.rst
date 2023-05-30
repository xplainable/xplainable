Custom Transformers
=========================

Custom transformers are user-defined functions that transform 
data in an ``XPipeline``. They are defined in a separate file and imported into
a Python session.

Custom transformers inherit from the ``XBaseTransformer`` class in the xplainable
package. The ``XBaseTransformer`` class provides several methods you can override to customize the transformer's behaviour. The most important of these is the ``transform`` method, which every transformer in a pipeline will call.

You can find details about the ``XBaseTransformer`` class in the API documentation.

Example
----------
To create a custom transformer, create a Python file in your working
directory called ``custom_transformers.py``. In this file, define a class
that inherits from ``XBaseTransformer``::

    from xplainable.core.preprocessing.base import XBaseTransformer


    class MyTransformer(XBaseTransformer):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        # Optional
        def fit(self, data):
            # learn something from the data
            return data

        # Required
        def transform(self, data):
            # do something to the data
            return data

        # Optional
        def inverse_transform(self, data):
            # reverse the transformation
            return data

You can then import this class into a Python session and use it in an
``XPipeline``::

    from xplainable.preprocessing.pipeline import XPipeline
    from custom_transformers import MyTransformer

    pipeline = XPipeline([
        {"transformer": MyTransformer()},
    ])


Persisting Custom Transformers
--------------------------------

Documentation coming soon.


Embedding Custom Transformers in GUI
-------------------------------------

Documentation coming soon.