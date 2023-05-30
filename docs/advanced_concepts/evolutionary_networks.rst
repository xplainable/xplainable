XEvolutionaryNetwork
=========================

Overview
-------------------------

``XEvolutionaryNetwork`` is a novel optimisation framework for ``XRegressor``
models.

It works by taking a pre-trained ``XRegressor`` model and fitting it with
training data to a network of optimisation layers. Each network layer is
responsible for optimising the model weights given a set of constraints.

The inspiration for the network concept came from deep learning frameworks but
is applied over additive models for weight optimisation that is understandable
and explainable.

What are Layers?
-------------------

Layers are the building blocks of ``XEvolutionaryNetwork``. Each layer runs
sequentially and optimises the model's weights given a set of constraints. There
are currently two types of layers:

**Tighten**
    This layer is a leaf-boosting method that optimises the weights of each leaf
    node in the model. It does this by using a gradient descent method to
    minimise the model's loss function. The default loss function is the
    mean absolute error of predictions calculated using the initial model
    training data.

    The name *Tighten* comes from the visual effect that the model has  when
    plotting the predictions of the model before and after the layer is run as
    the predictions are "tightened" around the training data.

    The ``Tighten`` layer brings determinism to the network and is used to
    improve the model's accuracy at a granular level. The deterministic nature
    of the layer means that it will always find a better set of weights for the 
    training data on each iteration – this can make it prone to overfitting.

**Evolve**
    This layer is a genetic algorithm that optimises the model weights
    by starting with a population of model weights and mutating them
    continuously until they produce a more optimal set of weights. The initial
    chromosomes are mutations of the current model weights, and the default
    fitness function is the mean absolute error of predictions. The genetic
    algorithm runs for a specified number of generations, and the best
    chromosome updates the final model weights.

    The ``Evolve`` layer brings stochasticity to the network and is used to
    escape local minima in the loss function. Its stochastic nature means that
    it will unlikely find weights that perfectly fit any minima, making it a
    stronger layer for avoiding overfitting earlier in the network, and a weaker
    layer later in the network.

While each layer is effective in isolation, they are more powerful when used
together.

A Typical Network
-------------------

A typical network will start and end with a ``Tighten`` layer, with one or more
Evolve layers in between. The ``Tighten`` layers find the nearest
minima to the current model weights, and the Evolve layers help to escape
local minima and find a set of weights that exist closer to a better minimum.

Example
~~~~~~~~~
::

    from xplainable.core.optimisation.genetic import XEvolutionaryNetwork
    from xplainable.core.optimisation.layers import Tighten, Evolve
    from xplainable.core.models import XRegressor
    from sklearn.model_selection import train_test_split

    # Load the data
    data = pd.read_csv("data.csv")
    x, y = data.drop("target", axis=1), data["target"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # Create the initial model
    model = XRegressor()
    model.fit(x_train, y_train)

    # Create the network
    network = XEvolutionaryNetwork(model)

    # Add the layers
    # Start with an initial Tighten layer
    network.add_layer(
        Tighten(
            iterations=100,
            learning_rate=0.1,
            early_stopping=20
            )
        )

    # Add an Evolve layer with a high severity
    network.add_layer(
        Evolve(
            mutations=100,
            generations=50,
            max_severity=0.5,
            max_leaves=20,
            early_stopping=20
            )
        )

    # Add another Evolve layer with a lower severity and reach
    network.add_layer(
        Evolve(
            mutations=100,
            generations=50,
            max_severity=0.3,
            max_leaves=15,
            early_stopping=20
            )
        )

    # Add a final Tighten layer with a low learning rate
    network.add_layer(
        Tighten(
            iterations=100,
            learning_rate=0.025,
            early_stopping=20
            )
        )

    # Fit the network (before or after adding layers)
    network.fit(x_train, y_train)

    # Run the network
    network.optimise()

    # Predict the test data
    y_pred = model.predict(x_test)


The above example has a lot to unpack, so let's go through it step by
step. First, we load the data and split it into training and test sets. Then we
create the initial model and fit it to the training data. This process is
vanilla data science and is the starting point for the network.

Next, we create the network::
    
    network = XEvolutionaryNetwork(model)

This line creates the network and allows it to update the model weights in
place. This characteristic is essential as each layer will permanently affect
the model weights from the point that the layer finishes.

Next, we add the layers to the network. We generally start with a ``Tighten``
layer as this will find the nearest minima to the current model weights::

    network.add_layer(
        Tighten(
            iterations=100,
            learning_rate=0.1,
            early_stopping=20
            )
        )


The Tighten layer has three parameters:

 * ``iterations``
 * ``learning_rate``
 * ``early_stopping``
 
The ``iterations`` parameter is the number of iterations that the leaf boosting
method will run for, and the ``learning_rate`` specifies how much a given weight
will update on each iteration. The ``early_stopping`` parameter is the number of
iterations that the layer will run without improving the loss function before it
stops.

Next, we add two Evolve layers::

    network.add_layer(
        Evolve(
            mutations=100,
            generations=50,
            max_severity=0.5,
            max_leaves=20,
            early_stopping=20
            )
        )

    # Other layer...

We generally add one or more Evolve layers after the initial ``Tighten`` layer
as this will allow the network to escape local minima and find its way to a
better minima.

The Evolve layer has five parameters:

 * ``mutations``
 * ``generations``
 * ``max_severity``
 * ``max_leaves``
 * ``early_stopping``

The ``mutations`` parameter is the number of mutations created for
each generation, and the ``generations`` parameter is the number of generations
that the genetic algorithm will run for.

The ``max_severity`` and ``max_leaves`` parameters dictate the significance of
each mutation. The ``max_severity`` parameter is the maximum severity of the
mutation relative to the current weights, and the ``max_leaves`` parameter is
the maximum number of leaf nodes the mutation can affect. Generally, at the
start of the network, we want to allow for significant mutations and then reduce
the severity and reach of the mutations as the network progresses.

The ``early_stopping`` parameter is the number of generations that the layer
will run without improving the loss function before it stops.

Finally, we add a final Tighten layer::

    network.add_layer(
        Tighten(
            iterations=100,
            learning_rate=0.025,
            early_stopping=20
            )
        )

You will notice that this layer has a lower learning rate than the initial
``Tighten`` layer. A low learning rate makes smaller adjustments to the model
weights as the network progresses to maximise our chances of finding a strong
minima.

Now that we have added the layers, we can fit the network to the training data::

    network.fit(x_train, y_train)

This line fits the network to the training data. At its core, it creates a
matrix of the model weights based on the model leaf nodes. The layers then use
this matrix to calculate better model weights. The fit method is independent of
the layers and is run before or after adding the layers.

Finally, we can run the network::

    network.optimise()

This line sequentially runs each layer in the network, updating the model
weights in place at the end of each layer. Once the network has finished
running, the model weights are updated to the final weights of the network.

It is possible to add more layers to an existing ``XEvolutionaryNetwork`` object
and continue to run the network. More, shorter layers can be useful if you want
closer control of performance monitoring.

Once the network finishes, we can simply use the model to make predictions
and explanations as we would typically::

    y_pred = model.predict(x_test)


Considerations
--------------
It's important to note the limitations of ``XEvolutionaryNetwork``. While
networks consistently improve the accuracy of a model, they have drawbacks.

Overfitting
~~~~~~~~~~~~
Long-running networks can be prone to overfitting if the network isn't
well-designed. Networks should be 3-6 layers with parameters suitable to the
training data and model structure. We recommend monitoring the performance of
the network with a validation set.

Time Complexity
~~~~~~~~~~~~~~~~
The ``Evolve`` layer can be computationally expensive, especially if the
``mutations`` parameter is high. We recommend experimenting with lower numbers
when training on a large dataset and increasing the number of mutations when
you understand how these parameters affect the network.

Reproducibility
~~~~~~~~~~~~~~~~
Due to the stochastic nature of the ``Evolve`` layer, it is not always possible
to reproduce the same results for long-running networks. We recommend using
a random seed to ensure that the network is reproducible.