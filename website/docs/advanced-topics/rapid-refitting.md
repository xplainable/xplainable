---
sidebar_position: 3
---

# Rapid Refitting
## Overview
The concept of refitting a model with a new set of parameters is commonplace in machine learning. Hyperparameter optimisation involves trialling several parameter sets, recording their effectiveness on model performance, and permanently fitting the best group. The problem with this approach is that it is computationally expensive, as the model must be refitted from scratch each time, setting the parameters across the entire dataset. Xplainable has implemented a method called `rapid refitting` to overcome this time complexity.

:::info
Rapid refitting is separate from refitting a model to new data. These docs cover the case where a model refits to the same data but with different parameters.
:::


## What is Rapid Refitting?
Rapid refitting is a novel concept introduced by xplainable. It involves making an initial fit on a set of training data with an initial set of parameters. Then, when you want to trial a new set of parameters, the model can be refit in a fraction of the initial fit time, even for large datasets.

This enormous speed improvement enables rapid and effective hyperparameter optimisation, as the model can be refit many times quickly, allowing for a more exhaustive search of the parameter space.

The concept of `partial fitting` is what enables rapid refitting.

## What is Partial Fitting?
The idea behind rapid refitting is only possible with the introduction of `partial fitting`. A partial fit occurs when the model first calculates the key metadata required for refitting and stores it in memory. This initial fit is the most time-expensive fitting stage, and the calculated metadata forms the base to refit the model with a new set of parameters.

The metadata contains information about the relationship between each feature and the target at each possible split value. Having this data stored means the model can refit to a new set of parameters without recalculating this metadata, making refitting incredibly rapid.

## Single-Feature Refitting
Rapid refitting is not limited to refitting the entire model with a new set of parameters – it also supports refitting a single feature. Single-feature refitting is a much faster and more precise parameter optimisation method, as you can adjust individual step functions in the ensemble.

By adjusting each step function individually, you can find the optimal function for each feature and combine them to create the optimal ensemble.

## Types of Feature Changes
When refitting a single feature, two types of changes can occur:

## Structural Change
A structural change is when the shape of the step function changes. In other words, when the feature refits to a new set of parameters that change the number of leaf nodes in the step function, it is considered a structural change. Structural changes generally occur when changing the `max_depth`, `min_leaf_size`, or `min_info_gain` parameters.

You should aim for structural changes when trying to fix large amounts of overfitting or underfitting.

## Weight Change
A weight change is when the shape of the step function remains the same, but the weights of the step function change. This change occurs when the feature refits to a new set of parameters that change the weights of the leaf nodes in the step function. Generally, changing the `weight`, `power_degree`, and `sigmoid_exponent` parameters for `XClassifier` models and the `tail_sensitivity` parameter for `XRegressor` models results in weight changes.

You should aim for weight changes when trying to fix small amounts of overfitting or underfitting.

## The Effects of Refitting
The effects of single-feature refitting are essential to understand and vary between the `XClassifier` and `XRegressor` models.

# XClassifier

Following every refit, the model normalises the weights across all features so that, at maximum, they sum to 1, and at minimum, they sum to 0. This normalisation stage means that when refitting a single feature, all other features will likely experience a small weight change.

When changing a highly influential feature, the other features will likely experience a slightly more significant weight change as changes to influential features can greatly affect the normalisation stage. Weight changes of other features are more pronounced after structural changes.

Any change to insignificant features will likely have a limited effect on other features, as the normalisation stage will have a very small effect.

# XRegressor

`XRegressor` models are affected differently to `XClassifier` models. There is no normalisation stage for `XRegressor` models, so each feature’s weights are independent. Independence means the other features will not be affected when refitting a single feature.

If you refit a feature you previously optimised in an `XEvolutionaryNetwork`, problems can arise as the `XEvolutionaryNetwork` will have already optimised the weights of the step function, and refitting the feature will reset these weights. To overcome this, you can run another `XEvolutionaryNetwork` on the refit feature to re-optimise the weights. This process will be much faster than re-optimising the entire model.

## The Effect of Alpha on Refitting
The `alpha` parameter is a generalisation parameter that controls the possible shapes of the step function. A higher `alpha` value will result in fewer steps and a more generalised model, whereas a lower `alpha` value will result in more steps and a more granular model. Larger values of `alpha` will result in faster partial-fitting and refitting as fewer possible split points require consideration.

`alpha` should be set between 0 and 1, with 0 being the most granular and 1 being the most generalised. The default value is 0.05.

The `alpha` parameter only affects the initial partial fit and is not used when refitting, meaning that the `alpha` parameter will not influence weights when refitting a single feature. You should use `alpha` primarily to control the possible shapes of the step function to control granularity and speed.
