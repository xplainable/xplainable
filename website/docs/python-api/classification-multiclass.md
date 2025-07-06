---
sidebar_position: 3
---

# Classification – Multi-Class

:::info Coming Soon
**Multi-Class Classification** is currently in active development and will be available in an upcoming release of xplainable. We're working hard to bring you transparent, explainable multi-class models with the same ease of use you've come to expect.
:::

## What to Expect

When released, xplainable's Multi-Class Classification will provide:

<div className="row">
  <div className="col col--6">
    <div className="card">
      <div className="card__header">
        <h3>🎯 Multi-Class Support</h3>
      </div>
      <div className="card__body">
        <p>Handle classification problems with 3+ classes while maintaining full transparency and explainability.</p>
      </div>
    </div>
  </div>
  <div className="col col--6">
    <div className="card">
      <div className="card__header">
        <h3>🔍 Class-Specific Insights</h3>
      </div>
      <div className="card__body">
        <p>Understand what drives predictions for each individual class with detailed feature importance.</p>
      </div>
    </div>
  </div>
</div>

<div className="row">
  <div className="col col--6">
    <div className="card">
      <div className="card__header">
        <h3>⚡ Real-Time Explanations</h3>
      </div>
      <div className="card__body">
        <p>Get instant explanations for multi-class predictions with the same speed as binary classification.</p>
      </div>
    </div>
  </div>
  <div className="col col--6">
    <div className="card">
      <div className="card__header">
        <h3>🎨 GUI Integration</h3>
      </div>
      <div className="card__body">
        <p>Train and explore multi-class models using the intuitive xplainable GUI interface.</p>
      </div>
    </div>
  </div>
</div>

## Planned Features

### XMultiClassifier API
The upcoming `XMultiClassifier` will follow the same intuitive API pattern as our binary classifier:

```python
from xplainable.core.models import XMultiClassifier

# Simple, familiar API
model = XMultiClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Get explanations for each class
explanations = model.explain(X_test)
```

### GUI Integration
Train multi-class models with the embedded GUI:

```python
import xplainable as xp

# Initialize session
xp.initialise(api_key=os.environ['XP_API_KEY'])

# Train with GUI (coming soon)
model = xp.multiclass_classifier(data)
```

### Partitioned Multi-Class Models
Support for partitioned multi-class models for complex segmentation:

```python
from xplainable.core.models import PartitionedMultiClassifier

# Advanced partitioning (coming soon)
partitioned_model = PartitionedMultiClassifier(partition_on='segment')
```

## Current Alternatives

While we work on multi-class support, you can:

### 1. Use Binary Classification
For problems with 3+ classes, consider:
- **One-vs-Rest approach**: Train separate binary classifiers for each class
- **Binary decomposition**: Break down into multiple binary problems

### 2. Preprocessing Strategies
- **Class grouping**: Combine similar classes into broader categories
- **Hierarchical classification**: Use a tree-like structure of binary classifiers

### 3. Stay Updated
- **Follow our releases**: Check the [GitHub repository](https://github.com/xplainable/xplainable) for updates
- **Join our community**: Get notified when multi-class support is released

## Timeline

:::tip Development Status
Multi-class classification is a **high priority** feature currently in active development. We're targeting release in the coming months and will announce availability through our official channels.
:::

## Get Notified

Want to be the first to know when multi-class classification is available?

- ⭐ **Star our [GitHub repository](https://github.com/xplainable/xplainable)**
- 📧 **Follow our release notes**
- 💬 **Join our community discussions**

---

*In the meantime, explore our powerful [binary classification](./classification-binary.md) and [regression](./regression.md) capabilities, or check out [advanced topics](../advanced-topics/) for sophisticated modeling techniques.*