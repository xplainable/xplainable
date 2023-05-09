from sklearn.metrics import mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt


def plot_error(model, x, y, alpha=0.5):
    fig = plt.figure(figsize=(12, 8))

    y_pred = model.predict(x)

    plt.scatter(y, y_pred, alpha=alpha)

    plt.plot(np.linspace(0, np.maximum(y, y_pred).max(), 2), \
        np.linspace(0, np.maximum(y, y_pred).max(), 2))

    plt.show()
