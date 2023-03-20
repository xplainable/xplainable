import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_log_error


def feature_bin_error(X, y_true, y_pred):

    df = pd.DataFrame(
            {
                'true': y_true,
                'pred': y_pred
            }
        )
    response_mse = []
    for idx in range(X.shape[1]):
        
        df['bin'] = pd.qcut(X[:,idx], 20)
        
        plot_data = df.groupby(['bin']).agg({
            'true': 'mean',
            'pred': 'mean'}).dropna()

        response_mse.append(mean_squared_log_error(
            plot_data['true'], plot_data['pred']))

    return np.mean(response_mse)
