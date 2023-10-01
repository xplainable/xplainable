""" Copyright Xplainable Pty Ltd, 2023"""

import numpy as np
import pandas as pd
from ..utils.encoders import force_json_compliant
from sklearn.metrics import (
    roc_auc_score, matthews_corrcoef, log_loss, auc, mean_absolute_error,
    mean_squared_error, r2_score, explained_variance_score,
    mean_squared_log_error, mean_absolute_percentage_error)


def calculate_classification_metrics(tp, fp, tn, fn):
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    specificity = tn / (tn + fp + 1e-9)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
    
    return {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "accuracy": accuracy,
        "f1": f1
        }

def calculate_probability_bins(y_true, y_prob):
    output_data = []

    for cls in np.unique(y_true):
        counts = np.bincount((y_prob[y_true == cls]*100).round().astype(int))

        counts = np.concatenate(
            [counts, np.zeros(101 - len(counts), dtype=int)])

        output_data.append({
            "class": cls,
            "values": list(counts)
        })
    
    return output_data

def calculate_regression_bins(y_true, y_pred, bin_count):

    # Ensure y_true and y_pred are NumPy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate the min and max of the predicted values
    min_val = min(y_pred)
    max_val = max(y_pred)

    # Define the bins
    bins = np.linspace(min_val, max_val, bin_count+1)

    # Assign each predicted value to a bin
    true_bin_indices = np.digitize(y_true, bins)
    pred_bin_indices = np.digitize(y_pred, bins)

    # For each bin, count the number of true values
    true_output = []
    pred_output = []
    for i in range(1, bin_count+1):
        true_values_in_bin = y_true[true_bin_indices == i]
        true_counts = len(true_values_in_bin)
        true_output.append(true_counts)
        
        pred_values_in_bin = y_pred[pred_bin_indices == i]
        pred_counts = len(pred_values_in_bin)
        pred_output.append(pred_counts)
    
    output = [
        {
            "class": "true",
            "values": true_output
        },
        {
            "class": "pred",
            "values": pred_output
        }
    ]
    
    return output

def evaluate_classification(y_true, y_pred):
    results = {}
    thresholds = np.linspace(0, 1, 101)
    
    # TP, FP, TN, FN at each threshold value
    metrics = []
    for threshold in thresholds:
        y_pred_thresholded = np.round(y_pred > threshold)
        tp = np.sum((y_true == 1) & (y_pred_thresholded == 1))
        fp = np.sum((y_true == 0) & (y_pred_thresholded == 1))
        tn = np.sum((y_true == 0) & (y_pred_thresholded == 0))
        fn = np.sum((y_true == 1) & (y_pred_thresholded == 0))
        metrics.append({
            "threshold": threshold,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn
            })
    
    # Compute other metrics based on TP, FP, TN, and FN
    scores = []
    for metric in metrics:
        scores.append({
            **metric, **calculate_classification_metrics(metric["tp"],
            metric["fp"], metric["tn"], metric["fn"])})

    results["scores"] = scores
    
    # ROC at each threshold value
    fpr = [score["fp"] / (score["fp"] + score["tn"] + 1e-9) for score in scores]
    tpr = [score["tp"] / (score["tp"] + score["fn"] + 1e-9) for score in scores]
    results["roc"] = {"fpr": fpr, "tpr": tpr}
    
    # ROC-AUC
    try:
        results["roc_auc"] = roc_auc_score(y_true, y_pred)
    except:
        results["roc_auc"] = np.nan
    
    # Precision-recall curve
    precision = [score["precision"] for score in scores]
    recall = [score["recall"] for score in scores]
    results["precision_recall_curve"] = {
        "precision": precision, "recall": recall}
    
    # AUC-PR
    try:
        results["auc_pr"] = auc(recall, precision)
    except:
        results["auc_pr"] = np.nan
    
    # Matthews Correlation Coefficient (MCC)
    try:
        results["mcc"] = matthews_corrcoef(y_true, np.round(y_pred))
    except:
        results["mcc"] = np.nan
    
    # Log Loss (Cross-Entropy Loss)
    try:
        results["log_loss"] = log_loss(y_true, y_pred)
    except:
        results["log_loss"] = np.nan

    # Calculates the number of predictions in each probability bin
    results["probability_bins"] = calculate_probability_bins(y_true, y_pred)

    return force_json_compliant(results)

def evaluate_regression(y_true, y_pred):
    results = {
        "charts": {
            'true': list(y_true.values if len(y_true) < 10000 else y_true[:10000].values),
            'prediction': list(y_pred if len(y_pred) < 10000 else y_pred[:10000]),
        }
    }
    
    # Mean Absolute Error (MAE)
    try:
        results["mae"] = mean_absolute_error(y_true, y_pred)
    except:
        results["mae"] = np.nan
    
    # Mean Squared Error (MSE)
    try:
        results["mse"] = mean_squared_error(y_true, y_pred)
    except:
        results["mse"] = np.nan
    
    # Root Mean Squared Error (RMSE)
    try:
        results["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
    except:
        results["rmse"] = np.nan

    # R-squared (R2) Score
    try:
        results["r2_score"] = r2_score(y_true, y_pred)
    except:
        results["r2_score"] = np.nan
    
    # Explained Variance Score
    try:
        results["explained_variance_score"] = explained_variance_score(
            y_true, y_pred)
    except:
        results["explained_variance_score"] = np.nan
    
    # Mean Squared Logarithmic Error (MSLE)
    try:
        results["msle"] = mean_squared_log_error(y_true, y_pred)
    except:
        results["msle"] = np.nan
    
    # Root Mean Squared Logarithmic Error (RMSLE)
    try:
        results["rmsle"] = np.sqrt(mean_squared_log_error(y_true, y_pred))
    except:
        results["rmsle"] = np.nan
    
    # Mean Absolute Percentage Error (MAPE)
    try:
        results["mape"] = mean_absolute_percentage_error(y_true, y_pred)
    except:
        results["mape"] = np.nan

    results["prediction_bins"] = calculate_regression_bins(y_true, y_pred, 100)
    results["observed_min"] = np.nanmin(y_true)
    results["observed_max"] = np.nanmax(y_true)
    
    return force_json_compliant(results)
