import numpy as np
import pandas as pd
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
    results["roc_auc"] = roc_auc_score(y_true, y_pred)
    
    # Precision-recall curve
    precision = [score["precision"] for score in scores]
    recall = [score["recall"] for score in scores]
    results["precision_recall_curve"] = {"precision": precision, "recall": recall}
    
    # AUC-PR
    results["auc_pr"] = auc(recall, precision)
    
    # Matthews Correlation Coefficient (MCC)
    results["mcc"] = matthews_corrcoef(y_true, np.round(y_pred))
    
    # Log Loss (Cross-Entropy Loss)
    results["log_loss"] = log_loss(y_true, y_pred)

    return results

def evaluate_regression(y_true, y_pred):
    results = {
        "charts": {
            'true': y_true.values if len(y_true) < 10000 else y_true[:10000].values,
            'prediction': y_pred if len(y_pred) < 10000 else y_pred[:10000],
        }
    }
    
    # Mean Absolute Error (MAE)
    results["mae"] = mean_absolute_error(y_true, y_pred)
    
    # Mean Squared Error (MSE)
    results["mse"] = mean_squared_error(y_true, y_pred)
    
    # Root Mean Squared Error (RMSE)
    results["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # R-squared (R2) Score
    results["r2_score"] = r2_score(y_true, y_pred)
    
    # Explained Variance Score
    results["explained_variance_score"] = explained_variance_score(y_true, y_pred)
    
    # Mean Squared Logarithmic Error (MSLE)
    results["msle"] = mean_squared_log_error(y_true, y_pred)
    
    # Root Mean Squared Logarithmic Error (RMSLE)
    results["rmsle"] = np.sqrt(mean_squared_log_error(y_true, y_pred))
    
    # Mean Absolute Percentage Error (MAPE)
    results["mape"] = mean_absolute_percentage_error(y_true, y_pred)
    
    return results