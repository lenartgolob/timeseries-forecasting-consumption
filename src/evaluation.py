"""
Metric utilities used by every model.
"""

import numpy as np
import pandas as pd


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def nrmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2)) / (y_true.max() - y_true.min())

def mape(y_true, y_pred):
    # Filter out values where actual is very close to zero to avoid infinite MAPE
    mask = np.abs(y_true) > 0.1  # Increase threshold to 0.1 kWh (more conservative)
    if mask.sum() == 0:  # If all values are near zero, return NaN
        return np.nan
    
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]
    
    return np.mean(np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered)) * 100


def compute_metrics(y_true, y_pred):
    """
    Returns a dict (💡  easy to turn into DataFrame row)
    """
    return dict(
        MAE=mae(y_true, y_pred),
        RMSE=rmse(y_true, y_pred),
        MAPE=mape(y_true, y_pred),
        nRMSE=nrmse(y_true, y_pred),
    )
