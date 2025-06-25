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
    rmse_val = np.sqrt(np.mean((y_true - y_pred) ** 2))
    data_range = y_true.max() - y_true.min()
    
    # Avoid division by zero or very small ranges
    if data_range < 1e-8:
        return np.nan
    
    return rmse_val / data_range


def mape(y_true, y_pred):
    # Standard MAPE calculation - filtering handled at higher level
    mask = np.abs(y_true) > 1e-8  # Only avoid true zeros for division
    if mask.sum() == 0:
        return np.nan
    
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]
    
    return np.mean(np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered)) * 100


def compute_metrics(y_true, y_pred, apply_filter=False, threshold=0.1):
    """
    Returns a dict of metrics. If apply_filter=True, excludes periods with 
    actual consumption <= threshold from ALL metrics.
    """
    if apply_filter:
        mask = y_true > threshold
        if mask.sum() == 0:  # All values below threshold
            return dict(
                MAE=np.nan, RMSE=np.nan, MAPE=np.nan, nRMSE=np.nan,
                filtered_hours=0, total_hours=len(y_true)
            )
        
        y_true_filtered = y_true[mask]
        y_pred_filtered = y_pred[mask]
        
        return dict(
            MAE=mae(y_true_filtered, y_pred_filtered),
            RMSE=rmse(y_true_filtered, y_pred_filtered),
            MAPE=mape(y_true_filtered, y_pred_filtered),
            nRMSE=nrmse(y_true_filtered, y_pred_filtered),
            filtered_hours=mask.sum(),
            total_hours=len(y_true)
        )
    else:
        return dict(
            MAE=mae(y_true, y_pred),
            RMSE=rmse(y_true, y_pred),
            MAPE=mape(y_true, y_pred),
            nRMSE=nrmse(y_true, y_pred),
            filtered_hours=len(y_true),
            total_hours=len(y_true)
        )
