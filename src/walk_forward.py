"""
Walk-forward split generator supporting both expanding and rolling windows.
"""

import pandas as pd
from .config import TRAIN_WINDOW_HOURS, HORIZON_HOURS


def splits(df: pd.DataFrame, train_window: str, window_days: int, horizon=HORIZON_HOURS):
    """
    Generate walk-forward splits with configurable training windows.
    
    Args:
        df: Time series DataFrame with hourly data
        train_window: Window type ("all", "6m", "3m", "1m", "14d")
        window_days: Number of days for the training window
        horizon: Forecast horizon in hours
    
    For "all": Expanding window starting with 2024 (8784 hours), growing by 24h each walk
        Walk 1: Train on 8784h (2024), test on Jan 1, 2025
        Walk 2: Train on 8808h (2024+Jan1), test on Jan 2, 2025
        
    For rolling windows: Fixed window size, sliding forward
        Walk 1: Train on latest X days, test on next day
        Walk 2: Train on latest X days (shifted by 1), test on next day
    """
    window_hours = window_days * 24
    
    if train_window == "all":
        # Expanding window implementation (original behavior)
        start_train_hours = TRAIN_WINDOW_HOURS  # 8784 hours (2024)
        for walk_num, cut in enumerate(range(start_train_hours, len(df) - horizon + 1, 24)):
            # Expanding training window: 8784 + (walk_num * 24) hours
            current_train_hours = start_train_hours + (walk_num * 24)
            yield (
                df.iloc[0 : current_train_hours].reset_index(drop=True),
                df.iloc[cut : cut + horizon].reset_index(drop=True),
                cut,
            )
    else:
        # Rolling window implementation for 6m, 3m, 1m, 14d
        # Start from the point where we have enough data for the rolling window
        start_cut = window_hours
        for cut in range(start_cut, len(df) - horizon + 1, 24):
            # Rolling training window: always use latest window_hours
            train_start = cut - window_hours
            yield (
                df.iloc[train_start : cut].reset_index(drop=True),
                df.iloc[cut : cut + horizon].reset_index(drop=True),
                cut,
            )
