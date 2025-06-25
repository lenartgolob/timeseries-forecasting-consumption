"""
Rolling-origin split generator with a **fixed training window**.
"""

import pandas as pd
from .config import TRAIN_WINDOW_HOURS, HORIZON_HOURS


def splits(df: pd.DataFrame, train_hours=TRAIN_WINDOW_HOURS, horizon=HORIZON_HOURS):
    """
    Expanding window starting with 2024 (8784 hours), growing by 24h each walk.
    Walk 1: Train on 8784h (2024), test on Jan 1, 2025
    Walk 2: Train on 8808h (2024+Jan1), test on Jan 2, 2025  
    Walk 3: Train on 8832h (2024+Jan1-2), test on Jan 3, 2025
    """
    for walk_num, cut in enumerate(range(train_hours, len(df) - horizon + 1, 24)):
        # Expanding training window: 8784 + (walk_num * 24) hours
        current_train_hours = train_hours + (walk_num * 24)
        yield (
            df.iloc[0 : current_train_hours].reset_index(drop=True),
            df.iloc[cut : cut + horizon].reset_index(drop=True),
            cut,
        )
