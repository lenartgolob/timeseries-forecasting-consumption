#!/usr/bin/env python3
"""
Test script to verify walk_forward logic for different training windows.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.walk_forward import splits

# Create a simple test dataset
dates = pd.date_range('2024-01-01', '2025-06-24', freq='h')
df = pd.DataFrame({
    'ds': dates,
    'y': range(len(dates))
})

print(f"Total data points: {len(df)}")
print(f"Date range: {df['ds'].min()} to {df['ds'].max()}")

# Test different training windows
test_windows = [
    ("all", 366),
    ("14d", 14),
    ("1m", 30),
    ("3m", 91)
]

for train_window, window_days in test_windows:
    print(f"\n=== Testing {train_window} (window_days={window_days}) ===")
    
    count = 0
    for train_df, test_df, cut_idx in splits(df, train_window, window_days):
        if count >= 3:  # Just show first 3 walks
            break
            
        train_start = train_df['ds'].iloc[0].strftime('%Y-%m-%d')
        train_end = train_df['ds'].iloc[-1].strftime('%Y-%m-%d')
        test_start = test_df['ds'].iloc[0].strftime('%Y-%m-%d')
        test_end = test_df['ds'].iloc[-1].strftime('%Y-%m-%d')
        
        print(f"  Walk {count+1}: Train {train_start} to {train_end} ({len(train_df)}h), Test {test_start} to {test_end} ({len(test_df)}h)")
        count += 1
    
    print(f"  ... (showing first {count} walks)")