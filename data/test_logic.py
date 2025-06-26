#!/usr/bin/env python3
"""
Simple test to verify the walk_forward logic math without pandas.
"""

def test_window_logic():
    # Simulate data length (hours in dataset from 2024-01-01 to 2025-06-24)
    total_hours = 13008  # Approximate total hours in dataset
    horizon = 24  # 24-hour forecast horizon
    
    print(f"Dataset: {total_hours} hours total")
    print(f"Forecast horizon: {horizon} hours")
    
    test_cases = [
        ("all", 366),   # expanding window starting with 2024 (8784 hours)
        ("14d", 14),    # rolling 14-day window
        ("1m", 30),     # rolling 30-day window
        ("3m", 91),     # rolling 3-month window
    ]
    
    for train_window, window_days in test_cases:
        window_hours = window_days * 24
        print(f"\n=== {train_window} (window_days={window_days}, window_hours={window_hours}) ===")
        
        if train_window == "all":
            # Expanding window logic
            start_train_hours = 8784  # 2024 (366 days * 24 hours)
            walks = 0
            for cut in range(start_train_hours, total_hours - horizon + 1, 24):
                walks += 1
                current_train_hours = start_train_hours + ((walks-1) * 24)
                if walks <= 3:  # Show first 3 walks
                    print(f"  Walk {walks}: Train 0-{current_train_hours}h, Test {cut}-{cut+horizon}h")
            print(f"  Total walks: {walks}")
        else:
            # Rolling window logic
            start_cut = window_hours
            walks = 0
            for cut in range(start_cut, total_hours - horizon + 1, 24):
                walks += 1
                train_start = cut - window_hours
                if walks <= 3:  # Show first 3 walks
                    print(f"  Walk {walks}: Train {train_start}-{cut}h, Test {cut}-{cut+horizon}h")
            print(f"  Total walks: {walks}")

if __name__ == "__main__":
    test_window_logic()