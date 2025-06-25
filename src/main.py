"""
Single CLI for running any model:

    python -m src.main --model prophet
"""

import argparse
import importlib
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import EXPERIMENT_DIR, PROCESSED_FILE, HORIZON_HOURS, TRAIN_WINDOW_DAYS, IGNORE_ZERO_CONSUMPTION, ZERO_THRESHOLD
from .evaluation import compute_metrics
from .walk_forward import splits
from .visualization import generate_plots


def load_model_class(name: str):
    mdl = importlib.import_module(f"src.models.{name}_model")
    return getattr(mdl, f"{name.capitalize()}Model")


def run(model_name: str):
    # ---------- directories --------------------------------------------------
    run_stamp = datetime.now().strftime("%Y-%m-%dT%H-%M") + f"-{model_name.capitalize()}"
    run_dir = EXPERIMENT_DIR / run_stamp
    fc_dir = run_dir / "forecasts"
    fc_dir.mkdir(parents=True, exist_ok=True)

    # ---------- model & config ----------------------------------------------
    cfg_path = Path("configs") / f"{model_name}.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    model = load_model_class(model_name)(cfg)

    params_out = run_dir / "params.json"
    params_out.write_text(json.dumps(cfg, indent=2))

    # ---------- data ---------------------------------------------------------
    df = pd.read_parquet(PROCESSED_FILE)

    # ---------- walk-forward -------------------------------------------------
    metrics_rows = []
    print(f"\n📊 Starting walk-forward validation...")
    print(f"   Training window: {TRAIN_WINDOW_DAYS} days ({TRAIN_WINDOW_DAYS*24} hours)")
    print(f"   Forecast horizon: {HORIZON_HOURS} hours")
    print(f"   Total data points: {len(df)}")
    
    for walk_num, (train_df, test_df, cut_idx) in enumerate(tqdm(splits(df), desc="walks"), 1):
        walk_start_time = time.time()
        
        # Log training and test periods
        train_start = train_df["ds"].iloc[0].strftime("%Y-%m-%d %H:%M")
        train_end = train_df["ds"].iloc[-1].strftime("%Y-%m-%d %H:%M") 
        test_start = test_df["ds"].iloc[0].strftime("%Y-%m-%d %H:%M")
        test_end = test_df["ds"].iloc[-1].strftime("%Y-%m-%d %H:%M")
        
        print(f"\n🔄 Walk {walk_num} (cut_idx={cut_idx}):")
        print(f"   Training: {train_start} → {train_end} ({len(train_df)} hours)")
        print(f"   Testing:  {test_start} → {test_end} ({len(test_df)} hours)")
        
        model.fit(train_df)
        y_hat = model.predict(HORIZON_HOURS)

        # save forecast (still use cut_idx for filename to maintain data alignment)
        (fc_dir / f"{model_name}_{cut_idx}.csv").write_text(
            pd.DataFrame({"ds": test_df["ds"], "y_hat": y_hat}).to_csv(index=False)
        )

        # Calculate execution time
        walk_time = time.time() - walk_start_time
        
        # Calculate average consumption for this day
        avg_consumption = test_df["y"].mean()
        
        # Apply filtering for evaluation (training uses all data)
        metrics_dict = compute_metrics(
            test_df["y"].values, y_hat, 
            apply_filter=IGNORE_ZERO_CONSUMPTION, 
            threshold=ZERO_THRESHOLD
        )
        
        mae_val = metrics_dict["MAE"]
        filtered_hours = metrics_dict["filtered_hours"]
        total_hours = metrics_dict["total_hours"]
        
        # Debug information for problematic cases
        if IGNORE_ZERO_CONSUMPTION and filtered_hours < 5:
            print(f"   WARNING: Very few hours retained ({filtered_hours}/{total_hours})")
            print(f"   Consumption range after filtering: {test_df['y'][test_df['y'] > ZERO_THRESHOLD].min():.3f} - {test_df['y'][test_df['y'] > ZERO_THRESHOLD].max():.3f}")
        
        print(f"   MAE: {mae_val:.3f}")
        print(f"   Avg consumption: {avg_consumption:.2f} kWh/h")
        if IGNORE_ZERO_CONSUMPTION:
            print(f"   Evaluated hours: {filtered_hours}/{total_hours} (>{ZERO_THRESHOLD} kWh)")
        print(f"   Walk time: {walk_time:.2f}s")
        
        metrics_rows.append(
            dict(walk=walk_num, cut_idx=cut_idx, walk_time=walk_time, avg_consumption=avg_consumption, 
                 **metrics_dict)
        )

    pd.DataFrame(metrics_rows).to_csv(run_dir / "metrics.csv", index=False)

    # ---------- finish -------------------------------------------------------
    # Calculate averages, handling NaN values
    valid_metrics = [row for row in metrics_rows if not np.isnan(row.get("MAE", np.nan))]
    
    avg_mae = sum(row["MAE"] for row in valid_metrics) / len(valid_metrics) if valid_metrics else np.nan
    avg_rmse = sum(row["RMSE"] for row in valid_metrics) / len(valid_metrics) if valid_metrics else np.nan
    avg_mape = sum(row["MAPE"] for row in valid_metrics if not np.isnan(row["MAPE"])) / len([r for r in valid_metrics if not np.isnan(r["MAPE"])]) if valid_metrics else np.nan
    avg_nrmse = sum(row["nRMSE"] for row in valid_metrics) / len(valid_metrics) if valid_metrics else np.nan
    avg_consumption = sum(row["avg_consumption"] for row in metrics_rows) / len(metrics_rows)
    avg_walk_time = sum(row["walk_time"] for row in metrics_rows) / len(metrics_rows)
    
    # Calculate total filtered hours
    total_filtered = sum(row["filtered_hours"] for row in metrics_rows)
    total_all_hours = sum(row["total_hours"] for row in metrics_rows)
    filter_percentage = (total_filtered / total_all_hours) * 100 if total_all_hours > 0 else 0
    
    print(f"\n✅ Completed {len(metrics_rows)} walks")
    print(f"   Average MAE: {avg_mae:.3f}")
    print(f"   Average RMSE: {avg_rmse:.3f}")
    print(f"   Average MAPE: {avg_mape:.2f}%")
    print(f"   Average nRMSE: {avg_nrmse:.3f}")
    print(f"   Average consumption: {avg_consumption:.2f} kWh/h")
    if IGNORE_ZERO_CONSUMPTION:
        print(f"   Evaluation coverage: {filter_percentage:.1f}% ({total_filtered}/{total_all_hours} hours)")
        print(f"   Zero threshold: >{ZERO_THRESHOLD} kWh")
    print(f"   Average walk time: {avg_walk_time:.2f}s")
    print(f"   Results saved to: {run_dir}")
    
    filter_info = f"\nEvaluation: {filter_percentage:.1f}% coverage (>{ZERO_THRESHOLD} kWh)" if IGNORE_ZERO_CONSUMPTION else ""
    
    run_info = f"""Completed OK - {len(metrics_rows)} walks
Average MAE: {avg_mae:.3f}
Average RMSE: {avg_rmse:.3f}
Average MAPE: {avg_mape:.2f}%
Average nRMSE: {avg_nrmse:.3f}
Average consumption: {avg_consumption:.2f} kWh/h{filter_info}
Average walk time: {avg_walk_time:.2f}s
"""
    (run_dir / "run_info.txt").write_text(run_info)
    
    # Generate visualizations
    try:
        generate_plots(run_dir, model_name)
    except Exception as e:
        print(f"⚠️  Warning: Failed to generate visualizations: {e}")
    
    print(f"\n📁 Run directory: {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=["prophet", "lightgbm", "chronos"])
    args = parser.parse_args()
    run(args.model)
