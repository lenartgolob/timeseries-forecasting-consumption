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

from .config import EXPERIMENT_DIR, PROCESSED_FILE, HORIZON_HOURS, TRAIN_WINDOW_DAYS
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
        
        # metrics
        mae_val = compute_metrics(test_df["y"].values, y_hat)["MAE"]
    
        print(f"   MAE: {mae_val:.3f}")
        print(f"   Avg consumption: {avg_consumption:.2f} kWh/h")
        print(f"   Walk time: {walk_time:.2f}s")
        
        metrics_rows.append(
            dict(walk=walk_num, cut_idx=cut_idx, walk_time=walk_time, avg_consumption=avg_consumption, 
                 **compute_metrics(test_df["y"].values, y_hat))
        )

    pd.DataFrame(metrics_rows).to_csv(run_dir / "metrics.csv", index=False)

    # ---------- finish -------------------------------------------------------
    avg_mae = sum(row["MAE"] for row in metrics_rows) / len(metrics_rows)
    avg_rmse = sum(row["RMSE"] for row in metrics_rows) / len(metrics_rows)
     # Handle NaN values in MAPE calculation
    mape_values = [row["MAPE"] for row in metrics_rows if not np.isnan(row["MAPE"])]
    avg_mape = sum(mape_values) / len(mape_values) if mape_values else 0
    avg_nrmse = sum(row["nRMSE"] for row in metrics_rows) / len(metrics_rows)
    avg_consumption = sum(row["avg_consumption"] for row in metrics_rows) / len(metrics_rows)
    avg_walk_time = sum(row["walk_time"] for row in metrics_rows) / len(metrics_rows)
    
    print(f"\n✅ Completed {len(metrics_rows)} walks")
    print(f"   Average MAE: {avg_mae:.3f}")
    print(f"   Average RMSE: {avg_rmse:.3f}")
    print(f"   Average MAPE: {avg_mape:.2f}%")
    print(f"   Average nRMSE: {avg_nrmse:.3f}")
    print(f"   Average consumption: {avg_consumption:.2f} kWh/h")
    print(f"   Average walk time: {avg_walk_time:.2f}s")
    print(f"   Results saved to: {run_dir}")
        
    run_info = f"""Completed OK - {len(metrics_rows)} walks
Average MAE: {avg_mae:.3f}
Average RMSE: {avg_rmse:.3f}
Average MAPE: {avg_mape:.2f}%
Average nRMSE: {avg_nrmse:.3f}
Average consumption: {avg_consumption:.2f} kWh/h
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
