# src/main.py
"""
Command-line entry point for every model.

Example:
    python -m src.main --model lightgbm
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .config import (
    EXPERIMENT_DIR,
    PROCESSED_FILE,
    HORIZON_HOURS,
)
from .evaluation import compute_metrics
from .walk_forward import splits


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def load_model_class(model_name: str):
    """
    Dynamically import `<repo>/src/models/{model_name}_model.py`
    and return the `<CamelCase>Model` class inside.
    """
    module = importlib.import_module(f"src.models.{model_name}_model")
    class_name = f"{model_name.capitalize()}Model"
    return getattr(module, class_name)


def load_cfg(model_name: str) -> dict:
    cfg_path = Path("configs") / f"{model_name}.json"
    return json.loads(cfg_path.read_text()) if cfg_path.exists() else {}


# --------------------------------------------------------------------------- #
# Main run logic
# --------------------------------------------------------------------------- #
def run(model_name: str):
    # 1: data
    df = pd.read_parquet(PROCESSED_FILE)

    # 2: model
    cfg = load_cfg(model_name)
    ModelClass = load_model_class(model_name)
    model = ModelClass(cfg)

    # 3: prepare run directory
    stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = EXPERIMENT_DIR / f"{stamp}-{model_name.capitalize()}"
    forecasts_dir = run_dir / "forecasts"
    forecasts_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "params.json").write_text(json.dumps(cfg, indent=2))

    # 4: walk-forward loop
    metrics_rows = []
    t0 = time.time()
    for train_df, test_df, cut_idx in tqdm(splits(df), desc=f"{model_name} walks"):
        model.fit(train_df)
        y_hat = model.predict(HORIZON_HOURS)

        # 4a: save forecast (24 timestamps, one CSV per walk)
        pd.DataFrame({"ds": test_df["ds"], "y_hat": y_hat}).to_csv(
            forecasts_dir / f"{model_name}_{cut_idx}.csv", index=False
        )

        # 4b: compute & store metrics
        metrics_rows.append(
            dict(walk=cut_idx, **compute_metrics(test_df["y"].values, y_hat))
        )

    # 5: dump aggregated metrics & timing
    pd.DataFrame(metrics_rows).to_csv(run_dir / "metrics.csv", index=False)
    runtime = time.time() - t0
    (run_dir / "run_info.txt").write_text(
        f"runtime_seconds: {runtime:.1f}\n"
        f"num_walks: {len(metrics_rows)}\n"
        f"model: {model_name}\n"
    )
    print(f"✅  finished → {run_dir.relative_to(Path.cwd())}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run walk-forward evaluation.")
    parser.add_argument(
        "--model",
        required=True,
        choices=["lightgbm", "prophet", "chronos"],
        help="Which model implementation in src/models/ to use.",
    )
    args = parser.parse_args()
    run(args.model)
