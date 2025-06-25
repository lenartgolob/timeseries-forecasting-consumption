from pathlib import Path

# ---------- Paths -----------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
EXPERIMENT_DIR = ROOT_DIR / "experiments"
FIG_DIR = EXPERIMENT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Forecasting parameters -----------------------------------------
HORIZON_HOURS = 24            # forecast length
TRAIN_WINDOW_DAYS = 366       # full 2024 year (leap year) for training
TRAIN_WINDOW_HOURS = TRAIN_WINDOW_DAYS * 24  # convert to hours (8784)
FREQUENCY = "h"               # hourly after aggregation

# ---------- Evaluation parameters -------------------------------------------
IGNORE_ZERO_CONSUMPTION = True  # Filter out low consumption periods during testing
ZERO_THRESHOLD = 0.1         # kWh threshold for "zero" consumption (lowered from 0.1)

# ---------- File names ------------------------------------------------------
PROCESSED_FILE = DATA_PROCESSED_DIR / "hourly.parquet"
RAW_CSV_FILES = [
    DATA_RAW_DIR / "2-8008081-15minMeritve2024-01-01-2025-01-01.csv",
    DATA_RAW_DIR / "2-8008081-15minMeritve2025-01-01-2025-06-24.csv"
]
POWER_COLUMN = "Energija A+"
TIMESTAMP_COLUMN = "Časovna značka"

# ---------- Misc ------------------------------------------------------------
RANDOM_SEED = 42
