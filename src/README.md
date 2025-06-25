# Battery-Scheduling Load-Forecasting

End-to-end code accompanying our 4-page paper.

## Quick start

```bash
# 1. clone & install
pip install -r requirements.txt   # or poetry install

# 2. build hourly dataset (one-off)
python -m src.preprocessing

# 3. run a model (replace prophet/lightgbm/chronos)
python -m src.main --model prophet

# 4. standard plots
python - <<'PY'
from src.visualization import plot_daily_example, plot_learning_curve
plot_daily_example(["prophet", "lightgbm", "chronos"], start_idx=-30)
plot_learning_curve(["prophet", "lightgbm", "chronos"])
PY
