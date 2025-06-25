"""
LightGBM-based global regressor that follows the shared BaseModel API.
Uses Nixtla-MLForecast to auto-generate lag & calendar features.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor
from mlforecast import MLForecast
from mlforecast.utils import PredictionIntervals

from ..config import HORIZON_HOURS, FREQUENCY
from .base_model import BaseModel


class LightgbmModel(BaseModel):
    """
    cfg keys (all optional — default table shown):
    -------------------------------------------------------------
    lags            | [1,24]      | list of integer lags (hours)
    date_features   |             | any from MLForecast list
                      ["hour","dayofweek","month"]
    num_leaves      | 31
    learning_rate   | 0.05
    n_estimators    | 200
    """

    def __init__(self, cfg=None):
        super().__init__(cfg)
        # ------- defaults ----------------------------------------------------
        defaults = dict(
            lags=list(range(1, 25)),  # 24 lags (previous 24 h)
            date_features=["hour", "dayofweek", "month"],
            num_leaves=31,
            learning_rate=0.05,
            n_estimators=200,
        )
        self.cfg = {**defaults, **(cfg or {})}

        # build the scikit-style regressor
        self._reg = LGBMRegressor(
            num_leaves=self.cfg["num_leaves"],
            learning_rate=self.cfg["learning_rate"],
            n_estimators=self.cfg["n_estimators"],
            min_child_samples=self.cfg.get("min_child_samples", 20),
            verbosity=-1,  # Suppress LightGBM warnings  
            force_col_wise=True,  # Reduce overhead warnings
        )

        # MLForecast wrapper (created in fit because it needs lags/feats)
        self._mlf: MLForecast | None = None

        # persistent dataframe to hold latest training slice
        self._last_train: pd.DataFrame | None = None

    # --------------------------------------------------------------------- #
    # Required BaseModel interface
    # --------------------------------------------------------------------- #
    def fit(self, df: pd.DataFrame) -> None:
        """
        df: DataFrame with columns ['ds','y'] — hourly, continuous.
        Re-creates MLForecast each call to respect possibly updated lags.
        """
        self._last_train = df.copy()

        self._mlf = MLForecast(
            models=[self._reg],
            lags=self.cfg["lags"],
            date_features=self.cfg["date_features"],
            freq=FREQUENCY,
            num_threads=1,  # single-core for reproducibility
        )
        # PredictionIntervals only for quantiles if needed later
        self._mlf.fit(df, prediction_intervals=PredictionIntervals())

    def predict(self, horizon: int) -> np.ndarray:
        if self._mlf is None or self._last_train is None:
            raise RuntimeError("Must call .fit() before .predict().")

        # MLForecast expects a horizon integer
        fcst = self._mlf.predict(horizon)
        # the column is automatically named after the estimator
        y_hat = fcst[self._reg.__class__.__name__].values
        return y_hat

    # ------------------------------------------------------------------ #
    #  Optional helpers (dump hyper-params used in each run directory)
    # ------------------------------------------------------------------ #
    def save(self, path: Path) -> None:  # noqa: D401  (simple name)
        with open(Path(path) / "lightgbm_cfg.json", "w") as fp:
            json.dump(self.cfg, fp, indent=2)
