"""
Prophet-based forecasting model that follows the shared BaseModel API.
Facebook's Prophet is designed for time series with strong seasonal patterns.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import warnings

# Suppress Prophet's verbose logging
import logging
logging.getLogger('prophet').setLevel(logging.WARNING)
warnings.filterwarnings('ignore', category=FutureWarning)

from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

from ..config import FREQUENCY
from .base_model import BaseModel


class ProphetModel(BaseModel):
    """
    Prophet forecasting model with configurable seasonality and holidays.
    
    cfg keys (all optional — default table shown):
    -------------------------------------------------------------
    yearly_seasonality    | True        | Enable yearly seasonality
    weekly_seasonality    | True        | Enable weekly seasonality  
    daily_seasonality     | True        | Enable daily seasonality
    seasonality_mode      | 'additive'  | 'additive' or 'multiplicative'
    changepoint_prior_scale | 0.05      | Flexibility of trend changes
    seasonality_prior_scale | 10.0      | Flexibility of seasonality
    holidays_prior_scale  | 10.0        | Flexibility of holidays
    mcmc_samples          | 0           | MCMC samples for uncertainty
    interval_width        | 0.8         | Uncertainty interval width
    growth                | 'linear'    | 'linear' or 'logistic'
    """

    def __init__(self, cfg=None):
        super().__init__(cfg)
        
        # ------- defaults ----------------------------------------------------
        defaults = dict(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            seasonality_mode='additive',
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0,
            holidays_prior_scale=10.0,
            mcmc_samples=0,
            interval_width=0.8,
            growth='linear'
        )
        self.cfg = {**defaults, **(cfg or {})}

        # Prophet model instance (created in fit)
        self._model: Prophet | None = None
        
        # Store last training data for prediction
        self._last_train: pd.DataFrame | None = None

    # --------------------------------------------------------------------- #
    # Required BaseModel interface
    # --------------------------------------------------------------------- #
    def fit(self, df: pd.DataFrame) -> None:
        """
        df: DataFrame with columns ['ds','y'] — hourly, continuous.
        Creates a new Prophet model instance each time.
        """
        self._last_train = df.copy()
        
        # Create Prophet model with configuration
        self._model = Prophet(
            yearly_seasonality=self.cfg["yearly_seasonality"],
            weekly_seasonality=self.cfg["weekly_seasonality"],
            daily_seasonality=self.cfg["daily_seasonality"],
            seasonality_mode=self.cfg["seasonality_mode"],
            changepoint_prior_scale=self.cfg["changepoint_prior_scale"],
            seasonality_prior_scale=self.cfg["seasonality_prior_scale"],
            holidays_prior_scale=self.cfg["holidays_prior_scale"],
            mcmc_samples=self.cfg["mcmc_samples"],
            interval_width=self.cfg["interval_width"],
            growth=self.cfg["growth"]
        )
        
        # Fit the model
        self._model.fit(df)

    def predict(self, horizon: int) -> np.ndarray:
        if self._model is None or self._last_train is None:
            raise RuntimeError("Must call .fit() before .predict().")

        # Create future dataframe for the horizon
        future = self._model.make_future_dataframe(
            periods=horizon, 
            freq=FREQUENCY,
            include_history=False  # Only predict future periods
        )
        
        # Generate forecast
        forecast = self._model.predict(future)
        
        # Return only the point forecasts (yhat)
        return forecast['yhat'].values

    # ------------------------------------------------------------------ #
    # Optional helpers
    # ------------------------------------------------------------------ #
    def get_forecast_components(self, horizon: int) -> pd.DataFrame:
        """
        Get detailed forecast with trend, seasonality components.
        Useful for model interpretation.
        """
        if self._model is None:
            raise RuntimeError("Must call .fit() before getting components.")
            
        future = self._model.make_future_dataframe(
            periods=horizon, 
            freq=FREQUENCY,
            include_history=False
        )
        
        forecast = self._model.predict(future)
        
        # Return relevant columns
        components = ['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend']
        
        # Add seasonal components if they exist
        for col in ['yearly', 'weekly', 'daily']:
            if col in forecast.columns:
                components.append(col)
                
        return forecast[components]

    def cross_validate_model(self, initial: str = '730 days', period: str = '30 days', 
                           horizon: str = '24 hours') -> pd.DataFrame:
        """
        Perform Prophet's built-in cross-validation.
        Returns performance metrics across different cutoffs.
        """
        if self._model is None or self._last_train is None:
            raise RuntimeError("Must call .fit() before cross-validation.")
            
        # Perform cross validation
        df_cv = cross_validation(
            self._model, 
            initial=initial, 
            period=period, 
            horizon=horizon
        )
        
        # Calculate performance metrics
        df_performance = performance_metrics(df_cv)
        
        return df_performance

    def save(self, path: Path) -> None:
        """Save model configuration and parameters."""
        with open(Path(path) / "prophet_cfg.json", "w") as fp:
            json.dump(self.cfg, fp, indent=2)
            
        # Note: Prophet models are lightweight and fast to retrain,
        # so we don't save the actual fitted model object