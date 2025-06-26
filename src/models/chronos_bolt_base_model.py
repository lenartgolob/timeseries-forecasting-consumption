"""
Chronos-bolt-base model from Amazon - a larger pretrained time series foundation model.
Zero-shot forecasting with transformer architecture (larger than tiny version).
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import warnings

import torch
from chronos import BaseChronosPipeline

from ..config import HORIZON_HOURS
from .base_model import BaseModel


class Chronos_bolt_baseModel(BaseModel):
    """
    Chronos-bolt-base forecasting model with zero-shot capabilities.
    Larger model with potentially better accuracy than tiny version.
    
    cfg keys (all optional — default table shown):
    -------------------------------------------------------------
    model_name          | "amazon/chronos-bolt-base" | HuggingFace model ID
    device              | "auto"      | Device: "auto", "cpu", "cuda", "mps"
    torch_dtype         | "auto"      | Torch dtype: "auto", "float16", "bfloat16"
    context_length      | None        | Max context length (None = use all available)
    """

    def __init__(self, cfg=None):
        super().__init__(cfg)
        
        # ------- defaults ----------------------------------------------------
        defaults = dict(
            model_name="amazon/chronos-bolt-base",
            device="auto",
            torch_dtype="auto", 
            context_length=None
        )
        self.cfg = {**defaults, **(cfg or {})}

        # Initialize model pipeline
        self._pipeline = None
        self._device = None
        self._torch_dtype = None
        
        # Store training data context
        self._context_data = None

    def _setup_pipeline(self):
        """Initialize the Chronos pipeline with configuration."""
        if self._pipeline is not None:
            return
            
        # Determine device
        if self.cfg["device"] == "auto":
            if torch.cuda.is_available():
                self._device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self._device = "mps" 
            else:
                self._device = "cpu"
        else:
            self._device = self.cfg["device"]
            
        # Determine torch dtype
        if self.cfg["torch_dtype"] == "auto":
            if self._device == "cuda":
                self._torch_dtype = torch.bfloat16
            else:
                self._torch_dtype = torch.float32
        else:
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32
            }
            self._torch_dtype = dtype_map.get(self.cfg["torch_dtype"], torch.float32)

        print(f"   Loading Chronos-bolt-base model on {self._device} with {self._torch_dtype}")
        
        # Load the model pipeline
        self._pipeline = BaseChronosPipeline.from_pretrained(
            self.cfg["model_name"],
            device_map=self._device,
            torch_dtype=self._torch_dtype
        )

    # --------------------------------------------------------------------- #
    # Required BaseModel interface
    # --------------------------------------------------------------------- #
    def fit(self, df: pd.DataFrame) -> None:
        """
        df: DataFrame with columns ['ds','y'] — hourly, continuous.
        Chronos doesn't require training, just stores context data.
        """
        # Initialize pipeline if needed
        self._setup_pipeline()
        
        # Store the time series data as context for prediction
        self._context_data = df['y'].values.astype(np.float32)
        
        # Apply context length limit if specified
        if self.cfg["context_length"] is not None:
            max_len = self.cfg["context_length"]
            if len(self._context_data) > max_len:
                self._context_data = self._context_data[-max_len:]

    def predict(self, horizon: int) -> np.ndarray:
        """
        Generate forecasts using the Chronos model.
        Returns the median forecast from the quantile predictions.
        """
        if self._pipeline is None or self._context_data is None:
            raise RuntimeError("Must call .fit() before .predict().")

        # Convert context to tensor
        context_tensor = torch.tensor(self._context_data, dtype=self._torch_dtype)
        
        # Generate forecast
        with torch.no_grad():
            forecast = self._pipeline.predict(
                context=context_tensor,
                prediction_length=horizon
            )
        
        # forecast shape: [1, num_samples, prediction_length]
        # Take median across samples for point forecast
        forecast_median = torch.median(forecast[0], dim=0)[0]
        
        return forecast_median.cpu().numpy()

    # ------------------------------------------------------------------ #
    # Optional helpers
    # ------------------------------------------------------------------ #
    def predict_quantiles(self, horizon: int, quantiles=[0.1, 0.5, 0.9]) -> pd.DataFrame:
        """
        Generate quantile forecasts for uncertainty quantification.
        Returns DataFrame with columns for each quantile.
        """
        if self._pipeline is None or self._context_data is None:
            raise RuntimeError("Must call .fit() before prediction.")

        context_tensor = torch.tensor(self._context_data, dtype=self._torch_dtype)
        
        with torch.no_grad():
            forecast = self._pipeline.predict(
                context=context_tensor,
                prediction_length=horizon
            )
        
        # forecast shape: [1, num_samples, prediction_length]
        forecast_samples = forecast[0].cpu().numpy()  # [num_samples, prediction_length]
        
        # Calculate quantiles
        quantile_forecasts = {}
        for q in quantiles:
            quantile_forecasts[f'q{int(q*100)}'] = np.quantile(forecast_samples, q, axis=0)
        
        return pd.DataFrame(quantile_forecasts)

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model_name": self.cfg["model_name"],
            "device": self._device,
            "torch_dtype": str(self._torch_dtype) if self._torch_dtype else None,
            "context_length": len(self._context_data) if self._context_data is not None else None,
            "num_parameters": sum(p.numel() for p in self._pipeline.model.parameters()) if self._pipeline else None
        }

    def save(self, path: Path) -> None:
        """Save model configuration."""
        config_to_save = self.cfg.copy()
        
        # Add runtime info
        config_to_save["runtime_info"] = self.get_model_info()
        
        with open(Path(path) / "chronos_bolt_base_cfg.json", "w") as fp:
            json.dump(config_to_save, fp, indent=2, default=str)