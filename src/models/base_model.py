"""
Every concrete model (Prophet, LightGBM, Chronos) must inherit from BaseModel.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class BaseModel(ABC):
    """Minimal, shared interface."""

    def __init__(self, cfg=None):
        self.cfg = cfg or {}

    # -------------- mandatory ------------------------------------------------
    @abstractmethod
    def fit(self, df: pd.DataFrame) -> None:
        """df has columns ['ds', 'y']; training happens in-place."""

    @abstractmethod
    def predict(self, horizon: int) -> np.ndarray:
        """Return a 1-D array of length == horizon."""

    # -------------- optional convenience ------------------------------------
    def save(self, path):
        pass

    def load(self, path):
        pass
