#!/usr/bin/env python3
"""Binary classifier ensemble for three ESP-specific models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np


class BinaryClassificator:
    """Majority-vote binary classifier over three ESP models."""

    def __init__(
        self,
        weights_dev1: str | Path | Any,
        weights_dev2: str | Path | Any,
        weights_dev3: str | Path | Any,
    ) -> None:
        self.model_dev1 = self._load_model(weights_dev1)
        self.model_dev2 = self._load_model(weights_dev2)
        self.model_dev3 = self._load_model(weights_dev3)

    @staticmethod
    def _load_model(weights: str | Path | Any) -> Any:
        if isinstance(weights, (str, Path)):
            return joblib.load(Path(weights))
        if hasattr(weights, "predict"):
            return weights
        raise TypeError("Classifier weights must be path-like or sklearn-like model with predict")

    @staticmethod
    def _predict_one(model: Any, embedding: np.ndarray) -> int:
        x = np.asarray(embedding, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        pred_raw = model.predict(x)
        pred = int(np.asarray(pred_raw).reshape(-1)[0])
        if pred not in (0, 1):
            raise ValueError(f"Expected binary class 0/1, got {pred}")
        return pred

    def predict(self, embedding_dev1: np.ndarray, embedding_dev2: np.ndarray, embedding_dev3: np.ndarray) -> int:
        """Predict binary class by majority vote across three ESP models."""
        pred1 = self._predict_one(self.model_dev1, embedding_dev1)
        pred2 = self._predict_one(self.model_dev2, embedding_dev2)
        pred3 = self._predict_one(self.model_dev3, embedding_dev3)

        votes_dynamic = int(pred1 == 1) + int(pred2 == 1) + int(pred3 == 1)
        return 1 if votes_dynamic >= 2 else 0
