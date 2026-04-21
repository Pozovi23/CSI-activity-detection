#!/usr/bin/env python3
"""Feature preprocessing for classic binary CSI models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from tools.filters import median_filter


class Preprocessor:
    """Build one embedding from a parsed CSI DataFrame.

    Parameters are loaded from a preprocessing bundle produced by training.
    You can pass either a path to ``preprocessing.joblib`` or a dict-like bundle.
    """

    def __init__(self, weights: str | Path | dict[str, Any]) -> None:
        if isinstance(weights, (str, Path)):
            self.bundle = joblib.load(Path(weights))
        elif isinstance(weights, dict):
            self.bundle = dict(weights)
        else:
            raise TypeError("weights must be path-like or dict")

        self.median_width = int(self.bundle["median_width"])
        self.global_min = float(self.bundle["global_min"])
        self.global_max = float(self.bundle["global_max"])
        self.feature_cols = list(self.bundle["feature_cols"])
        self.feature_min = np.asarray(self.bundle["feature_min"], dtype=np.float64)
        self.feature_max = np.asarray(self.bundle["feature_max"], dtype=np.float64)
        self.scaler_pca = self.bundle["scaler_pca"]
        self.pca = self.bundle["pca"]
        self.k_95 = int(self.bundle["k_95"])

        if self.median_width % 2 == 0:
            raise ValueError("median_width must be odd")

    @staticmethod
    def _skewness(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64)
        mu = x.mean()
        sigma = x.std()
        if sigma == 0:
            return 0.0
        z = (x - mu) / sigma
        return float(np.mean(z ** 3))

    @staticmethod
    def _kurtosis_excess(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64)
        mu = x.mean()
        sigma = x.std()
        if sigma == 0:
            return 0.0
        z = (x - mu) / sigma
        return float(np.mean(z ** 4) - 3.0)

    @staticmethod
    def _spectral_features(signal: np.ndarray) -> tuple[float, float, float, float, float]:
        x = np.asarray(signal, dtype=np.float64)
        n = len(x)
        if n < 2:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        spectrum = np.fft.rfft(x)
        power = np.abs(spectrum) ** 2
        freqs = np.fft.rfftfreq(n, d=1.0)

        power_sum = power.sum()
        if power_sum <= 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        centroid = float((freqs * power).sum() / power_sum)
        spread = float(np.sqrt(((freqs - centroid) ** 2 * power).sum() / power_sum))
        p_norm = power / power_sum
        entropy = float(-(p_norm * np.log2(p_norm + 1e-12)).sum())
        dominant_freq = float(freqs[int(np.argmax(power))])
        rolloff = float(freqs[np.searchsorted(np.cumsum(power), 0.85 * power_sum)])

        return centroid, spread, entropy, dominant_freq, rolloff

    def _build_signal(self, df: pd.DataFrame) -> np.ndarray:
        amp_matrix = np.stack(df["amplitude"].to_numpy(), axis=0).astype(np.float64)
        raw_time_signal = amp_matrix.mean(axis=1)
        return median_filter(raw_time_signal, self.median_width)

    def _build_feature_row(self, df: pd.DataFrame) -> dict[str, float]:
        signal = self._build_signal(df)

        den = self.global_max - self.global_min
        if den <= 0:
            raise ValueError("global_max must be greater than global_min")

        scaled_signal = (signal - self.global_min) / den
        scaled_signal = np.clip(scaled_signal, 0.0, 1.0)

        mean_val = float(np.mean(scaled_signal))
        std_val = float(np.std(scaled_signal))
        median_val = float(np.median(scaled_signal))
        min_val = float(np.min(scaled_signal))
        max_val = float(np.max(scaled_signal))
        q25 = float(np.percentile(scaled_signal, 25))
        q75 = float(np.percentile(scaled_signal, 75))
        iqr = q75 - q25
        range_val = max_val - min_val
        rms = float(np.sqrt(np.mean(scaled_signal ** 2)))
        energy = float(np.sum(scaled_signal ** 2))
        zcr = float(np.mean(np.abs(np.diff(np.signbit(scaled_signal - mean_val)).astype(np.int8))))

        sk = self._skewness(scaled_signal)
        kt = self._kurtosis_excess(scaled_signal)
        sc, ss, se, dom_freq, rolloff = self._spectral_features(scaled_signal)

        return {
            "mean": mean_val,
            "std": std_val,
            "median": median_val,
            "skew": sk,
            "kurtosis": kt,
            "spectral_centroid": sc,
            "spectral_spread": ss,
            "spectral_entropy": se,
            "dominant_freq": dom_freq,
            "spectral_rolloff_85": rolloff,
            "min": min_val,
            "max": max_val,
            "q25": q25,
            "q75": q75,
            "iqr": iqr,
            "range": range_val,
            "rms": rms,
            "energy": energy,
            "zcr": zcr,
        }

    def preprocess(self, df: pd.DataFrame) -> np.ndarray:
        """Convert parsed CSI DataFrame into model embedding (1, k_95)."""
        feature_row = self._build_feature_row(df)
        feature_df = pd.DataFrame([feature_row])

        x = feature_df[self.feature_cols].to_numpy(dtype=np.float64)
        x = np.clip(x, self.feature_min, self.feature_max)

        x_std = self.scaler_pca.transform(x)
        x_pca = self.pca.transform(x_std)
        return x_pca[:, : self.k_95]
