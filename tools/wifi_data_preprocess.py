"""Reusable CSI filtering and normalization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import savgol_filter


PipelineName = Literal["none", "median", "hampel_savgol"]


@dataclass(frozen=True)
class PreprocessConfig:
    """Configuration for CSI amplitude preprocessing.

    Detrending and robust z-score normalization are intentionally disabled by
    default so they can be switched on explicitly from an experiment notebook.
    """

    pipeline: PipelineName = "hampel_savgol"
    apply_detrend: bool = False
    apply_zscore: bool = False
    hampel_window: int = 7
    hampel_sigma: float = 3.0
    sg_window: int = 11
    sg_poly: int = 3
    median_window: int = 7
    detrend_window: int = 51
    scale_floor_percentile: float = 10.0
    z_clip: float | None = 8.0


def valid_odd_window(n: int, requested_window: int, min_window: int = 3) -> int:
    """Return an odd window size that fits into a signal of length n."""

    if n <= 1:
        return 1
    w = min(requested_window, n if n % 2 == 1 else n - 1)
    if w % 2 == 0:
        w -= 1
    return max(min_window, w)


def valid_savgol_window(n: int, requested_window: int, polyorder: int) -> int | None:
    """Return a valid Savitzky-Golay window, or None for too-short signals."""

    if n <= polyorder + 1:
        return None
    w = min(requested_window, n if n % 2 == 1 else n - 1)
    if w % 2 == 0:
        w -= 1
    min_w = polyorder + 2
    if min_w % 2 == 0:
        min_w += 1
    if w < min_w:
        return None
    return w


def hampel_filter_matrix(
    A: np.ndarray,
    window_size: int = 7,
    n_sigma: float = 3.0,
) -> np.ndarray:
    """Vectorized Hampel-like filter along time for a [T, S] or [T, D, S] array."""

    X = np.asarray(A, dtype=np.float32)
    w = valid_odd_window(X.shape[0], window_size, min_window=3)
    size = (w,) + (1,) * (X.ndim - 1)

    rolling_median = median_filter(X, size=size, mode="nearest")
    abs_dev = np.abs(X - rolling_median)
    rolling_mad = median_filter(abs_dev, size=size, mode="nearest")
    threshold = n_sigma * 1.4826 * (rolling_mad + 1e-8)

    return np.where(abs_dev > threshold, rolling_median, X).astype(np.float32)


def median_filter_matrix(A: np.ndarray, window_size: int = 7) -> np.ndarray:
    """Median filter along time for a [T, S] or [T, D, S] array."""

    X = np.asarray(A, dtype=np.float32)
    w = valid_odd_window(X.shape[0], window_size, min_window=3)
    size = (w,) + (1,) * (X.ndim - 1)
    return median_filter(X, size=size, mode="nearest").astype(np.float32)


def savgol_filter_matrix(
    A: np.ndarray,
    window_length: int = 11,
    polyorder: int = 3,
) -> np.ndarray:
    """Savitzky-Golay smoothing along time for a [T, S] or [T, D, S] array."""

    X = np.asarray(A, dtype=np.float32)
    w = valid_savgol_window(X.shape[0], window_length, polyorder)
    if w is None:
        return X.copy()
    return savgol_filter(X, window_length=w, polyorder=polyorder, axis=0, mode="nearest").astype(
        np.float32
    )


def robust_zscore(
    X: np.ndarray,
    axis: int = 0,
    scale_floor_percentile: float = 10.0,
    clip: float | None = 8.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Robust z-score with a shared lower scale floor to avoid MAD blow-ups."""

    A = np.asarray(X, dtype=np.float32)
    med = np.median(A, axis=axis, keepdims=True)
    mad = np.median(np.abs(A - med), axis=axis, keepdims=True)
    scale = 1.4826 * mad

    finite_scale = scale[np.isfinite(scale) & (scale > 0)]
    if finite_scale.size == 0:
        scale_floor = 1.0
    else:
        scale_floor = max(float(np.percentile(finite_scale, scale_floor_percentile)), 1e-6)

    scale = np.maximum(scale, scale_floor)
    Z = (A - med) / scale
    if clip is not None:
        Z = np.clip(Z, -clip, clip)

    return Z.astype(np.float32), med.astype(np.float32), scale.astype(np.float32), scale_floor


def rolling_mean_detrend(X: np.ndarray, window: int = 51) -> tuple[np.ndarray, np.ndarray]:
    """Subtract a rolling mean trend along time."""

    A = np.asarray(X, dtype=np.float32)
    size = max(1, min(int(window), A.shape[0]))
    trend = uniform_filter1d(A, size=size, axis=0, mode="nearest").astype(np.float32)
    return trend, (A - trend).astype(np.float32)


def preprocess_amplitude(
    amplitude: np.ndarray,
    config: PreprocessConfig | None = None,
    **overrides: object,
) -> dict[str, object]:
    """Preprocess CSI amplitude and keep intermediate arrays.

    The input can be either [T, S] for one receiver or [T, D, S] for multiple
    receivers recorded in parallel.
    """

    cfg = config or PreprocessConfig()
    if overrides:
        cfg = PreprocessConfig(**{**cfg.__dict__, **overrides})

    raw = np.asarray(amplitude, dtype=np.float32)
    out: dict[str, object] = {
        "pipeline": cfg.pipeline,
        "apply_detrend": cfg.apply_detrend,
        "apply_zscore": cfg.apply_zscore,
        "raw": raw.copy(),
    }

    if cfg.pipeline in ("none", "raw"):
        X = raw.copy()
    elif cfg.pipeline == "median":
        X = median_filter_matrix(raw, window_size=cfg.median_window)
        out["median_filtered"] = X
    elif cfg.pipeline == "hampel_savgol":
        X_hampel = hampel_filter_matrix(
            raw,
            window_size=cfg.hampel_window,
            n_sigma=cfg.hampel_sigma,
        )
        X = savgol_filter_matrix(
            X_hampel,
            window_length=cfg.sg_window,
            polyorder=cfg.sg_poly,
        )
        out["hampel"] = X_hampel
    else:
        raise ValueError("Unknown pipeline. Use 'none', 'median', or 'hampel_savgol'.")

    out["smoothed"] = X.copy()

    if cfg.apply_detrend:
        trend, X_work = rolling_mean_detrend(X, window=cfg.detrend_window)
    else:
        trend = np.zeros_like(X, dtype=np.float32)
        X_work = X.copy()

    out["trend"] = trend
    out["detrended"] = X_work.copy()

    if cfg.apply_zscore:
        X_norm, med, scale, scale_floor = robust_zscore(
            X_work,
            axis=0,
            scale_floor_percentile=cfg.scale_floor_percentile,
            clip=cfg.z_clip,
        )
    else:
        X_norm = X_work.copy()
        med = np.zeros((1,) + X.shape[1:], dtype=np.float32)
        scale = np.ones((1,) + X.shape[1:], dtype=np.float32)
        scale_floor = 1.0

    out["normalized"] = X_norm.astype(np.float32)
    out["normalization_median"] = med
    out["normalization_scale"] = scale
    out["normalization_scale_floor"] = scale_floor
    return out


def motion_energy(amplitude_like: np.ndarray, axis: tuple[int, ...] | None = None) -> np.ndarray:
    """Mean absolute temporal derivative for CSI-like arrays."""

    X = np.asarray(amplitude_like, dtype=np.float32)
    d = np.diff(X, axis=0)
    if axis is None:
        axis = tuple(range(1, d.ndim))
    return np.mean(np.abs(d), axis=axis)
