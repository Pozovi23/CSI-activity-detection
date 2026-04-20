"""Signal filtering helpers used in CSI notebooks."""

from __future__ import annotations

import numpy as np


def _validate_window(window_size: int) -> int:
	"""Validate and normalize odd window size for filters."""
	if not isinstance(window_size, int):
		raise TypeError("window_size must be int")
	if window_size < 1:
		raise ValueError("window_size must be >= 1")
	if window_size % 2 == 0:
		raise ValueError("window_size must be odd")
	return window_size


def moving_average_filter(signal: np.ndarray | list[float], window_size: int) -> np.ndarray:
	"""Return moving-average smoothed 1D signal."""
	w = _validate_window(window_size)
	x = np.asarray(signal, dtype=np.float32)
	if x.ndim != 1:
		raise ValueError("signal must be 1D")

	kernel = np.ones(w, dtype=np.float32) / float(w)
	return np.convolve(x, kernel, mode="same")


def median_filter(signal: np.ndarray | list[float], window_size: int) -> np.ndarray:
	"""Return median-filtered 1D signal using edge padding."""
	w = _validate_window(window_size)
	x = np.asarray(signal, dtype=np.float32)
	if x.ndim != 1:
		raise ValueError("signal must be 1D")

	half = w // 2
	padded = np.pad(x, pad_width=half, mode="edge")
	out = np.empty_like(x)

	for i in range(x.shape[0]):
		out[i] = np.median(padded[i : i + w])
	return out

