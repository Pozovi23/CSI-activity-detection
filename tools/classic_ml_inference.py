#!/usr/bin/env python3
"""CLI inference for classic CSI ML pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

# Make repository root importable when script is called from any directory.
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.csi_parser import Parser
from tools.filters import median_filter


def skewness(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean()
    sigma = x.std()
    if sigma == 0:
        return 0.0
    z = (x - mu) / sigma
    return float(np.mean(z ** 3))


def kurtosis_excess(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean()
    sigma = x.std()
    if sigma == 0:
        return 0.0
    z = (x - mu) / sigma
    return float(np.mean(z ** 4) - 3.0)


def spectral_features(signal: np.ndarray) -> tuple[float, float, float, float, float]:
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
    eps = 1e-12
    entropy = float(-(p_norm * np.log2(p_norm + eps)).sum())

    dominant_idx = int(np.argmax(power))
    dominant_freq = float(freqs[dominant_idx])
    rolloff_threshold = 0.85 * power_sum
    rolloff = float(freqs[np.searchsorted(np.cumsum(power), rolloff_threshold)])

    return centroid, spread, entropy, dominant_freq, rolloff


def build_unit_signal(df: pd.DataFrame, window: int) -> np.ndarray:
    amp_matrix = np.stack(df["amplitude"].to_numpy(), axis=0).astype(np.float64)
    raw_time_signal = amp_matrix.mean(axis=1)
    return median_filter(raw_time_signal, window)


def build_feature_row(df: pd.DataFrame, signal: np.ndarray, global_min: float, global_max: float) -> dict[str, float]:
    den = global_max - global_min
    if den <= 0:
        raise ValueError("Invalid scaling parameters: global_max must be > global_min")

    scaled_signal = (signal - global_min) / den
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

    sk = skewness(scaled_signal)
    kt = kurtosis_excess(scaled_signal)
    sc, ss, se, dom_freq, rolloff = spectral_features(scaled_signal)

    rssi_dbm = float(df["rssi_dbm"].median())

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
        "rssi_dbm": rssi_dbm,
    }


def run_inference(model_path: Path, preprocessing_path: Path, data_file: Path) -> dict[str, object]:
    preproc_bundle = joblib.load(preprocessing_path)

    median_width = int(preproc_bundle["median_width"])
    global_min = float(preproc_bundle["global_min"])
    global_max = float(preproc_bundle["global_max"])
    feature_cols = list(preproc_bundle["feature_cols"])
    scaler_pca = preproc_bundle["scaler_pca"]
    pca = preproc_bundle["pca"]
    k_95 = int(preproc_bundle["k_95"])

    model = CatBoostClassifier()
    model.load_model(str(model_path))

    df = Parser(data_file).parse()
    signal = build_unit_signal(df, median_width)
    feature_row = build_feature_row(df, signal, global_min, global_max)

    feature_df = pd.DataFrame([feature_row])
    X = feature_df[feature_cols].to_numpy(dtype=np.float64)
    X_std = scaler_pca.transform(X)
    X_pca_95 = pca.transform(X_std)[:, :k_95]

    pred = int(model.predict(X_pca_95)[0])
    proba = model.predict_proba(X_pca_95)[0].tolist()

    return {
        "data_file": str(data_file),
        "predicted_class": pred,
        "predicted_label_name": "label_00" if pred == 0 else "other_labels",
        "probability_class_0": float(proba[0]),
        "probability_class_1": float(proba[1]),
        "k_95": k_95,
        "n_packets": int(len(df)),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Infer one CSI .data sample with saved CatBoost model and preprocessing artifacts"
    )
    parser.add_argument("--model-path", type=Path, required=True, help="Path to saved CatBoost .cbm model")
    parser.add_argument(
        "--preprocessing-path",
        type=Path,
        required=True,
        help="Path to preprocessing.joblib from training notebook",
    )
    parser.add_argument("--data-file", type=Path, required=True, help="Path to one .data file to classify")
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if not args.model_path.exists():
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    if not args.preprocessing_path.exists():
        raise FileNotFoundError(f"Preprocessing file not found: {args.preprocessing_path}")
    if not args.data_file.exists():
        raise FileNotFoundError(f"Data file not found: {args.data_file}")

    result = run_inference(args.model_path, args.preprocessing_path, args.data_file)
    if args.pretty:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
