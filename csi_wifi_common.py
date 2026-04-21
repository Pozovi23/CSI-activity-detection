"""Common data and training utilities for CSI WiFi experiments.

This file keeps the whole pipeline in one place so the entrypoint scripts stay
short and easy to read:
1. read fixed CSI recordings from `wifi_data_set_fixed/`
2. split recordings without leakage
3. remove null subcarriers using train only
4. apply signal preprocessing and cut windows
5. normalize by train statistics
6. train a minimal `SincConv -> mean(abs) -> linear` model
"""

from __future__ import annotations

import copy
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent

from tools.csi_parser import Parser
from tools.wifi_data_preprocess import PreprocessConfig, preprocess_amplitude

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
except ModuleNotFoundError:
    torch = None
    nn = None
    F = None
    DataLoader = None
    TensorDataset = None


DATASET_ROOT = PROJECT_ROOT / "wifi_data_set_fixed"
DEVICE_ORDER = ("dev1", "dev2", "dev3")
NO_MOTION_LABEL = "label_00"
DISTANCE_CLASSES = [0, 1, 2, 3]
DISTANCE_CLASS_NAMES = ["0m", "1m", "2m", "3m"]
LABEL_TO_DISTANCE_M = {
    "label_00": 0,
    "label_01": 1,
    "label_02": 2,
    "label_03": 3,
}

DEFAULT_RANDOM_STATE = 42
DEFAULT_WINDOW = 20
DEFAULT_STEP = 20
DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_SINC_FILTERS = 8
DEFAULT_SINC_KERNEL_SIZE = 7

DEFAULT_PREPROCESS_CONFIG = PreprocessConfig(
    pipeline="hampel_savgol",
    apply_detrend=False,
    apply_zscore=False,
    hampel_window=7,
    hampel_sigma=3.0,
    sg_window=11,
    sg_poly=3,
    median_window=7,
    detrend_window=51,
    scale_floor_percentile=10.0,
    z_clip=8.0,
)

DEVICE_RE = re.compile(r"__(dev\d+)_")
TEST_RE = re.compile(r"test_(\d+)$")


def require_torch() -> None:
    if torch is None or nn is None or F is None or DataLoader is None or TensorDataset is None:
        raise RuntimeError("PyTorch не найден. Установите зависимости из requirements.txt.")


def set_seed(seed: int = DEFAULT_RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def natural_test_key(path: Path) -> tuple[int, str]:
    match = TEST_RE.search(path.name)
    return (int(match.group(1)) if match else 10**9, path.name)


def device_name_from_path(path: Path) -> str:
    match = DEVICE_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot infer device name from {path.name}")
    return match.group(1)


def build_recording_index(dataset_root: Path = DATASET_ROOT) -> pd.DataFrame:
    """Build one table row per recording folder."""
    rows = []
    for person_dir in sorted(dataset_root.glob("id_person_*")):
        for label_dir in sorted(person_dir.glob("label_*")):
            for test_dir in sorted(label_dir.glob("test_*"), key=natural_test_key):
                files_by_device = {
                    device_name_from_path(path): path for path in sorted(test_dir.glob("*.data"))
                }
                row = {
                    "recording_id": str(test_dir.relative_to(dataset_root)),
                    "person": person_dir.name,
                    "source_label": label_dir.name,
                    "binary_label": int(label_dir.name != NO_MOTION_LABEL),
                    "distance_m": LABEL_TO_DISTANCE_M[label_dir.name],
                }
                for dev in DEVICE_ORDER:
                    row[f"{dev}_path"] = files_by_device[dev]
                rows.append(row)
    return pd.DataFrame(rows)


def parse_amplitude_file(path: Path) -> tuple[np.ndarray, pd.DataFrame]:
    """Read one fixed `.data` file and return packet amplitudes."""
    df = Parser(path).parse()
    amplitude = np.stack(df["amplitude"].to_numpy()).astype(np.float32)
    return amplitude, df


def infer_null_subcarriers(
    recordings: pd.DataFrame,
    sample_per_label: int = 4,
    zero_ratio_threshold: float = 0.99,
) -> list[int]:
    """Find subcarriers that are almost always zero on the train split."""
    sample = pd.concat(
        [group.head(sample_per_label) for _, group in recordings.groupby("source_label", sort=True)],
        ignore_index=True,
    )
    zero_counts = None
    total_packets = 0

    iterator = tqdm(sample.iterrows(), total=len(sample), desc="Infer null subcarriers", leave=False)
    for _, row in iterator:
        for dev in DEVICE_ORDER:
            amplitude, _ = parse_amplitude_file(row[f"{dev}_path"])
            zeros = np.isclose(amplitude, 0.0).sum(axis=0)
            zero_counts = zeros if zero_counts is None else zero_counts + zeros
            total_packets += amplitude.shape[0]

    zero_ratio = zero_counts / total_packets
    return np.flatnonzero(zero_ratio >= zero_ratio_threshold).astype(int).tolist()


def remove_null_subcarriers(amplitude: np.ndarray, null_subcarriers: list[int]) -> np.ndarray:
    return np.delete(np.asarray(amplitude, dtype=np.float32), null_subcarriers, axis=-1)


def load_recording(row: pd.Series, null_subcarriers: list[int]) -> dict[str, object]:
    """Load one recording folder as a tensor [T, D, S]."""
    amplitudes = []
    for dev in DEVICE_ORDER:
        amplitude, _ = parse_amplitude_file(row[f"{dev}_path"])
        amplitudes.append(remove_null_subcarriers(amplitude, null_subcarriers))

    min_len = min(x.shape[0] for x in amplitudes)
    amplitudes = [x[:min_len] for x in amplitudes]

    return {
        "amplitude": np.stack(amplitudes, axis=1).astype(np.float32),
        "recording_id": row["recording_id"],
        "person": row["person"],
        "source_label": row["source_label"],
        "binary_label": int(row["binary_label"]),
        "distance_m": int(row["distance_m"]),
    }


def round_robin_sample_by_column(
    df: pd.DataFrame,
    column: str,
    n: int,
    random_state: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    groups = [
        group.iloc[rng.permutation(len(group))].reset_index(drop=True)
        for _, group in df.groupby(column, sort=True)
    ]
    selected_rows = []
    cursors = [0] * len(groups)
    while len(selected_rows) < n:
        added = False
        for group_i, group in enumerate(groups):
            if cursors[group_i] < len(group) and len(selected_rows) < n:
                selected_rows.append(group.iloc[cursors[group_i]])
                cursors[group_i] += 1
                added = True
        if not added:
            break
    return pd.DataFrame(selected_rows).reset_index(drop=True)


def balance_binary_recordings(
    recordings: pd.DataFrame,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    """Balance train recordings for the binary task only."""
    target = int(recordings["binary_label"].value_counts().min())
    selected = []
    for binary_label, group in recordings.groupby("binary_label", sort=True):
        selected.append(
            round_robin_sample_by_column(
                group,
                column="source_label",
                n=target,
                random_state=random_state + int(binary_label),
            )
        )
    return pd.concat(selected, ignore_index=True).sample(frac=1.0, random_state=random_state)


def split_recordings(
    recordings: pd.DataFrame,
    group_col: str = "recording_id",
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by recording or person without leaking windows across splits."""
    if test_size < 0.0 or val_size <= 0.0 or test_size + val_size >= 1.0:
        raise ValueError("Require 0 <= test_size, 0 < val_size and test_size + val_size < 1")

    if test_size == 0.0:
        train_val = recordings.reset_index(drop=True)
        test = recordings.iloc[0:0].copy().reset_index(drop=True)
        inner_groups = train_val[group_col].astype(str).to_numpy()
        inner = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state + 1)
        train_idx, val_idx = next(
            inner.split(train_val, train_val["binary_label"], groups=inner_groups)
        )
        train = train_val.iloc[train_idx].reset_index(drop=True)
        val = train_val.iloc[val_idx].reset_index(drop=True)
        return train, val, test

    groups = recordings[group_col].astype(str).to_numpy()
    outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(outer.split(recordings, recordings["binary_label"], groups=groups))

    train_val = recordings.iloc[train_val_idx].reset_index(drop=True)
    test = recordings.iloc[test_idx].reset_index(drop=True)

    inner_groups = train_val[group_col].astype(str).to_numpy()
    val_fraction = val_size / (1.0 - test_size)
    inner = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=random_state + 1)
    train_idx, val_idx = next(
        inner.split(train_val, train_val["binary_label"], groups=inner_groups)
    )
    train = train_val.iloc[train_idx].reset_index(drop=True)
    val = train_val.iloc[val_idx].reset_index(drop=True)
    return train, val, test


def summarize_recording_overlap(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, int]:
    train_recordings = set(train_df["recording_id"])
    val_recordings = set(val_df["recording_id"])
    test_recordings = set(test_df["recording_id"])
    train_people = set(train_df["person"])
    val_people = set(val_df["person"])
    test_people = set(test_df["person"])
    return {
        "recording_train_val": len(train_recordings & val_recordings),
        "recording_train_test": len(train_recordings & test_recordings),
        "recording_val_test": len(val_recordings & test_recordings),
        "person_train_val": len(train_people & val_people),
        "person_train_test": len(train_people & test_people),
        "person_val_test": len(val_people & test_people),
    }


def segment_array(X: np.ndarray, window: int, step: int) -> list[np.ndarray]:
    return [X[start : start + window] for start in range(0, len(X) - window + 1, step)]


def build_binary_windows(
    recordings: pd.DataFrame,
    null_subcarriers: list[int],
    window: int,
    step: int,
    preprocess_config: PreprocessConfig = DEFAULT_PREPROCESS_CONFIG,
    group_col: str = "recording_id",
    desc: str = "Build binary windows",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Build binary windows and their metadata from recording-level rows."""
    X_list = []
    y_list = []
    groups = []
    meta_rows = []

    iterator = tqdm(recordings.iterrows(), total=len(recordings), desc=desc, leave=False)
    for _, row in iterator:
        rec = load_recording(row, null_subcarriers)
        X = preprocess_amplitude(rec["amplitude"], preprocess_config)["normalized"].astype(np.float32)
        for segment_idx, x_seg in enumerate(segment_array(X, window=window, step=step)):
            X_list.append(x_seg)
            y_list.append(rec["binary_label"])
            groups.append(rec[group_col])
            meta_rows.append(
                {
                    "recording_id": rec["recording_id"],
                    "person": rec["person"],
                    "source_label": rec["source_label"],
                    "binary_label": rec["binary_label"],
                    "distance_m": rec["distance_m"],
                    "segment": segment_idx,
                }
            )

    return (
        np.stack(X_list).astype(np.float32),
        np.asarray(y_list, dtype=np.int64),
        np.asarray(groups),
        pd.DataFrame(meta_rows),
    )


def build_distance_windows(
    recordings: pd.DataFrame,
    null_subcarriers: list[int],
    window: int,
    step: int,
    preprocess_config: PreprocessConfig = DEFAULT_PREPROCESS_CONFIG,
    group_col: str = "recording_id",
    desc: str = "Build distance windows",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Build distance windows and their metadata from recording-level rows."""
    X_list = []
    y_list = []
    groups = []
    meta_rows = []

    iterator = tqdm(recordings.iterrows(), total=len(recordings), desc=desc, leave=False)
    for _, row in iterator:
        rec = load_recording(row, null_subcarriers)
        X = preprocess_amplitude(rec["amplitude"], preprocess_config)["normalized"].astype(np.float32)
        for segment_idx, x_seg in enumerate(segment_array(X, window=window, step=step)):
            X_list.append(x_seg)
            y_list.append(rec["distance_m"])
            groups.append(rec[group_col])
            meta_rows.append(
                {
                    "recording_id": rec["recording_id"],
                    "person": rec["person"],
                    "source_label": rec["source_label"],
                    "binary_label": rec["binary_label"],
                    "distance_m": rec["distance_m"],
                    "segment": segment_idx,
                }
            )

    return (
        np.stack(X_list).astype(np.float32),
        np.asarray(y_list, dtype=np.int64),
        np.asarray(groups),
        pd.DataFrame(meta_rows),
    )


def reshape_for_model(X: np.ndarray) -> np.ndarray:
    """Convert [N, T, D, S] to [N, C, T] where C = D * S."""
    n, t, d, s = X.shape
    return X.transpose(0, 2, 3, 1).reshape(n, d * s, t).astype(np.float32)


def fit_normalizer(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute train-only channel mean and std."""
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_normalizer(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)


if nn is not None:

    class SincConv1d(nn.Module):
        def __init__(self, out_channels: int, kernel_size: int, sample_rate: float = 100.0) -> None:
            super().__init__()
            if kernel_size % 2 == 0:
                raise ValueError("sinc kernel size must be odd")

            self.out_channels = out_channels
            self.kernel_size = kernel_size
            self.sample_rate = sample_rate
            self.min_low_hz = 1.0
            self.min_band_hz = 2.0

            low_hz = 2.0
            high_hz = sample_rate / 2.0 - (self.min_low_hz + self.min_band_hz)
            mel = np.linspace(self.to_mel(low_hz), self.to_mel(high_hz), out_channels + 1)
            hz = self.to_hz(mel)

            self.low_hz_ = nn.Parameter(torch.tensor(hz[:-1] - self.min_low_hz).view(-1, 1).float())
            self.band_hz_ = nn.Parameter(torch.tensor(np.diff(hz) - self.min_band_hz).view(-1, 1).float())

            half = kernel_size // 2
            self.register_buffer(
                "t_right",
                torch.arange(1, half + 1, dtype=torch.float32).view(1, -1) / sample_rate,
            )
            self.register_buffer(
                "window",
                torch.hamming_window(half, periodic=False, dtype=torch.float32).view(1, -1),
            )

        @staticmethod
        def to_mel(hz):
            return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)

        @staticmethod
        def to_hz(mel):
            return 700.0 * (10 ** (np.asarray(mel) / 2595.0) - 1.0)

        def current_bands(self) -> tuple[np.ndarray, np.ndarray]:
            low = (self.min_low_hz + torch.abs(self.low_hz_)).detach().cpu().numpy().reshape(-1)
            high = low + self.min_band_hz + torch.abs(self.band_hz_).detach().cpu().numpy().reshape(-1)
            high = np.minimum(high, self.sample_rate / 2.0 - 1e-3)
            return low, high

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch_size, in_channels, seq_len = x.shape
            low = self.min_low_hz + torch.abs(self.low_hz_)
            high = low + self.min_band_hz + torch.abs(self.band_hz_)
            high = torch.clamp(high, max=self.sample_rate / 2.0 - 1e-3)
            band = (high - low).clamp_min(self.min_band_hz)

            low_pass = 2.0 * low / self.sample_rate * torch.sinc(2.0 * low * self.t_right)
            high_pass = 2.0 * high / self.sample_rate * torch.sinc(2.0 * high * self.t_right)
            band_pass_right = (high_pass - low_pass) * self.window
            band_pass_left = torch.flip(band_pass_right, dims=[1])
            center = (2.0 * band / self.sample_rate).view(-1, 1)
            filters = torch.cat([band_pass_left, center, band_pass_right], dim=1)
            filters = filters / filters.abs().amax(dim=1, keepdim=True).clamp_min(1e-6)

            y = F.conv1d(
                x.reshape(batch_size * in_channels, 1, seq_len),
                filters.unsqueeze(1),
                stride=1,
                padding=0,
            )
            return y.view(batch_size, in_channels * self.out_channels, y.shape[-1])


    class SincLinearBinary(nn.Module):
        def __init__(self, input_channels: int, sinc_filters: int, sinc_kernel_size: int) -> None:
            super().__init__()
            self.sinc = SincConv1d(sinc_filters, sinc_kernel_size)
            self.feature_dim = input_channels * sinc_filters
            self.head = nn.Linear(self.feature_dim, 1)

        def extract_features(self, x: torch.Tensor) -> torch.Tensor:
            return self.sinc(x).abs().mean(dim=-1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(self.extract_features(x)).squeeze(-1)


    class SincLinearDistance(nn.Module):
        def __init__(self, input_channels: int, sinc_filters: int, sinc_kernel_size: int) -> None:
            super().__init__()
            self.sinc = SincConv1d(sinc_filters, sinc_kernel_size)
            self.feature_dim = input_channels * sinc_filters
            self.head = nn.Linear(self.feature_dim, len(DISTANCE_CLASSES))

        def extract_features(self, x: torch.Tensor) -> torch.Tensor:
            return self.sinc(x).abs().mean(dim=-1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(self.extract_features(x))


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool):
    require_torch()
    y_tensor = torch.from_numpy(y)
    return DataLoader(
        TensorDataset(torch.from_numpy(X).float(), y_tensor),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def evaluate_binary(model, loader, device) -> tuple[float, str]:
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for xb, yb in loader:
            probs = torch.sigmoid(model(xb.to(device))).cpu().numpy()
            y_true.append(yb.numpy().astype(np.int64))
            y_pred.append((probs >= 0.5).astype(np.int64))

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    return (
        float(balanced_accuracy_score(y_true, y_pred)),
        classification_report(y_true, y_pred, target_names=["no_motion", "motion"], digits=4),
    )


def class_weight_tensor(y: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.bincount(y, minlength=len(DISTANCE_CLASSES)).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def evaluate_distance(model, loader, device) -> dict[str, object]:
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device))
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            y_true.append(yb.numpy().astype(np.int64))
            y_pred.append(pred)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "report": classification_report(
            y_true,
            y_pred,
            labels=DISTANCE_CLASSES,
            target_names=DISTANCE_CLASS_NAMES,
            digits=4,
            zero_division=0,
        ),
    }


def train_binary_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    sinc_filters: int,
    sinc_kernel_size: int,
) -> tuple[nn.Module, torch.device]:
    require_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SincLinearBinary(
        input_channels=X_train.shape[1],
        sinc_filters=sinc_filters,
        sinc_kernel_size=sinc_kernel_size,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    pos_weight = torch.tensor(
        [float(np.sum(y_train == 0) / max(np.sum(y_train == 1), 1))],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_loader = make_loader(X_train, y_train.astype(np.float32), batch_size=batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val.astype(np.float32), batch_size=batch_size, shuffle=False)

    best_state = None
    best_val_score = -1.0
    epoch_bar = tqdm(range(1, epochs + 1), desc="Train binary", leave=True)
    for epoch in epoch_bar:
        model.train()
        batch_bar = tqdm(train_loader, desc=f"Epoch {epoch:02d}", leave=False)
        for xb, yb in batch_bar:
            xb = xb.to(device)
            yb = yb.to(device).float()
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")

        val_score, _ = evaluate_binary(model, val_loader, device)
        epoch_bar.set_postfix(val_bal_acc=f"{val_score:.4f}")
        if val_score > best_val_score:
            best_val_score = val_score
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, device


def train_distance_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    sinc_filters: int,
    sinc_kernel_size: int,
) -> tuple[nn.Module, torch.device]:
    require_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SincLinearDistance(
        input_channels=X_train.shape[1],
        sinc_filters=sinc_filters,
        sinc_kernel_size=sinc_kernel_size,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor(y_train, device))

    train_loader = make_loader(X_train, y_train.astype(np.int64), batch_size=batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val.astype(np.int64), batch_size=batch_size, shuffle=False)

    best_state = None
    best_val_score = -1.0
    epoch_bar = tqdm(range(1, epochs + 1), desc="Train distance", leave=True)
    for epoch in epoch_bar:
        model.train()
        batch_bar = tqdm(train_loader, desc=f"Epoch {epoch:02d}", leave=False)
        for xb, yb in batch_bar:
            xb = xb.to(device)
            yb = yb.to(device).long()
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")

        val_metrics = evaluate_distance(model, val_loader, device)
        epoch_bar.set_postfix(val_bal_acc=f"{val_metrics['balanced_accuracy']:.4f}")
        if val_metrics["balanced_accuracy"] > best_val_score:
            best_val_score = val_metrics["balanced_accuracy"]
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, device


def filter_band_summary(model) -> pd.DataFrame:
    low, high = model.sinc.current_bands()
    summary = pd.DataFrame(
        {
            "filter_idx": np.arange(len(low), dtype=int),
            "low_hz": low,
            "high_hz": high,
        }
    )
    summary["center_hz"] = 0.5 * (summary["low_hz"] + summary["high_hz"])
    summary["bandwidth_hz"] = summary["high_hz"] - summary["low_hz"]
    return summary.round(4)


def filter_activity_summary_binary(model, X: np.ndarray, y: np.ndarray, device) -> pd.DataFrame:
    model.eval()
    with torch.no_grad():
        z = model.sinc(torch.from_numpy(X).float().to(device)).cpu()

    n_samples = z.shape[0]
    n_input_channels = X.shape[1]
    n_filters = model.sinc.out_channels
    z = z.view(n_samples, n_input_channels, n_filters, z.shape[-1])
    activity = z.abs().mean(dim=(1, 3)).numpy()

    summary = filter_band_summary(model)
    summary["act_no_motion"] = activity[y == 0].mean(axis=0)
    summary["act_motion"] = activity[y == 1].mean(axis=0)
    summary["motion_to_static_ratio"] = summary["act_motion"] / np.maximum(summary["act_no_motion"], 1e-6)
    return summary.round(4)


def filter_activity_summary_by_class(
    model,
    X: np.ndarray,
    y: np.ndarray,
    device,
    class_names: list[str],
) -> pd.DataFrame:
    model.eval()
    with torch.no_grad():
        z = model.sinc(torch.from_numpy(X).float().to(device)).cpu()

    n_samples = z.shape[0]
    n_input_channels = X.shape[1]
    n_filters = model.sinc.out_channels
    z = z.view(n_samples, n_input_channels, n_filters, z.shape[-1])
    activity = z.abs().mean(dim=(1, 3)).numpy()

    summary = filter_band_summary(model)
    for class_idx, class_name in enumerate(class_names):
        mask = y == class_idx
        summary[f"act_{class_name}"] = activity[mask].mean(axis=0) if np.any(mask) else np.nan
    return summary.round(4)
