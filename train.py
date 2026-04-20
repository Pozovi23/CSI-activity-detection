import json
import time
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional

import cv2
import mlflow
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =========================
# CONFIG
# =========================

@dataclass
class Config:
    # Paths
    csi_dir: str = "/home/beltis/.cache/kagglehub/datasets/shuokanghuang/wimans/versions/1/wifi_csi/amp"
    motion_edge_video_dir: str = "/home/beltis/.cache/kagglehub/datasets/shuokanghuang/wimans/versions/1/video_person_mask_binary_deeplabv3"
    original_video_dir: str = "/home/beltis/.cache/kagglehub/datasets/shuokanghuang/wimans/versions/1/video"
    output_dir: str = "./runs/csi2segmentation_mlflow"

    # MLflow
    mlflow_tracking_uri: str = "file:./mlruns"
    mlflow_experiment_name: str = "segmentation"
    mlflow_run_name: str = "segmentation"

    # Target size
    image_h: int = 128
    image_w: int = 128

    # CSI window per frame
    csi_window: int = 64

    # Limit frames per video while building datasets
    max_frames_per_video: Optional[int] = None

    # Split by videos
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10
    seed: int = 42

    # DataLoader
    batch_size: int = 8
    num_workers: int = 0

    # Train
    epochs: int = 160
    lr: float = 1e-3
    weight_decay: float = 1e-4
    binary_threshold: float = 0.5

    # Checkpoints
    save_every: int = 5

    # Preview after each epoch
    preview_num_videos: int = 10
    preview_take_every_n_test_video: int = 100
    preview_max_frames_per_video: int = 100
    preview_fps: int = 10

    # CSI formats
    csi_exts: Tuple[str, ...] = (".npy", ".npz", ".csv")

    # Device
    force_cpu: bool = False


CFG = Config()


# =========================
# DEVICE
# =========================

def get_device(force_cpu: bool = False) -> str:
    if force_cpu:
        return "cpu"

    if torch.cuda.is_available():
        try:
            _ = torch.cuda.get_device_name(0)
            _ = torch.zeros(1).to("cuda")
            return "cuda"
        except Exception as e:
            print(f"[WARN] CUDA недоступна, fallback на CPU: {e}")
            return "cpu"

    return "cpu"


# =========================
# UTILS
# =========================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        try:
            torch.cuda.manual_seed_all(seed)
        except Exception:
            pass


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def stem_without_suffixes(video_path: Path) -> str:
    name = video_path.stem
    for suffix in ["_motion_edges", "_edges_thick", "_edges", "_binary", "_person_mask"]:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    return name


def load_csi_file(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        arr = np.load(path)
    elif path.suffix == ".npz":
        data = np.load(path)
        if len(data.files) == 0:
            raise ValueError(f"Пустой npz: {path}")
        arr = data[data.files[0]]
    elif path.suffix == ".csv":
        arr = pd.read_csv(path).values
    else:
        raise ValueError(f"Unsupported CSI format: {path}")

    arr = np.asarray(arr)

    if np.iscomplexobj(arr):
        amp = np.abs(arr)
        phase = np.angle(arr)
        arr = np.concatenate(
            [amp.reshape(amp.shape[0], -1), phase.reshape(phase.shape[0], -1)],
            axis=1
        )
    else:
        if arr.ndim == 1:
            arr = arr[:, None]
        elif arr.ndim == 2:
            pass
        else:
            arr = arr.reshape(arr.shape[0], -1)

    arr = arr.astype(np.float32)

    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True) + 1e-6
    arr = (arr - mean) / std

    return arr


def preprocess_mask(frame_gray: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    frame = cv2.resize(frame_gray, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    frame = (frame > 127).astype(np.float32)
    return frame


def read_video_frames_gray(video_path: Path, max_frames: Optional[int] = None) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frames.append(frame)
        count += 1

        if max_frames is not None and count >= max_frames:
            break

    cap.release()
    return frames


def read_video_frames_bgr(video_path: Path, max_frames: Optional[int] = None) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame.copy())
        count += 1

        if max_frames is not None and count >= max_frames:
            break

    cap.release()
    return frames


def find_triplets(
    csi_dir: Path,
    motion_edge_video_dir: Path,
    original_video_dir: Path,
    csi_exts: Tuple[str, ...],
) -> List[Tuple[Path, Path, Path]]:
    csi_map: Dict[str, Path] = {}
    for ext in csi_exts:
        for p in csi_dir.glob(f"*{ext}"):
            csi_map[p.stem] = p

    orig_map: Dict[str, Path] = {}
    for p in original_video_dir.iterdir():
        if p.is_file() and p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
            orig_map[p.stem] = p

    triplets = []
    for motion_edge_path in motion_edge_video_dir.iterdir():
        if not motion_edge_path.is_file():
            continue
        if motion_edge_path.suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv"}:
            continue

        base = stem_without_suffixes(motion_edge_path)
        if base in csi_map and base in orig_map:
            triplets.append((csi_map[base], motion_edge_path, orig_map[base]))

    return sorted(triplets, key=lambda x: x[0].stem)


def split_triplets(
    triplets: List[Tuple[Path, Path, Path]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
):
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must be 1.0")

    rng = random.Random(seed)
    triplets = triplets[:]
    rng.shuffle(triplets)

    n = len(triplets)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio)) if n >= 3 else 1
    n_test = n - n_train - n_val

    if n_test <= 0:
        n_test = 1
        if n_train > 1:
            n_train -= 1
        else:
            n_val -= 1

    train_triplets = triplets[:n_train]
    val_triplets = triplets[n_train:n_train + n_val]
    test_triplets = triplets[n_train + n_val:]

    if len(val_triplets) == 0:
        val_triplets = train_triplets[-1:]
        train_triplets = train_triplets[:-1]

    if len(test_triplets) == 0:
        test_triplets = val_triplets[-1:]
        val_triplets = val_triplets[:-1]

    return train_triplets, val_triplets, test_triplets


def build_samples_for_triplet(
    csi_path: Path,
    motion_edge_video_path: Path,
    original_video_path: Path,
    cfg: Config,
) -> List[Dict]:
    csi = load_csi_file(csi_path)
    motion_edge_frames = read_video_frames_gray(
        motion_edge_video_path,
        max_frames=cfg.max_frames_per_video
    )

    num_frames = len(motion_edge_frames)
    if num_frames == 0:
        return []

    T = csi.shape[0]
    half = cfg.csi_window // 2
    samples = []

    for frame_idx in range(num_frames):
        if num_frames == 1:
            csi_center = T // 2
        else:
            csi_center = int(round(frame_idx * (T - 1) / (num_frames - 1)))

        left = csi_center - half
        right = left + cfg.csi_window

        if left < 0:
            left = 0
            right = cfg.csi_window
        if right > T:
            right = T
            left = T - cfg.csi_window

        if left < 0 or right > T or right <= left:
            continue
        if (right - left) != cfg.csi_window:
            continue

        mask = preprocess_mask(motion_edge_frames[frame_idx], cfg.image_h, cfg.image_w)

        samples.append({
            "pair_id": csi_path.stem,
            "frame_idx": frame_idx,
            "csi_path": str(csi_path),
            "motion_edge_video_path": str(motion_edge_video_path),
            "original_video_path": str(original_video_path),
            "left": left,
            "right": right,
            "mask": mask,
        })

    return samples


# =========================
# DATASET
# =========================

class CSIMotionEdgeDataset(Dataset):
    def __init__(self, samples: List[Dict]):
        self.samples = samples
        self._csi_local_cache: Dict[str, np.ndarray] = {}

    def __len__(self):
        return len(self.samples)

    def _get_csi(self, csi_path: str) -> np.ndarray:
        if csi_path not in self._csi_local_cache:
            self._csi_local_cache[csi_path] = load_csi_file(Path(csi_path))
        return self._csi_local_cache[csi_path]

    def __getitem__(self, idx):
        s = self.samples[idx]
        csi = self._get_csi(s["csi_path"])
        x = csi[s["left"]:s["right"]]   # [W, F]
        y = s["mask"]                   # [H, W]

        x = torch.from_numpy(x).float().transpose(0, 1)   # [F, W]
        y = torch.from_numpy(y).float().unsqueeze(0)      # [1, H, W]

        return {
            "x": x,
            "y": y,
            "pair_id": s["pair_id"],
            "frame_idx": s["frame_idx"],
        }


# =========================
# MODEL
# =========================

class CSIEncoder(nn.Module):
    def __init__(self, in_features: int, latent_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_features, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128, 256, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Conv1d(256, 256, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Conv1d(256, 512, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(512, latent_dim)

    def forward(self, x):
        z = self.net(x).squeeze(-1)
        z = self.fc(z)
        return z


class MaskDecoder(nn.Module):
    def __init__(self, latent_dim: int = 512, out_h: int = 128, out_w: int = 128):
        super().__init__()
        self.out_h = out_h
        self.out_w = out_w

        self.fc = nn.Linear(latent_dim, 256 * 8 * 8)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(z.size(0), 256, 8, 8)
        x = self.dec(x)
        x = F.interpolate(x, size=(self.out_h, self.out_w), mode="bilinear", align_corners=False)
        return x


class CSI2MotionEdgeNet(nn.Module):
    def __init__(self, in_features: int, out_h: int, out_w: int):
        super().__init__()
        self.encoder = CSIEncoder(in_features=in_features, latent_dim=512)
        self.decoder = MaskDecoder(latent_dim=512, out_h=out_h, out_w=out_w)

    def forward(self, x):
        z = self.encoder(x)
        logits = self.decoder(z)
        return logits


# =========================
# LOSSES / METRICS
# =========================

def dice_loss_from_logits(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    num = 2 * (probs * targets).sum(dim=(1, 2, 3))
    den = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + eps
    return (1 - (num / den)).mean()


def bce_dice_loss(logits, targets):
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = dice_loss_from_logits(logits, targets)
    return bce + dice


@torch.no_grad()
def compute_iou_from_logits(logits, targets, threshold=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    inter = (preds * targets).sum(dim=(1, 2, 3))
    union = ((preds + targets) > 0).float().sum(dim=(1, 2, 3))
    iou = (inter + eps) / (union + eps)
    return iou.mean().item()


# =========================
# PREVIEW
# =========================

def make_triplet_frame(pred_mask, gt_mask, original_frame, size=(128, 128)):
    gt = (gt_mask * 255).astype(np.uint8)
    pred = (pred_mask * 255).astype(np.uint8)

    gt = cv2.resize(gt, size, interpolation=cv2.INTER_NEAREST)
    pred = cv2.resize(pred, size, interpolation=cv2.INTER_NEAREST)

    gt_bgr = cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR)
    pred_bgr = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)

    original = cv2.resize(original_frame, (size[0] * 2, size[1]), interpolation=cv2.INTER_LINEAR)

    cv2.putText(gt_bgr, "target_motion_edges", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(pred_bgr, "predicted_motion_edges", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(original, "original_video", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    top_row = np.concatenate([gt_bgr, pred_bgr], axis=1)
    canvas = np.concatenate([top_row, original], axis=0)
    return canvas


def select_preview_triplets(
    test_triplets: List[Tuple[Path, Path, Path]],
    take_every_n: int,
    max_videos: int,
) -> List[Tuple[Path, Path, Path]]:
    selected = test_triplets[::take_every_n] if take_every_n > 1 else test_triplets[:]
    if len(selected) == 0 and len(test_triplets) > 0:
        selected = test_triplets[:1]
    return selected[:max_videos]


@torch.no_grad()
def save_preview_videos_for_epoch(
    model: nn.Module,
    preview_triplets: List[Tuple[Path, Path, Path]],
    cfg: Config,
    epoch: int,
    device: str,
) -> List[str]:
    saved_files = []
    if len(preview_triplets) == 0:
        return saved_files

    epoch_dir = Path(cfg.output_dir) / "preview_videos" / f"epoch_{epoch:03d}"
    ensure_dir(str(epoch_dir))

    model.eval()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    for video_idx, (csi_path, motion_edge_video_path, original_video_path) in enumerate(preview_triplets):
        try:
            csi = load_csi_file(csi_path)
            gt_frames = read_video_frames_gray(
                motion_edge_video_path,
                max_frames=cfg.preview_max_frames_per_video
            )
            orig_frames = read_video_frames_bgr(
                original_video_path,
                max_frames=cfg.preview_max_frames_per_video
            )

            num_frames = min(len(gt_frames), len(orig_frames))
            if num_frames == 0:
                continue

            gt_frames = gt_frames[:num_frames]
            orig_frames = orig_frames[:num_frames]

            T = csi.shape[0]
            half = cfg.csi_window // 2

            vis_w = cfg.image_w
            vis_h = cfg.image_h
            out_size = (vis_w * 2, vis_h * 2)

            out_path = epoch_dir / f"{video_idx:02d}_{csi_path.stem}_motion_triplet.mp4"
            writer = cv2.VideoWriter(str(out_path), fourcc, cfg.preview_fps, out_size)

            if not writer.isOpened():
                print(f"[WARN] Не удалось создать видео: {out_path}")
                continue

            for frame_idx in range(num_frames):
                if num_frames == 1:
                    csi_center = T // 2
                else:
                    csi_center = int(round(frame_idx * (T - 1) / (num_frames - 1)))

                left = csi_center - half
                right = left + cfg.csi_window

                if left < 0:
                    left = 0
                    right = cfg.csi_window
                if right > T:
                    right = T
                    left = T - cfg.csi_window

                if left < 0 or right > T or (right - left) != cfg.csi_window:
                    continue

                x = csi[left:right]
                x = torch.from_numpy(x).float().transpose(0, 1).unsqueeze(0).to(device)

                logits = model(x)
                pred = torch.sigmoid(logits)[0, 0].cpu().numpy()
                pred = (pred > cfg.binary_threshold).astype(np.uint8)

                gt = preprocess_mask(gt_frames[frame_idx], cfg.image_h, cfg.image_w)

                frame = make_triplet_frame(
                    pred_mask=pred,
                    gt_mask=gt,
                    original_frame=orig_frames[frame_idx],
                    size=(vis_w, vis_h),
                )
                writer.write(frame)

            writer.release()
            saved_files.append(str(out_path))

        except Exception as e:
            print(f"[WARN] Preview error for {csi_path.stem}: {e}")

    return saved_files


# =========================
# DATA BUILDING
# =========================

def build_datasets(cfg: Config):
    csi_dir = Path(cfg.csi_dir)
    motion_edge_dir = Path(cfg.motion_edge_video_dir)
    original_dir = Path(cfg.original_video_dir)

    triplets = find_triplets(csi_dir, motion_edge_dir, original_dir, cfg.csi_exts)
    if len(triplets) == 0:
        raise RuntimeError("Не найдено ни одной тройки CSI + motion edge video + original video")

    train_triplets, val_triplets, test_triplets = split_triplets(
        triplets,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        seed=cfg.seed,
    )

    train_samples = []
    val_samples = []
    test_samples = []

    print("Building train samples...")
    for csi_path, motion_edge_video_path, original_video_path in tqdm(train_triplets, desc="train triplets"):
        train_samples.extend(build_samples_for_triplet(csi_path, motion_edge_video_path, original_video_path, cfg))

    print("Building val samples...")
    for csi_path, motion_edge_video_path, original_video_path in tqdm(val_triplets, desc="val triplets"):
        val_samples.extend(build_samples_for_triplet(csi_path, motion_edge_video_path, original_video_path, cfg))

    print("Building test samples...")
    for csi_path, motion_edge_video_path, original_video_path in tqdm(test_triplets, desc="test triplets"):
        test_samples.extend(build_samples_for_triplet(csi_path, motion_edge_video_path, original_video_path, cfg))

    random.Random(cfg.seed).shuffle(train_samples)
    random.Random(cfg.seed + 1).shuffle(val_samples)
    random.Random(cfg.seed + 2).shuffle(test_samples)

    train_ds = CSIMotionEdgeDataset(train_samples)
    val_ds = CSIMotionEdgeDataset(val_samples)
    test_ds = CSIMotionEdgeDataset(test_samples)

    return train_ds, val_ds, test_ds, train_triplets, val_triplets, test_triplets


# =========================
# TRAIN / EVAL
# =========================

def train_one_epoch(model, loader, optimizer, device, threshold):
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    total_n = 0

    pbar = tqdm(loader, desc="train", leave=False)
    for batch in pbar:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = bce_dice_loss(logits, y)
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        batch_iou = compute_iou_from_logits(logits.detach(), y, threshold=threshold)

        total_loss += loss.item() * bs
        total_iou += batch_iou * bs
        total_n += bs

        pbar.set_postfix(loss=f"{loss.item():.4f}", iou=f"{batch_iou:.4f}")

    return total_loss / max(total_n, 1), total_iou / max(total_n, 1)


@torch.no_grad()
def evaluate_one_epoch(model, loader, device, threshold, desc="eval"):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_n = 0

    pbar = tqdm(loader, desc=desc, leave=False)
    for batch in pbar:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        logits = model(x)
        loss = bce_dice_loss(logits, y)
        batch_iou = compute_iou_from_logits(logits, y, threshold=threshold)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_iou += batch_iou * bs
        total_n += bs

        pbar.set_postfix(loss=f"{loss.item():.4f}", iou=f"{batch_iou:.4f}")

    return total_loss / max(total_n, 1), total_iou / max(total_n, 1)


def save_checkpoint(path, model, optimizer, epoch, best_val_iou, cfg: Config):
    obj = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_iou": best_val_iou,
        "config": asdict(cfg),
    }
    torch.save(obj, path)


def log_dataset_info(train_triplets, val_triplets, test_triplets, train_ds, val_ds, test_ds, in_features, device):
    mlflow.log_params({
        "device": device,
        "train_triplets": len(train_triplets),
        "val_triplets": len(val_triplets),
        "test_triplets": len(test_triplets),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
        "csi_feature_dim": in_features,
    })


# =========================
# MAIN
# =========================

def main():
    cfg = CFG
    set_seed(cfg.seed)

    device = get_device(cfg.force_cpu)
    print("Using device:", device)

    ensure_dir(cfg.output_dir)
    ensure_dir(str(Path(cfg.output_dir) / "checkpoints"))
    ensure_dir(str(Path(cfg.output_dir) / "preview_videos"))

    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment_name)

    with mlflow.start_run(run_name=cfg.mlflow_run_name):
        mlflow.log_params(asdict(cfg))
        mlflow.log_param("resolved_device", device)

        train_ds, val_ds, test_ds, train_triplets, val_triplets, test_triplets = build_datasets(cfg)

        if len(train_ds) == 0:
            raise RuntimeError("Train dataset пуст")
        if len(val_ds) == 0:
            raise RuntimeError("Val dataset пуст")
        if len(test_ds) == 0:
            raise RuntimeError("Test dataset пуст")

        sample_csi = load_csi_file(Path(train_ds.samples[0]["csi_path"]))
        in_features = sample_csi.shape[1]

        print(f"Train triplets: {len(train_triplets)}")
        print(f"Val triplets:   {len(val_triplets)}")
        print(f"Test triplets:  {len(test_triplets)}")
        print(f"Train samples:  {len(train_ds)}")
        print(f"Val samples:    {len(val_ds)}")
        print(f"Test samples:   {len(test_ds)}")
        print(f"CSI feature dim: {in_features}")

        log_dataset_info(train_triplets, val_triplets, test_triplets, train_ds, val_ds, test_ds, in_features, device)

        preview_triplets = select_preview_triplets(
            test_triplets=test_triplets,
            take_every_n=cfg.preview_take_every_n_test_video,
            max_videos=cfg.preview_num_videos,
        )
        print(f"Preview videos per epoch: {len(preview_triplets)}")

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=(device == "cuda"),
            drop_last=False,
            persistent_workers=False,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=(device == "cuda"),
            drop_last=False,
            persistent_workers=False,
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=(device == "cuda"),
            drop_last=False,
            persistent_workers=False,
        )

        model = CSI2MotionEdgeNet(
            in_features=in_features,
            out_h=cfg.image_h,
            out_w=cfg.image_w,
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        best_val_iou = -1.0
        history = []

        epoch_bar = tqdm(range(1, cfg.epochs + 1), desc="epochs")
        for epoch in epoch_bar:
            epoch_start = time.perf_counter()

            train_loss, train_iou = train_one_epoch(
                model, train_loader, optimizer, device, cfg.binary_threshold
            )

            val_loss, val_iou = evaluate_one_epoch(
                model, val_loader, device, cfg.binary_threshold, desc="val"
            )

            test_loss, test_iou = evaluate_one_epoch(
                model, test_loader, device, cfg.binary_threshold, desc="test"
            )

            epoch_time_sec = time.perf_counter() - epoch_start

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_iou": train_iou,
                "val_loss": val_loss,
                "val_iou": val_iou,
                "test_loss": test_loss,
                "test_iou": test_iou,
                "epoch_time_sec": epoch_time_sec,
            }
            history.append(row)

            epoch_bar.set_postfix(
                train_loss=f"{train_loss:.4f}",
                val_loss=f"{val_loss:.4f}",
                val_iou=f"{val_iou:.4f}",
                t=f"{epoch_time_sec:.1f}s",
            )

            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_loss:.4f} train_iou={train_iou:.4f} | "
                f"val_loss={val_loss:.4f} val_iou={val_iou:.4f} | "
                f"test_loss={test_loss:.4f} test_iou={test_iou:.4f} | "
                f"time={epoch_time_sec:.2f}s"
            )

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_iou": train_iou,
                "val_loss": val_loss,
                "val_iou": val_iou,
                "test_loss": test_loss,
                "test_iou": test_iou,
                "epoch_time_sec": epoch_time_sec,
            }, step=epoch)

            last_ckpt = str(Path(cfg.output_dir) / "last.pt")
            save_checkpoint(last_ckpt, model, optimizer, epoch, best_val_iou, cfg)
            mlflow.log_artifact(last_ckpt, artifact_path="checkpoints_last")

            if val_iou > best_val_iou:
                best_val_iou = val_iou
                best_ckpt = str(Path(cfg.output_dir) / "best.pt")
                save_checkpoint(best_ckpt, model, optimizer, epoch, best_val_iou, cfg)
                mlflow.log_artifact(best_ckpt, artifact_path="checkpoints_best")

            if epoch % cfg.save_every == 0:
                ckpt_path = str(Path(cfg.output_dir) / "checkpoints" / f"epoch_{epoch:03d}.pt")
                save_checkpoint(ckpt_path, model, optimizer, epoch, best_val_iou, cfg)
                mlflow.log_artifact(ckpt_path, artifact_path="checkpoints_periodic")

            history_path = Path(cfg.output_dir) / "history.json"
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
            mlflow.log_artifact(str(history_path), artifact_path="logs")

            saved_preview_files = save_preview_videos_for_epoch(
                model=model,
                preview_triplets=preview_triplets,
                cfg=cfg,
                epoch=epoch,
                device=device,
            )

            for fp in saved_preview_files:
                mlflow.log_artifact(fp, artifact_path=f"preview_videos/epoch_{epoch:03d}")

        mlflow.log_artifact(str(Path(cfg.output_dir) / "last.pt"), artifact_path="final")
        mlflow.log_artifact(str(Path(cfg.output_dir) / "history.json"), artifact_path="final")

        print(f"Training finished. Best val IoU = {best_val_iou:.4f}")


if __name__ == "__main__":
    main()   