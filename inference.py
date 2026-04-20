import os
import cv2
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm


# =========================
# CONFIG
# =========================

@dataclass
class InferConfig:
    # Путиgit
    csi_dir: str = "../.cache/kagglehub/datasets/shuokanghuang/wimans/versions/1/wifi_csi/amp"
    original_video_dir: str = "../.cache/kagglehub/datasets/shuokanghuang/wimans/versions/1/video"
    checkpoint_path: str = "./runs/csi2edges_mlflow/best.pt"
    output_dir: str = "../csi_inference_results"

    # Брать только каждое N-е видео
    take_every_n_video: int = 100

    # Дополнительно можно ограничить число видео
    max_videos: Optional[int] = None

    # Ограничить число кадров на видео
    max_frames_per_video: Optional[int] = None

    # Поддерживаемые CSI расширения
    csi_exts: Tuple[str, ...] = (".npy", ".npz", ".csv")

    # Порог бинаризации
    binary_threshold: float = 0.5

    # CPU fallback
    force_cpu: bool = False


CFG = InferConfig()


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
        x = F.interpolate(
            x,
            size=(self.out_h, self.out_w),
            mode="bilinear",
            align_corners=False
        )
        return x


class CSI2EdgeNet(nn.Module):
    def __init__(self, in_features: int, out_h: int, out_w: int):
        super().__init__()

        self.encoder = CSIEncoder(
            in_features=in_features,
            latent_dim=512
        )

        self.decoder = MaskDecoder(
            latent_dim=512,
            out_h=out_h,
            out_w=out_w
        )

    def forward(self, x):
        z = self.encoder(x)
        logits = self.decoder(z)
        return logits


# =========================
# UTILS
# =========================

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def stem_without_suffixes(video_path: Path) -> str:
    name = video_path.stem

    for suffix in ["_edges_thick", "_edges", "_binary"]:
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
            [
                amp.reshape(amp.shape[0], -1),
                phase.reshape(phase.shape[0], -1)
            ],
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


def find_pairs(
    csi_dir: Path,
    video_dir: Path,
    csi_exts: Tuple[str, ...]
) -> List[Tuple[Path, Path]]:
    csi_map: Dict[str, Path] = {}

    for ext in csi_exts:
        for p in csi_dir.glob(f"*{ext}"):
            csi_map[p.stem] = p

    pairs = []

    for video_path in video_dir.iterdir():
        if not video_path.is_file():
            continue

        if video_path.suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv"}:
            continue

        base = stem_without_suffixes(video_path)

        if base in csi_map:
            pairs.append((csi_map[base], video_path))

    return sorted(pairs, key=lambda x: x[0].stem)


def read_video_info(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    return fps, width, height, frame_count


def overlay_text(frame: np.ndarray, text: str, x: int, y: int):
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


# =========================
# MODEL LOADING
# =========================

def load_model_from_checkpoint(checkpoint_path: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device)

    if "config" not in ckpt:
        raise ValueError("В checkpoint нет config")

    cfg = ckpt["config"]

    image_h = cfg["image_h"]
    image_w = cfg["image_w"]
    csi_window = cfg["csi_window"]

    state_dict = ckpt["model_state_dict"]

    first_weight = state_dict["encoder.net.0.weight"]
    in_features = first_weight.shape[1]

    model = CSI2EdgeNet(
        in_features=in_features,
        out_h=image_h,
        out_w=image_w
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    return model, image_h, image_w, csi_window


# =========================
# INFERENCE
# =========================

@torch.no_grad()
def infer_one_video(
    model: nn.Module,
    csi_path: Path,
    video_path: Path,
    output_path: Path,
    image_h: int,
    image_w: int,
    csi_window: int,
    threshold: float,
    device: str,
    max_frames: Optional[int] = None,
):
    csi = load_csi_file(csi_path)  # [T, F]
    T, Fdim = csi.shape

    fps, orig_w, orig_h, total_frames = read_video_info(video_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")

    output_width = orig_w * 2
    output_height = orig_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (output_width, output_height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Не удалось создать output video: {output_path}")

    num_frames = total_frames
    if max_frames is not None:
        num_frames = min(num_frames, max_frames)

    half = csi_window // 2

    frame_idx = 0

    pbar = tqdm(
        total=num_frames,
        desc=video_path.name,
        leave=False
    )

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if max_frames is not None and frame_idx >= max_frames:
            break

        if num_frames == 1:
            csi_center = T // 2
        else:
            csi_center = int(round(
                frame_idx * (T - 1) / max(num_frames - 1, 1)
            ))

        left = csi_center - half
        right = left + csi_window

        if left < 0:
            left = 0
            right = csi_window

        if right > T:
            right = T
            left = T - csi_window

        if left < 0 or right > T or (right - left) != csi_window:
            frame_idx += 1
            pbar.update(1)
            continue

        x = csi[left:right]
        x = torch.from_numpy(x).float()
        x = x.transpose(0, 1).unsqueeze(0).to(device)

        logits = model(x)

        pred = torch.sigmoid(logits)[0, 0].cpu().numpy()
        pred = (pred > threshold).astype(np.uint8) * 255

        pred_big = cv2.resize(
            pred,
            (orig_w, orig_h),
            interpolation=cv2.INTER_NEAREST
        )

        pred_bgr = cv2.cvtColor(pred_big, cv2.COLOR_GRAY2BGR)

        left_frame = frame.copy()
        right_frame = pred_bgr.copy()

        overlay_text(left_frame, "original", 10, 25)
        overlay_text(right_frame, "predicted_edges", 10, 25)

        canvas = np.concatenate([left_frame, right_frame], axis=1)
        writer.write(canvas)

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    writer.release()


# =========================
# MAIN
# =========================

def main():
    cfg = CFG

    device = get_device(cfg.force_cpu)
    print("Using device:", device)

    ensure_dir(cfg.output_dir)

    model, image_h, image_w, csi_window = load_model_from_checkpoint(
        cfg.checkpoint_path,
        device
    )

    pairs = find_pairs(
        Path(cfg.csi_dir),
        Path(cfg.original_video_dir),
        cfg.csi_exts
    )

    if len(pairs) == 0:
        raise RuntimeError("Не найдено ни одной пары CSI + video")

    # брать каждое сотое видео
    pairs = pairs[::cfg.take_every_n_video]

    if cfg.max_videos is not None:
        pairs = pairs[:cfg.max_videos]

    print(f"Найдено пар для инференса: {len(pairs)}")

    meta = {
        "checkpoint_path": cfg.checkpoint_path,
        "device": device,
        "num_videos": len(pairs),
        "take_every_n_video": cfg.take_every_n_video,
        "image_h": image_h,
        "image_w": image_w,
        "csi_window": csi_window,
        "binary_threshold": cfg.binary_threshold,
    }

    with open(
        Path(cfg.output_dir) / "inference_meta.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    for csi_path, video_path in tqdm(pairs, desc="All videos"):
        output_name = f"{video_path.stem}_side_by_side.mp4"
        output_path = Path(cfg.output_dir) / output_name

        try:
            infer_one_video(
                model=model,
                csi_path=csi_path,
                video_path=video_path,
                output_path=output_path,
                image_h=image_h,
                image_w=image_w,
                csi_window=csi_window,
                threshold=cfg.binary_threshold,
                device=device,
                max_frames=cfg.max_frames_per_video,
            )

            print(f"[OK] {video_path.name} -> {output_path.name}")

        except Exception as e:
            print(f"[ERROR] {video_path.name}: {e}")

    print(f"Готово. Результаты сохранены в: {cfg.output_dir}")


if __name__ == "__main__":
    main()