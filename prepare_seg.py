from pathlib import Path

import cv2
import numpy as np
from tqdm.auto import tqdm

import torch
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights,
)

# ===== Пути =====
INPUT_DIR = Path("/home/beltis/.cache/kagglehub/datasets/shuokanghuang/wimans/versions/1/video")
OUTPUT_DIR = Path("/home/beltis/.cache/kagglehub/datasets/shuokanghuang/wimans/versions/1/video_person_mask_binary_deeplabv3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV"}

# ===== Настройки =====
PERSON_CLASS = 15          # person для VOC-стиля segmentation head
FORCE_CPU = False

BATCH_SIZE = 8
MAX_INFER_SIZE = 640
USE_PIN_MEMORY = True

APPLY_MORPH = True
MORPH_KERNEL_SIZE = 3
UPSCALE_LINEAR_THEN_THRESHOLD = True


def get_device(force_cpu: bool = False) -> str:
    if force_cpu:
        return "cpu"

    if torch.cuda.is_available():
        try:
            _ = torch.zeros(1, device="cuda")
            return "cuda"
        except Exception as e:
            print(f"[WARN] CUDA недоступна, fallback на CPU: {e}")
            return "cpu"

    return "cpu"


DEVICE = get_device(FORCE_CPU)


def load_model(device: str):
    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    model = deeplabv3_resnet50(weights=weights)
    model.eval()
    model.to(device)
    return model


def video_writer(output_path: Path, fps: float, width: int, height: int):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
        True,
    )
    return writer


def resize_keep_aspect(frame: np.ndarray, max_size: int | None):
    h, w = frame.shape[:2]

    if max_size is None:
        return frame, h, w

    long_side = max(h, w)
    if long_side <= max_size:
        return frame, h, w

    scale = max_size / long_side
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized, h, w


def frame_to_tensor(frame_bgr: np.ndarray) -> torch.Tensor:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = (
        torch.from_numpy(frame_rgb)
        .permute(2, 0, 1)
        .float()
        .div_(255.0)
        .contiguous()
    )

    if DEVICE == "cuda" and USE_PIN_MEMORY:
        tensor = tensor.pin_memory()

    return tensor


def postprocess_mask(mask: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    if mask.shape != (out_h, out_w):
        if UPSCALE_LINEAR_THEN_THRESHOLD:
            resized = cv2.resize(mask, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            mask = (resized >= 127).astype(np.uint8) * 255
        else:
            mask = cv2.resize(mask, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

    if APPLY_MORPH:
        kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def segmentation_batch_to_masks(outputs: torch.Tensor, original_sizes):
    # outputs: [B, C, H, W]
    preds = outputs.argmax(1).byte().cpu().numpy()  # [B, H, W]

    result_masks = []
    for pred, (orig_h, orig_w) in zip(preds, original_sizes):
        person_mask = (pred == PERSON_CLASS).astype(np.uint8) * 255
        person_mask = postprocess_mask(person_mask, orig_h, orig_w)
        result_masks.append(person_mask)

    return result_masks


def run_model(model, tensors, device: str):
    with torch.no_grad():
        batch = torch.stack(tensors, dim=0)

        if device == "cuda":
            batch = batch.to(device, non_blocking=True)
        else:
            batch = batch.to(device)

        outputs = model(batch)["out"]  # [B, C, H, W]

    return outputs


def process_video(video_path: Path, output_path: Path, model, device: str):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return f"[ERROR] Не удалось открыть видео: {video_path}"

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = None

    writer = video_writer(output_path, fps, width, height)
    if not writer.isOpened():
        cap.release()
        return f"[ERROR] Не удалось создать выходное видео: {output_path}"

    frame_count = 0
    batch_tensors = []
    batch_original_sizes = []

    pbar = tqdm(total=total_frames, desc=video_path.name, leave=False)

    def flush_batch():
        nonlocal frame_count, batch_tensors, batch_original_sizes

        if not batch_tensors:
            return

        outputs = run_model(model, batch_tensors, device)
        masks = segmentation_batch_to_masks(outputs, batch_original_sizes)

        for person_mask in masks:
            out_frame = cv2.cvtColor(person_mask, cv2.COLOR_GRAY2BGR)
            writer.write(out_frame)
            frame_count += 1

        pbar.update(len(batch_tensors))
        batch_tensors.clear()
        batch_original_sizes.clear()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        infer_frame, orig_h, orig_w = resize_keep_aspect(frame, MAX_INFER_SIZE)
        tensor = frame_to_tensor(infer_frame)

        batch_tensors.append(tensor)
        batch_original_sizes.append((orig_h, orig_w))

        if len(batch_tensors) >= BATCH_SIZE:
            flush_batch()

    flush_batch()

    pbar.close()
    cap.release()
    writer.release()

    if device == "cuda":
        torch.cuda.empty_cache()

    return f"[OK] {video_path.name} -> {output_path.name}, frames={frame_count}, device={device}"


def main():
    if not INPUT_DIR.exists():
        print(f"[ERROR] Папка не найдена: {INPUT_DIR}")
        return

    video_files = sorted(
        [
            p
            for p in INPUT_DIR.iterdir()
            if p.is_file() and p.suffix in VIDEO_EXTS
        ]
    )

    if not video_files:
        print(f"[WARN] В папке нет видео: {INPUT_DIR}")
        return

    print(f"Найдено видео: {len(video_files)}")
    print(f"DEVICE = {DEVICE}")
    print(f"BATCH_SIZE = {BATCH_SIZE}")
    print(f"MAX_INFER_SIZE = {MAX_INFER_SIZE}")
    print(f"PERSON_CLASS = {PERSON_CLASS}")

    try:
        model = load_model(DEVICE)
    except Exception as e:
        print(f"[ERROR] Не удалось загрузить модель: {e}")
        return

    results = []
    for video_path in tqdm(video_files, desc="Processing videos"):
        output_path = OUTPUT_DIR / f"{video_path.stem}_person_mask.mp4"
        try:
            result = process_video(video_path, output_path, model, DEVICE)
        except Exception as e:
            result = f"[ERROR] {video_path.name}: {type(e).__name__}: {e}"

        print(result)
        results.append(result)

    ok_count = sum(r.startswith("[OK]") for r in results)
    err_count = len(results) - ok_count
    print(f"\nГотово. Успешно: {ok_count}, ошибок: {err_count}")


if __name__ == "__main__":
    main()