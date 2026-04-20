import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
from tqdm.auto import tqdm

# ===== Пути =====
INPUT_DIR = Path("/home/beltis/.cache/kagglehub/datasets/shuokanghuang/wimans/versions/1/video")
OUTPUT_DIR = Path("/home/beltis/.cache/kagglehub/datasets/shuokanghuang/wimans/versions/1/video_motion_edges_binary")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Поддерживаемые расширения
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV"}

# ===== Параметры обработки =====
GAUSSIAN_BLUR = (5, 5)

# Canny
CANNY_LOW = 30
CANNY_HIGH = 100

# Motion detection
MOTION_THRESH = 1
MOTION_KERNEL_SIZE = 3
MOTION_OPEN_ITER = 0
MOTION_CLOSE_ITER = 1
MIN_MOTION_AREA = 10

# Утолщение итоговых motion edges
DILATION_KERNEL_SIZE = 4
DILATION_ITERATIONS = 3

# Параллельность
NUM_WORKERS = max(1, min(8, (os.cpu_count() or 4) // 2))


def remove_small_components(binary_mask: np.ndarray, min_area: int) -> np.ndarray:
    # num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    # cleaned = np.zeros_like(binary_mask)

    # for label_id in range(1, num_labels):
    #     area = stats[label_id, cv2.CC_STAT_AREA]
    #     if area >= min_area:
    #         cleaned[labels == label_id] = 255

    # return cleaned
    return binary_mask


def build_motion_mask(prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
    diff = cv2.absdiff(curr_gray, prev_gray)
    _, motion = cv2.threshold(diff, MOTION_THRESH, 255, cv2.THRESH_BINARY)

    kernel = np.ones((MOTION_KERNEL_SIZE, MOTION_KERNEL_SIZE), np.uint8)

    motion = cv2.morphologyEx(
        motion,
        cv2.MORPH_OPEN,
        kernel,
        iterations=MOTION_OPEN_ITER
    )

    motion = cv2.morphologyEx(
        motion,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=MOTION_CLOSE_ITER
    )

    motion = remove_small_components(motion, MIN_MOTION_AREA)
    return motion


def process_video(video_path_str: str, output_path_str: str):
    cv2.setNumThreads(0)

    video_path = Path(video_path_str)
    output_path = Path(output_path_str)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return f"[ERROR] Не удалось открыть: {video_path}"

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
        isColor=False
    )

    if not writer.isOpened():
        cap.release()
        return f"[ERROR] Не удалось создать выходное видео: {output_path}"

    dilate_kernel = np.ones((DILATION_KERNEL_SIZE, DILATION_KERNEL_SIZE), np.uint8)

    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        writer.release()
        return f"[ERROR] Пустое видео: {video_path}"

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, GAUSSIAN_BLUR, 0)

    # Первый кадр — пустой, потому что ещё не с чем сравнивать
    first_out = np.zeros((height, width), dtype=np.uint8)
    writer.write(first_out)

    frame_count = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.GaussianBlur(curr_gray, GAUSSIAN_BLUR, 0)

        # Маска движения
        motion_mask = build_motion_mask(prev_gray, curr_gray)

        # Границы текущего кадра
        edges = cv2.Canny(curr_gray, CANNY_LOW, CANNY_HIGH)

        # Только границы в движущихся областях
        motion_edges = cv2.bitwise_and(edges, motion_mask)

        # Утолщаем
        motion_edges = cv2.dilate(
            motion_edges,
            dilate_kernel,
            iterations=DILATION_ITERATIONS
        )

        # Закрываем небольшие разрывы
        motion_edges = cv2.morphologyEx(
            motion_edges,
            cv2.MORPH_CLOSE,
            dilate_kernel
        )

        writer.write(motion_edges)

        prev_gray = curr_gray
        frame_count += 1

    cap.release()
    writer.release()

    return f"[OK] {video_path.name} -> {output_path.name}, кадров: {frame_count}"


def main():
    if not INPUT_DIR.exists():
        print(f"[ERROR] Папка не найдена: {INPUT_DIR}")
        return

    video_files = [
        p for p in INPUT_DIR.iterdir()
        if p.is_file() and p.suffix in VIDEO_EXTS
    ]

    if not video_files:
        print(f"[WARN] В папке нет видео: {INPUT_DIR}")
        return

    print(f"Найдено видео: {len(video_files)}")
    print(f"Используем процессов: {NUM_WORKERS}")

    tasks = []
    for video_path in sorted(video_files):
        output_name = video_path.stem + "_motion_edges.mp4"
        output_path = OUTPUT_DIR / output_name
        tasks.append((str(video_path), str(output_path)))

    results = []
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [
            executor.submit(process_video, video_path, output_path)
            for video_path, output_path in tasks
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing videos"):
            results.append(future.result())

    ok_count = 0
    err_count = 0
    for r in results:
        print(r)
        if r.startswith("[OK]"):
            ok_count += 1
        else:
            err_count += 1

    print(f"\nГотово. Успешно: {ok_count}, ошибок: {err_count}")


if __name__ == "__main__":
    main()