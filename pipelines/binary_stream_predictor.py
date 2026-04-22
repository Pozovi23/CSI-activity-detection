#!/usr/bin/env python3
"""Stream binary predictor: watch test_* folders and print 0/1 predictions."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

# Make repository root importable when script is called from any directory.
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.binary_predictor import BinaryPredictor


class DirectoryWatcher:
    """Poll watch directory for test_* folders and process each once when ready."""

    def __init__(
        self,
        watch_dir: Path,
        predictor: BinaryPredictor,
        poll_interval_sec: float,
        process_existing: bool,
    ) -> None:
        self.watch_dir = watch_dir
        self.predictor = predictor
        self.poll_interval_sec = poll_interval_sec
        self.processed_folders: set[Path] = set()

        if process_existing:
            return

        # Skip already existing test_* folders at startup.
        for folder in self._iter_test_dirs():
            self.processed_folders.add(folder.resolve())

    def _iter_test_dirs(self) -> list[Path]:
        if not self.watch_dir.exists() or not self.watch_dir.is_dir():
            return []
        return sorted(
            path for path in self.watch_dir.iterdir() if path.is_dir() and path.name.startswith("test_")
        )

    @staticmethod
    def _folder_has_three_esp_files(folder: Path) -> bool:
        # BinaryPredictor extracts dev ids from filenames like "...dev1...data".
        files = list(folder.glob("*.data"))
        names = [f.stem.lower() for f in files]
        return (
            any("dev1" in name for name in names)
            and any("dev2" in name for name in names)
            and any("dev3" in name for name in names)
        )

    def _try_process_folder(self, folder: Path) -> bool:
        folder_key = folder.resolve()
        if folder_key in self.processed_folders:
            return False

        if not self._folder_has_three_esp_files(folder):
            return False

        try:
            pred = self.predictor.predict_from_test_folder(folder)
        except Exception as exc:  # Keep watcher alive if one folder is malformed.
            print(f"[warn] failed to process {folder}: {exc}", file=sys.stderr)
            self.processed_folders.add(folder_key)
            return False

        ts = datetime.now().isoformat(timespec="seconds")
        print(ts)
        print(pred, flush=True)
        self.processed_folders.add(folder_key)
        return True

    def run_forever(self) -> None:
        while True:
            for folder in self._iter_test_dirs():
                self._try_process_folder(folder)
            time.sleep(self.poll_interval_sec)


def build_default_predictor(artifacts_dir: Path) -> BinaryPredictor:
    return BinaryPredictor(
        preproc_weights_dev1=artifacts_dir / "dev1" / "preprocessing.joblib",
        preproc_weights_dev2=artifacts_dir / "dev2" / "preprocessing.joblib",
        preproc_weights_dev3=artifacts_dir / "dev3" / "preprocessing.joblib",
        clf_weights_dev1=artifacts_dir / "dev1" / "model.joblib",
        clf_weights_dev2=artifacts_dir / "dev2" / "model.joblib",
        clf_weights_dev3=artifacts_dir / "dev3" / "model.joblib",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Watch csi_data/test_* folders and print binary predictions (0/1)"
    )
    parser.add_argument(
        "--watch-dir",
        type=Path,
        default=PROJECT_ROOT / "csi_data",
        help="Directory to watch for test_* folders (default: ./csi_data)",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "new_classic_ml_majority_vote_metrics_each_esp",
        help="Directory with dev1/dev2/dev3 preprocessing.joblib and model.joblib",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="Polling interval in seconds",
    )
    parser.add_argument(
        "--process-existing",
        action="store_true",
        help="Also process already existing test_* folders on startup",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    watch_dir = args.watch_dir.resolve()
    artifacts_dir = args.artifacts_dir.resolve()

    if not watch_dir.exists() or not watch_dir.is_dir():
        raise FileNotFoundError(f"Watch directory not found: {watch_dir}")

    required = [
        artifacts_dir / "dev1" / "preprocessing.joblib",
        artifacts_dir / "dev2" / "preprocessing.joblib",
        artifacts_dir / "dev3" / "preprocessing.joblib",
        artifacts_dir / "dev1" / "model.joblib",
        artifacts_dir / "dev2" / "model.joblib",
        artifacts_dir / "dev3" / "model.joblib",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing artifact files:\n" + "\n".join(missing))

    predictor = build_default_predictor(artifacts_dir)
    watcher = DirectoryWatcher(
        watch_dir=watch_dir,
        predictor=predictor,
        poll_interval_sec=max(float(args.poll_interval), 0.1),
        process_existing=bool(args.process_existing),
    )
    watcher.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
