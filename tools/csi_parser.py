#!/usr/bin/env python3
"""Parser for fixed ESP32 CSI log files."""

from __future__ import annotations

import ast
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Parser:
    """Parse one fixed CSI file and return a pandas DataFrame."""

    file_path: str | Path

    def __post_init__(self) -> None:
        self.path = Path(self.file_path)
        if not self.path.exists():
            raise FileNotFoundError(f"File does not exist: {self.path}")
        if not self.path.is_file():
            raise ValueError(f"Path is not a file: {self.path}")

    @staticmethod
    def _to_int(value: str, default: int = 0) -> int:
        text = (value or "").strip()
        if text == "":
            return default
        try:
            return int(text)
        except ValueError:
            return default

    @staticmethod
    def _parse_csi_data_field(raw_value: str) -> list[int]:
        text = (raw_value or "").strip()
        if text.startswith('"') and text.endswith('"') and len(text) >= 2:
            text = text[1:-1].strip()
        if not (text.startswith("[") and text.endswith("]")):
            raise ValueError(f"Invalid csi_data format: {raw_value}")

        parsed = ast.literal_eval(text)
        if not isinstance(parsed, list):
            raise ValueError("csi_data is not a list")

        out: list[int] = []
        for item in parsed:
            out.append(int(item))
        return out

    @staticmethod
    def _iq_to_complex(iq_list: list[int]) -> np.ndarray:
        if len(iq_list) % 2 != 0:
            raise ValueError("csi_data length must be even (I/Q pairs)")
        arr = np.asarray(iq_list, dtype=np.float32).reshape(-1, 2)
        # Raw order is imag, real, imag, real, ...
        imag = arr[:, 0]
        real = arr[:, 1]
        return real + 1j * imag

    @staticmethod
    def _iq_to_amplitude(iq_list: list[int]) -> np.ndarray:
        """Compute amplitude exactly as sqrt(imag^2 + real^2) per subcarrier."""
        if len(iq_list) % 2 != 0:
            raise ValueError("csi_data length must be even (imag/real pairs)")
        x = np.asarray(iq_list, dtype=np.float32)
        return np.sqrt(x[0::2] ** 2 + x[1::2] ** 2)

    @staticmethod
    def _complex_dict(csi_complex: np.ndarray) -> dict[int, complex]:
        return {idx: complex(value) for idx, value in enumerate(csi_complex.tolist())}

    def parse(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []

        with self.path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError(f"No header row in file: {self.path}")

            for idx, row in enumerate(reader, start=2):
                try:
                    csi_iq = self._parse_csi_data_field(row.get("csi_data", ""))
                    csi_complex = self._iq_to_complex(csi_iq)
                    amplitude = self._iq_to_amplitude(csi_iq)
                except Exception as exc:
                    raise ValueError(f"{self.path}: line {idx}: {exc}") from exc

                packet_seq = self._to_int(row.get("packet_seq", "0"))
                rssi = self._to_int(row.get("rssi_dbm", "0"))
                channel = self._to_int(row.get("wifi_channel", "0"))

                parsed_row = dict(row)
                parsed_row["packet_seq"] = packet_seq
                parsed_row["rssi_dbm"] = rssi
                parsed_row["wifi_channel"] = channel
                parsed_row["csi_data"] = self._complex_dict(csi_complex)
                parsed_row["amplitude"] = amplitude
                parsed_row["csi_complex"] = csi_complex
                parsed_row["phase"] = np.angle(csi_complex)

                rows.append(parsed_row)

        if not rows:
            raise ValueError(f"No data rows in file: {self.path}")

        csi_complex_rows = [row["csi_complex"] for row in rows]
        subcarrier_count = len(csi_complex_rows[0])
        for i, comp in enumerate(csi_complex_rows):
            if len(comp) != subcarrier_count:
                raise ValueError(
                    f"{self.path}: inconsistent subcarrier count at packet index {i}"
                )

        return pd.DataFrame(rows)
