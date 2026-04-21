#!/usr/bin/env python3
"""End-to-end binary prediction pipeline for three ESP streams."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from inference.binary_classificator import BinaryClassificator
from tools.csi_parser import Parser
from tools.preprocessor import Preprocessor


class BinaryPredictor:
    """Pipeline: parse -> preprocess (per ESP) -> majority-vote classify."""

    def __init__(
        self,
        preproc_weights_dev1,
        preproc_weights_dev2,
        preproc_weights_dev3,
        clf_weights_dev1,
        clf_weights_dev2,
        clf_weights_dev3,
    ) -> None:
        self.preprocessor_dev1 = Preprocessor(preproc_weights_dev1)
        self.preprocessor_dev2 = Preprocessor(preproc_weights_dev2)
        self.preprocessor_dev3 = Preprocessor(preproc_weights_dev3)

        self.classificator = BinaryClassificator(
            clf_weights_dev1,
            clf_weights_dev2,
            clf_weights_dev3,
        )

    @staticmethod
    def _parse_file(path: str | Path) -> pd.DataFrame:
        return Parser(path).parse()

    @staticmethod
    def _extract_esp_id(file_path: Path) -> str:
        match = re.search(r"dev(\d+)", file_path.stem, flags=re.IGNORECASE)
        if not match:
            raise ValueError(f"Could not extract ESP id from filename: {file_path.name}")
        return f"dev{int(match.group(1))}"

    def predict_from_dataframes(self, df_dev1: pd.DataFrame, df_dev2: pd.DataFrame, df_dev3: pd.DataFrame) -> int:
        emb1 = self.preprocessor_dev1.preprocess(df_dev1)
        emb2 = self.preprocessor_dev2.preprocess(df_dev2)
        emb3 = self.preprocessor_dev3.preprocess(df_dev3)
        return self.classificator.predict(emb1, emb2, emb3)

    def predict_from_files(self, file_dev1: str | Path, file_dev2: str | Path, file_dev3: str | Path) -> int:
        df1 = self._parse_file(file_dev1)
        df2 = self._parse_file(file_dev2)
        df3 = self._parse_file(file_dev3)
        return self.predict_from_dataframes(df1, df2, df3)

    def predict_from_test_folder(self, test_folder: str | Path) -> int:
        folder = Path(test_folder)
        if not folder.exists() or not folder.is_dir():
            raise ValueError(f"Invalid test folder: {folder}")

        files = sorted(folder.glob("*.data"))
        if not files:
            raise FileNotFoundError(f"No .data files found in {folder}")

        per_esp: dict[str, Path] = {}
        for file_path in files:
            esp_id = self._extract_esp_id(file_path)
            per_esp[esp_id] = file_path

        required = ("dev1", "dev2", "dev3")
        missing = [esp for esp in required if esp not in per_esp]
        if missing:
            raise ValueError(f"Missing files for ESP ids: {missing}")

        return self.predict_from_files(per_esp["dev1"], per_esp["dev2"], per_esp["dev3"])
