#!/usr/bin/env python3
"""Validate fixed CSI .data files against the fixed dataset structure."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_REFERENCE = Path(
    "wifi_data_set_fixed/id_person_01/label_00/test_01/"
    "test1__dev1_64_E8_33_57_AA_F4.data"
)
DEFAULT_DATASET = Path("wifi_data_set_fixed")

TIMESTAMP_RE = re.compile(
    r"^\d{2}\.\d{2}\.\d{4} "
    r"\d{2}:\d{2}:\d{2}\.\d{3} "
    r"[+-]\d{2}:\d{2}$"
)
MAC_RE = re.compile(r"^[0-9a-fA-F]{2}(?::[0-9a-fA-F]{2}){5}$")
INT_RE = re.compile(r"[+-]?\d+")


@dataclass(frozen=True)
class FixedCsiStructure:
    header: tuple[str, ...]
    rows: int
    csi_values: int


@dataclass(frozen=True)
class Problem:
    path: Path
    line: int | None
    message: str


def parse_int_list(cell: str) -> list[int]:
    if not cell.startswith("[") or not cell.endswith("]"):
        raise ValueError("csi_data is not enclosed in []")

    body = cell[1:-1]
    if not body:
        return []

    values: list[int] = []
    for index, raw_value in enumerate(body.split(","), start=1):
        value = raw_value.strip()
        if not INT_RE.fullmatch(value):
            raise ValueError(f"CSI value #{index} is not an integer: {value!r}")
        values.append(int(value))
    return values


def read_csv_rows(path: Path) -> list[list[str]]:
    with path.open("r", encoding="ascii", newline="") as file:
        return list(csv.reader(file))


def infer_structure(reference: Path) -> tuple[FixedCsiStructure, list[Problem]]:
    rows = read_csv_rows(reference)
    if not rows:
        return FixedCsiStructure((), 0, 0), [Problem(reference, None, "reference file is empty")]
    if len(rows) == 1:
        return FixedCsiStructure(tuple(rows[0]), 0, 0), [
            Problem(reference, None, "reference file has only a header")
        ]

    header = tuple(rows[0])
    first_data_row = rows[1]
    if "csi_data" not in header:
        return FixedCsiStructure(header, 0, 0), [
            Problem(reference, 1, "header does not contain csi_data column")
        ]

    csi_index = header.index("csi_data")
    csi_values = parse_int_list(first_data_row[csi_index])
    structure = FixedCsiStructure(
        header=header,
        rows=len(rows) - 1,
        csi_values=len(csi_values),
    )

    problems = validate_rows(reference, rows, structure, check_row_count=True)
    return structure, problems


def validate_rows(
    path: Path,
    rows: list[list[str]],
    structure: FixedCsiStructure,
    *,
    check_row_count: bool,
) -> list[Problem]:
    problems: list[Problem] = []

    if not rows:
        return [Problem(path, None, "file is empty")]

    header = tuple(rows[0])
    if header != structure.header:
        problems.append(Problem(path, 1, "header does not match the reference header"))
        return problems

    data_rows = rows[1:]
    if check_row_count and len(data_rows) != structure.rows:
        problems.append(
            Problem(
                path,
                None,
                f"expected {structure.rows} data rows, got {len(data_rows)}",
            )
        )

    column_count = len(structure.header)
    column_index = {name: index for index, name in enumerate(structure.header)}
    csi_index = column_index["csi_data"]
    csi_len_index = column_index.get("csi_len")

    text_columns = {"logger_timestamp", "record_type", "source_mac", "csi_data"}

    for row_number, row in enumerate(data_rows, start=2):
        if len(row) != column_count:
            problems.append(
                Problem(
                    path,
                    row_number,
                    f"expected {column_count} CSV fields, got {len(row)}",
                )
            )
            continue

        timestamp = row[column_index["logger_timestamp"]]
        if not TIMESTAMP_RE.fullmatch(timestamp):
            problems.append(
                Problem(path, row_number, f"invalid logger_timestamp: {timestamp!r}")
            )

        record_type = row[column_index["record_type"]]
        if record_type != "CSI_DATA":
            problems.append(
                Problem(path, row_number, f"expected record_type CSI_DATA, got {record_type!r}")
            )

        source_mac = row[column_index["source_mac"]]
        if not MAC_RE.fullmatch(source_mac):
            problems.append(Problem(path, row_number, f"invalid source_mac: {source_mac!r}"))

        for field_index, column_name in enumerate(structure.header):
            if column_name in text_columns:
                continue
            value = row[field_index]
            if not INT_RE.fullmatch(value):
                problems.append(
                    Problem(
                        path,
                        row_number,
                        f"column {column_name!r} is not an integer: {value!r}",
                    )
                )

        try:
            csi_values = parse_int_list(row[csi_index])
        except ValueError as exc:
            problems.append(Problem(path, row_number, str(exc)))
            continue

        if len(csi_values) != structure.csi_values:
            problems.append(
                Problem(
                    path,
                    row_number,
                    f"expected {structure.csi_values} CSI values, got {len(csi_values)}",
                )
            )

        if csi_len_index is not None and INT_RE.fullmatch(row[csi_len_index]):
            declared_len = int(row[csi_len_index])
            if declared_len != len(csi_values):
                problems.append(
                    Problem(
                        path,
                        row_number,
                        f"declared csi_len is {declared_len}, actual length is {len(csi_values)}",
                    )
                )

    return problems


def validate_file(
    path: Path,
    structure: FixedCsiStructure,
    *,
    check_row_count: bool,
) -> list[Problem]:
    try:
        rows = read_csv_rows(path)
    except UnicodeDecodeError as exc:
        return [Problem(path, None, f"file is not ASCII text: {exc}")]
    except csv.Error as exc:
        return [Problem(path, None, f"CSV parser error: {exc}")]

    return validate_rows(path, rows, structure, check_row_count=check_row_count)


def iter_data_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(root.rglob("*.data"))


def print_report(
    problems: list[Problem],
    dataset_root: Path,
    *,
    summary_only: bool = False,
    max_files: int | None = None,
) -> None:
    if not problems:
        print("OK: all files match the fixed reference structure")
        return

    by_path: dict[Path, list[Problem]] = {}
    for problem in problems:
        by_path.setdefault(problem.path, []).append(problem)

    print(f"Found {len(by_path)} invalid file(s), {len(problems)} problem(s):")
    if summary_only:
        return

    sorted_items = sorted(by_path.items())
    shown_items = sorted_items[:max_files] if max_files is not None else sorted_items

    for path, path_problems in shown_items:
        try:
            display_path = path.relative_to(dataset_root)
        except ValueError:
            display_path = path

        print(f"\n{display_path}")
        for problem in path_problems:
            location = f"line {problem.line}: " if problem.line is not None else ""
            print(f"  - {location}{problem.message}")

    hidden = len(sorted_items) - len(shown_items)
    if hidden > 0:
        print(f"\n... {hidden} more invalid file(s) not shown")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Find fixed .data files that do not match a reference CSI CSV structure."
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REFERENCE,
        help=f"reference fixed .data file (default: {DEFAULT_REFERENCE})",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"fixed dataset directory or a single .data file (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--no-row-count",
        action="store_true",
        help="do not require files to have the same number of data rows as the reference",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="print only totals, without listing every invalid file",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="maximum number of invalid files to print",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    reference = args.reference
    dataset_root = args.dataset_root

    if not reference.exists():
        print(f"Reference file does not exist: {reference}", file=sys.stderr)
        return 2
    if not dataset_root.exists():
        print(f"Dataset path does not exist: {dataset_root}", file=sys.stderr)
        return 2

    try:
        structure, reference_problems = infer_structure(reference)
    except (UnicodeDecodeError, csv.Error, ValueError) as exc:
        print(f"Could not infer structure from reference: {exc}", file=sys.stderr)
        return 2

    if reference_problems:
        print("Reference file does not have a consistent supported structure:", file=sys.stderr)
        print_report(reference_problems, reference.parent)
        return 2

    data_files = iter_data_files(dataset_root)
    problems: list[Problem] = []
    for path in data_files:
        problems.extend(
            validate_file(
                path,
                structure,
                check_row_count=not args.no_row_count,
            )
        )

    print(
        "Reference fixed structure: "
        f"{structure.rows} data rows, "
        f"{len(structure.header)} CSV fields, "
        f"{structure.csi_values} CSI values"
    )
    print(f"Checked {len(data_files)} file(s)")
    print_report(
        problems,
        dataset_root,
        summary_only=args.summary_only,
        max_files=args.max_files,
    )

    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
