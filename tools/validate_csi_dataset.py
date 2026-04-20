#!/usr/bin/env python3
"""Validate CSI .data files against a reference file structure."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_REFERENCE = Path(
    "data/wifi_data_set/id_person_01/label_00/test_01/"
    "test1__dev1_64_E8_33_57_AA_F4.data"
)
DEFAULT_DATASET = Path("data/wifi_data_set")

FIRST_FIELD_RE = re.compile(
    r"^\d{2}\.\d{2}\.\d{4} "
    r"\d{2}:\d{2}:\d{2}\.\d{3} "
    r"[+-]\d{2}:\d{2} "
    r"CSI_DATA$"
)
MAC_RE = re.compile(r"^[0-9a-fA-F]{2}(?::[0-9a-fA-F]{2}){5}$")


@dataclass(frozen=True)
class CsiStructure:
    csv_fields: int
    metadata_fields: int
    csi_values: int
    lines: int


@dataclass(frozen=True)
class Problem:
    path: Path
    line: int | None
    message: str


def parse_int_list(cell: str) -> list[int]:
    if not cell.startswith("[") or not cell.endswith("]"):
        raise ValueError("CSI cell is not enclosed in []")

    body = cell[1:-1]
    if not body:
        return []

    values: list[int] = []
    for index, raw_value in enumerate(body.split(","), start=1):
        value = raw_value.strip()
        if not re.fullmatch(r"[+-]?\d+", value):
            raise ValueError(f"CSI value #{index} is not an integer: {value!r}")
        values.append(int(value))
    return values


def read_csv_rows(path: Path) -> list[list[str]]:
    with path.open("r", encoding="ascii", newline="") as file:
        return list(csv.reader(file))


def validate_row(row: list[str], structure: CsiStructure, path: Path, line: int) -> list[Problem]:
    problems: list[Problem] = []

    if len(row) != structure.csv_fields:
        problems.append(
            Problem(
                path,
                line,
                f"expected {structure.csv_fields} CSV fields, got {len(row)}",
            )
        )
        return problems

    if not FIRST_FIELD_RE.fullmatch(row[0]):
        problems.append(
            Problem(
                path,
                line,
                "first field does not match '<date> <time> <timezone> CSI_DATA'",
            )
        )

    if not MAC_RE.fullmatch(row[2]):
        problems.append(Problem(path, line, f"invalid MAC address field: {row[2]!r}"))

    for field_index in range(1, structure.metadata_fields):
        if field_index == 2:
            continue
        value = row[field_index]
        if not re.fullmatch(r"[+-]?\d+", value):
            problems.append(
                Problem(
                    path,
                    line,
                    f"metadata field #{field_index} is not an integer: {value!r}",
                )
            )

    try:
        csi_values = parse_int_list(row[-1])
    except ValueError as exc:
        problems.append(Problem(path, line, str(exc)))
        return problems

    if len(csi_values) != structure.csi_values:
        problems.append(
            Problem(
                path,
                line,
                f"expected {structure.csi_values} CSI values, got {len(csi_values)}",
            )
        )

    declared_csi_len = row[structure.metadata_fields - 2]
    if declared_csi_len.isdigit() and int(declared_csi_len) != len(csi_values):
        problems.append(
            Problem(
                path,
                line,
                f"declared CSI length is {declared_csi_len}, actual length is {len(csi_values)}",
            )
        )

    return problems


def infer_structure(reference: Path) -> tuple[CsiStructure, list[Problem]]:
    rows = read_csv_rows(reference)
    if not rows:
        return CsiStructure(0, 0, 0, 0), [Problem(reference, None, "reference file is empty")]

    first_csi_values = parse_int_list(rows[0][-1]) if rows[0] else []
    structure = CsiStructure(
        csv_fields=len(rows[0]),
        metadata_fields=len(rows[0]) - 1,
        csi_values=len(first_csi_values),
        lines=len(rows),
    )

    problems: list[Problem] = []
    for line, row in enumerate(rows, start=1):
        problems.extend(validate_row(row, structure, reference, line))

    return structure, problems


def validate_file(path: Path, structure: CsiStructure, check_line_count: bool) -> list[Problem]:
    try:
        rows = read_csv_rows(path)
    except UnicodeDecodeError as exc:
        return [Problem(path, None, f"file is not ASCII text: {exc}")]
    except csv.Error as exc:
        return [Problem(path, None, f"CSV parser error: {exc}")]

    problems: list[Problem] = []

    if check_line_count and len(rows) != structure.lines:
        problems.append(
            Problem(
                path,
                None,
                f"expected {structure.lines} rows, got {len(rows)}",
            )
        )

    for line, row in enumerate(rows, start=1):
        problems.extend(validate_row(row, structure, path, line))

    return problems


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
        print("OK: all files match the reference structure")
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
        description="Find .data files that do not match a reference CSI data structure."
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REFERENCE,
        help=f"reference .data file (default: {DEFAULT_REFERENCE})",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"dataset directory or a single .data file (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--no-line-count",
        action="store_true",
        help="do not require files to have the same number of rows as the reference",
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
                check_line_count=not args.no_line_count,
            )
        )

    print(
        "Reference structure: "
        f"{structure.lines} rows, "
        f"{structure.csv_fields} CSV fields, "
        f"{structure.metadata_fields} metadata fields, "
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
