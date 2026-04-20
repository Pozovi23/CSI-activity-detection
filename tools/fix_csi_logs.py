#!/usr/bin/env python3
"""Fix CSI log files and add meaningful CSV headers.

What this script does:
1) Adds a header row with descriptive English column names.
2) Fixes rows where timestamp and record type are merged:
   "... +03:00 CSI_DATA,..." -> "... +03:00,CSI_DATA,..."
3) Repairs a malformed first data row by taking first 3 fields from row #1
   and fields 4..N from row #2 (if row #2 is valid).

By default, files are rewritten in place and a backup is created next to each
file with extension ".bak".


python tools/fix_csi_logs.py wifi_data_set --output-dir wifi_data_set_fixed

"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence


EXPECTED_COLUMNS = 26

HEADERS: List[str] = [
    "logger_timestamp",
    "record_type",
    "packet_seq",
    "source_mac",
    "rssi_dbm",
    "rate_code",
    "sig_mode",
    "mcs_index",
    "bandwidth",
    "smoothing",
    "not_sounding",
    "aggregation",
    "stbc",
    "fec_coding",
    "sgi",
    "noise_floor_dbm",
    "ampdu_count",
    "wifi_channel",
    "secondary_channel",
    "local_timestamp",
    "antenna",
    "signal_length",
    "rx_state",
    "csi_len",
    "first_word",
    "csi_data",
]


MERGED_TS_RECORD_RE = re.compile(
    r"^(\d{2}\.\d{2}\.\d{4}\s+\d{2}:\d{2}:\d{2}\.\d{3}\s+[+-]\d{2}:\d{2})\s+([A-Z_]+)$"
)


@dataclass
class FileStats:
    path: Path
    output_path: Path | None = None
    total_input_rows: int = 0
    kept_rows: int = 0
    dropped_rows: int = 0
    merged_ts_fixed_rows: int = 0
    first_row_repaired: bool = False
    already_had_header: bool = False
    changed: bool = False


def parse_line(line: str) -> List[str]:
    """Parse a CSV line robustly.

    Uses csv.reader first. If a row is malformed (e.g., broken quotes), falls
    back to a simple comma split.
    """
    try:
        row = next(csv.reader([line], delimiter=",", quotechar='"'))
    except csv.Error:
        row = line.split(",")
    return [cell.strip() for cell in row]


def normalize_row(row: Sequence[str]) -> tuple[List[str], bool]:
    """Normalize one parsed row and split merged timestamp+record_type if needed."""
    normalized = list(row)
    merged_fixed = False

    if normalized:
        match = MERGED_TS_RECORD_RE.match(normalized[0])
        if match:
            ts, record_type = match.groups()
            normalized = [ts, record_type] + normalized[1:]
            merged_fixed = True

    return normalized, merged_fixed


def is_header_row(row: Sequence[str]) -> bool:
    if len(row) < 2:
        return False
    return row[0] == HEADERS[0] and row[1] == HEADERS[1]


def is_valid_data_row(row: Sequence[str]) -> bool:
    return len(row) == EXPECTED_COLUMNS


def is_valid_csi_data_field(value: str) -> bool:
    text = value.strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1].strip()
    return text.startswith("[") and text.endswith("]")


def read_rows(path: Path) -> List[List[str]]:
    rows: List[List[str]] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            rows.append(parse_line(line))
    return rows


def process_rows(raw_rows: List[List[str]], stats: FileStats) -> List[List[str]]:
    if not raw_rows:
        return []

    rows: List[List[str]] = []
    for row in raw_rows:
        normalized, merged_fixed = normalize_row(row)
        if merged_fixed:
            stats.merged_ts_fixed_rows += 1
        rows.append(normalized)

    if is_header_row(rows[0]):
        stats.already_had_header = True
        rows = rows[1:]

    if len(rows) >= 2 and is_valid_data_row(rows[1]) and len(rows[0]) >= 3:
        first_row_needs_repair = (not is_valid_data_row(rows[0])) or (
            is_valid_data_row(rows[0]) and not is_valid_csi_data_field(rows[0][-1])
        )
        if first_row_needs_repair:
            rows[0] = rows[0][:3] + rows[1][3:]
            stats.first_row_repaired = True

    valid_rows: List[List[str]] = []
    for row in rows:
        if is_valid_data_row(row):
            valid_rows.append(row)
        else:
            stats.dropped_rows += 1

    stats.kept_rows = len(valid_rows)
    return valid_rows


def write_rows(path: Path, rows: List[List[str]], backup: bool) -> None:
    if backup:
        backup_path = path.with_suffix(path.suffix + ".bak")
        backup_path.write_bytes(path.read_bytes())

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(HEADERS)
        writer.writerows(rows)


def process_file(path: Path, output_path: Path, dry_run: bool, backup: bool) -> FileStats:
    stats = FileStats(path=path, output_path=output_path)
    raw_rows = read_rows(path)
    stats.total_input_rows = len(raw_rows)

    fixed_rows = process_rows(raw_rows, stats)

    output_would_change = (
        stats.merged_ts_fixed_rows > 0
        or stats.first_row_repaired
        or stats.dropped_rows > 0
        or not stats.already_had_header
    )

    stats.changed = output_would_change

    if not dry_run:
        write_rows(output_path, fixed_rows, backup=backup)

    return stats


def iter_target_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".csv", ".data"}:
            continue
        files.append(path)
    files.sort()
    return files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fix CSI log files: add header, split merged timestamp/record_type, "
            "repair malformed first row from second row."
        )
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to a CSI log file or a directory to scan recursively.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze and report changes without rewriting files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory where fixed files will be written (source structure is preserved). "
            "If omitted, a sibling '<input>_fixed' directory is used."
        ),
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Rewrite source files in place. Use with caution.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create .bak backup files before in-place rewriting.",
    )
    return parser


def print_file_summary(stats: FileStats) -> None:
    output_part = str(stats.output_path) if stats.output_path else "<none>"
    print(
        f"{stats.path} -> {output_part}: rows_in={stats.total_input_rows}, rows_out={stats.kept_rows}, "
        f"dropped={stats.dropped_rows}, merged_fixed={stats.merged_ts_fixed_rows}, "
        f"first_row_repaired={stats.first_row_repaired}, header_present={stats.already_had_header}, "
        f"changed={stats.changed}"
    )


def resolve_output_root(input_path: Path, explicit_output_dir: Path | None) -> Path:
    if explicit_output_dir is not None:
        return explicit_output_dir
    if input_path.is_dir():
        return input_path.parent / f"{input_path.name}_fixed"
    return input_path.parent / "fixed_output"


def resolve_output_path(input_root: Path, output_root: Path, source_file: Path) -> Path:
    if input_root.is_file():
        return output_root / source_file.name
    relative_path = source_file.relative_to(input_root)
    return output_root / relative_path


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    input_path: Path = args.input_path
    if not input_path.exists():
        print(f"Error: path does not exist: {input_path}", file=sys.stderr)
        return 2

    backup = not args.no_backup

    if input_path.is_file():
        targets = [input_path]
    else:
        targets = iter_target_files(input_path)

    if not targets:
        print("No .csv/.data files found.")
        return 0

    if args.in_place and args.output_dir is not None:
        print("Error: --in-place and --output-dir cannot be used together.", file=sys.stderr)
        return 2

    output_root = None
    if not args.in_place:
        output_root = resolve_output_root(input_path, args.output_dir)
        if output_root.resolve() == input_path.resolve():
            print(
                "Error: output directory must differ from input path when not using --in-place.",
                file=sys.stderr,
            )
            return 2

    changed_count = 0
    processed_count = 0

    for path in targets:
        try:
            if args.in_place:
                output_path = path
            else:
                output_path = resolve_output_path(input_path, output_root, path)

            stats = process_file(path, output_path, dry_run=args.dry_run, backup=backup and args.in_place)
            print_file_summary(stats)
            processed_count += 1
            if stats.changed:
                changed_count += 1
        except Exception as exc:  # Keep going even if one file fails.
            print(f"ERROR processing {path}: {exc}", file=sys.stderr)

    output_mode = "in-place" if args.in_place else f"to={output_root}"
    print(
        f"Done. processed={processed_count}, changed={changed_count}, dry_run={args.dry_run}, "
        f"mode={output_mode}, backup={backup and args.in_place}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
