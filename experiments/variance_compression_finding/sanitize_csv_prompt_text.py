#!/usr/bin/env python3
"""
Sanitize prompt text in CSV artifacts.

Creates a cleaned copy of a run directory or zip archive where long prompt text
columns are replaced with a short placeholder. Numeric bookkeeping columns such
as prompt_id, prompt_token_count, prompt_char_len are preserved.

Examples:
    python sanitize_csv_prompt_text.py /content/alignment_geometry_probability_run/run_20260613_082739

    python sanitize_csv_prompt_text.py /content/alignment_geometry_probability_run.zip \
        --output /content/alignment_geometry_probability_run_sanitized.zip

    python sanitize_csv_prompt_text.py ./run_dir \
        --extra-columns task base_text_full base_text_preview
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import zipfile
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import pandas as pd


DEFAULT_PLACEHOLDER = (
    "[PROMPT_TEXT_REMOVED_FOR_ANALYSIS] "
    "Long prompt/context text was removed from this CSV to reduce token load. "
    "Use the original raw artifact if exact prompt text is needed."
)

PRESERVE_EXACT_COLUMNS = {
    "prompt_id",
    "prompt_token_count",
    "prompt_char_len",
    "prompt_len",
    "prompt_length",
    "prompt_sha256",
    "prompt_hash",
    "final_kl_prompt_truncated",
    "tf_prompt_truncated",
}

TEXT_COLUMN_HINTS = (
    "prompt",
    "prompt_text",
    "full_prompt",
    "input_prompt",
)


def is_text_like(series: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return False
    sample = series.dropna().astype(str).head(20)
    if sample.empty:
        return False
    return bool(sample.map(lambda x: len(x.strip()) > 0).any())


def should_sanitize_column(column: str, series: pd.Series, extra_columns: Sequence[str]) -> bool:
    col = str(column)
    col_l = col.lower()
    if col_l in PRESERVE_EXACT_COLUMNS:
        return False
    if col in extra_columns or col_l in {x.lower() for x in extra_columns}:
        return is_text_like(series)
    if any(hint in col_l for hint in TEXT_COLUMN_HINTS):
        return is_text_like(series)
    return False


def sanitize_dataframe(
    df: pd.DataFrame,
    placeholder: str,
    extra_columns: Sequence[str],
) -> Tuple[pd.DataFrame, List[str], int]:
    out = df.copy()
    sanitized_columns: List[str] = []
    replaced_cells = 0
    for col in out.columns:
        if should_sanitize_column(str(col), out[col], extra_columns):
            mask = out[col].notna()
            replaced_cells += int(mask.sum())
            out.loc[mask, col] = placeholder
            sanitized_columns.append(str(col))
    return out, sanitized_columns, replaced_cells


def sanitize_csv_bytes(
    data: bytes,
    placeholder: str,
    extra_columns: Sequence[str],
) -> Tuple[bytes, dict]:
    df = pd.read_csv(io.BytesIO(data))
    cleaned, columns, replaced = sanitize_dataframe(df, placeholder, extra_columns)
    out = io.StringIO()
    cleaned.to_csv(out, index=False)
    meta = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "sanitized_columns": columns,
        "replaced_cells": int(replaced),
    }
    return out.getvalue().encode("utf-8"), meta


def iter_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file():
            yield path


def default_output_path(input_path: Path) -> Path:
    if input_path.suffix.lower() == ".zip":
        return input_path.with_name(input_path.stem + "_prompt_sanitized.zip")
    return input_path.with_name(input_path.name + "_prompt_sanitized")


def sanitize_directory(
    input_dir: Path,
    output_dir: Path,
    placeholder: str,
    extra_columns: Sequence[str],
) -> List[dict]:
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest: List[dict] = []
    for src in iter_files(input_dir):
        rel = src.relative_to(input_dir)
        dst = output_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix.lower() != ".csv":
            shutil.copy2(src, dst)
            continue
        try:
            data = src.read_bytes()
            cleaned_bytes, meta = sanitize_csv_bytes(data, placeholder, extra_columns)
            dst.write_bytes(cleaned_bytes)
            meta.update({"file": str(rel), "status": "sanitized"})
        except Exception as exc:
            shutil.copy2(src, dst)
            meta = {
                "file": str(rel),
                "status": "copied_after_error",
                "error": repr(exc),
                "sanitized_columns": [],
                "replaced_cells": 0,
            }
        manifest.append(meta)
    (output_dir / "prompt_sanitization_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def sanitize_zip(
    input_zip: Path,
    output_zip: Path,
    placeholder: str,
    extra_columns: Sequence[str],
) -> List[dict]:
    if output_zip.exists():
        raise FileExistsError(f"Output already exists: {output_zip}")
    manifest: List[dict] = []
    with zipfile.ZipFile(input_zip, "r") as zin, zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            data = zin.read(info.filename)
            if info.is_dir():
                zout.writestr(info, data)
                continue
            if not info.filename.lower().endswith(".csv"):
                zout.writestr(info, data)
                continue
            try:
                cleaned_bytes, meta = sanitize_csv_bytes(data, placeholder, extra_columns)
                zout.writestr(info.filename, cleaned_bytes)
                meta.update({"file": info.filename, "status": "sanitized"})
            except Exception as exc:
                zout.writestr(info.filename, data)
                meta = {
                    "file": info.filename,
                    "status": "copied_after_error",
                    "error": repr(exc),
                    "sanitized_columns": [],
                    "replaced_cells": 0,
                }
            manifest.append(meta)
        zout.writestr(
            "prompt_sanitization_manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2),
        )
    return manifest


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Replace long prompt text in CSV files with a compact placeholder.")
    ap.add_argument("input", help="Input run directory or .zip archive.")
    ap.add_argument("--output", default=None, help="Output directory or .zip path. Defaults to *_prompt_sanitized.")
    ap.add_argument("--placeholder", default=DEFAULT_PLACEHOLDER)
    ap.add_argument(
        "--extra-columns",
        nargs="*",
        default=[],
        help="Additional text columns to sanitize, e.g. task base_text_full base_text_preview.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(str(input_path))
    output_path = Path(args.output) if args.output else default_output_path(input_path)

    if input_path.is_dir():
        manifest = sanitize_directory(input_path, output_path, args.placeholder, args.extra_columns)
    elif input_path.suffix.lower() == ".zip":
        manifest = sanitize_zip(input_path, output_path, args.placeholder, args.extra_columns)
    else:
        raise ValueError(f"Input must be a directory or .zip archive: {input_path}")

    total_files = len(manifest)
    total_cells = sum(int(row.get("replaced_cells", 0)) for row in manifest)
    touched = [row for row in manifest if row.get("sanitized_columns")]
    print(f"saved: {output_path}")
    print(f"csv files scanned: {total_files}")
    print(f"csv files with sanitized columns: {len(touched)}")
    print(f"cells replaced: {total_cells}")
    for row in touched[:20]:
        print(f"- {row['file']}: {row['sanitized_columns']} cells={row['replaced_cells']}")
    if len(touched) > 20:
        print(f"... {len(touched) - 20} more")


if __name__ == "__main__":
    main()
