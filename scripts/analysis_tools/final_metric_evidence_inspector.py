#!/usr/bin/env python3
"""
Stream-inspect FINAL_* metric-lab evidence tables inside result zips/directories.

The metric-lab FINAL files can be several GB uncompressed. This inspector reads
them as a stream, avoids extracting the large CSV, and writes compact audit
tables that are practical to inspect.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import math
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import BinaryIO, Iterable, TextIO


FINAL_CANDIDATES = [
    "FINAL_DERIVED_METRIC_EVIDENCE.csv",
    "FINAL_LATENT_ATTRACTOR_METRICS.csv",
]

FOCUS_RE = re.compile(
    r"("
    r"grade4|x_order|x_content|x_full|x_order_orth|"
    r"component|causal|alpha|projection|norm|energy|"
    r"sae|feature|reconstruction|explained_variance|"
    r"target|sentence_shuffle|word_shuffle|shuffle|"
    r"state_space|pca|centroid|trajectory"
    r")",
    re.IGNORECASE,
)


def safe_label(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("_") or "run"


def parse_input_arg(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, path = value.split("=", 1)
        return safe_label(label), Path(path).expanduser().resolve()
    path = Path(value).expanduser().resolve()
    return safe_label(path.stem), path


def choose_final_member(names: Iterable[str]) -> str | None:
    name_list = list(names)
    for candidate in FINAL_CANDIDATES:
        matches = [n for n in name_list if Path(n).name == candidate]
        if matches:
            return sorted(matches, key=lambda x: (len(x), x))[0]
    matches = [
        n
        for n in name_list
        if Path(n).name.upper().startswith("FINAL_") and n.lower().endswith(".csv")
    ]
    if not matches:
        return None
    return sorted(matches, key=lambda x: (len(x), x))[0]


class FinalCsvSource:
    def __init__(self, path: Path):
        self.path = path
        self.archive: zipfile.ZipFile | None = None
        self.member_name: str | None = None
        self.file_path: Path | None = None

    def __enter__(self) -> "FinalCsvSource":
        if self.path.is_file() and self.path.suffix.lower() == ".zip":
            self.archive = zipfile.ZipFile(self.path)
            self.member_name = choose_final_member(self.archive.namelist())
            if self.member_name is None:
                raise FileNotFoundError(f"No FINAL_*.csv found in {self.path}")
            return self
        if self.path.is_file() and self.path.name.upper().startswith("FINAL_"):
            self.file_path = self.path
            return self
        if self.path.is_dir():
            candidates = [
                p
                for p in self.path.rglob("FINAL_*.csv")
                if p.name in FINAL_CANDIDATES or p.name.upper().startswith("FINAL_")
            ]
            if not candidates:
                raise FileNotFoundError(f"No FINAL_*.csv found under {self.path}")
            preferred = [p for p in candidates if p.name == "FINAL_DERIVED_METRIC_EVIDENCE.csv"]
            self.file_path = sorted(preferred or candidates, key=lambda x: (len(str(x)), str(x)))[0]
            return self
        raise FileNotFoundError(f"Input is not a zip, FINAL csv, or directory: {self.path}")

    def __exit__(self, *args) -> None:
        if self.archive is not None:
            self.archive.close()

    def open_text(self) -> TextIO:
        if self.archive is not None and self.member_name is not None:
            raw: BinaryIO = self.archive.open(self.member_name)
            return open_text_wrapper(raw)
        if self.file_path is not None:
            return self.file_path.open("r", encoding="utf-8", errors="replace", newline="")
        raise RuntimeError("Source is not initialized")

    def selected_name(self) -> str:
        if self.member_name is not None:
            return self.member_name
        if self.file_path is not None:
            return str(self.file_path)
        return ""

    def selected_size(self) -> int:
        if self.archive is not None and self.member_name is not None:
            return int(self.archive.getinfo(self.member_name).file_size)
        if self.file_path is not None:
            return int(self.file_path.stat().st_size)
        return 0


def open_text_wrapper(raw: BinaryIO) -> TextIO:
    import io

    return io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline="")


def parse_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        val = float(text)
    except Exception:
        return None
    if not math.isfinite(val):
        return None
    return val


def compact_row(row: dict[str, str], value: float | None = None) -> dict[str, object]:
    return {
        "evidence_type": row.get("evidence_type", ""),
        "source_file": row.get("source_file", ""),
        "formalism_class": row.get("formalism_class", ""),
        "metric": row.get("metric", ""),
        "context": row.get("context", ""),
        "primary_value_name": row.get("primary_value_name", ""),
        "primary_value": row.get("primary_value", ""),
        "primary_value_float": "" if value is None else value,
        "secondary_value_name": row.get("secondary_value_name", ""),
        "secondary_value": row.get("secondary_value", ""),
        "n": row.get("n", ""),
        "details_json": row.get("details_json", ""),
    }


def heap_push(heap: list[tuple[float, int, dict[str, object]]], max_size: int, score: float, serial: int, row: dict[str, object]) -> None:
    item = (float(score), int(serial), row)
    if len(heap) < max_size:
        heapq.heappush(heap, item)
    elif score > heap[0][0]:
        heapq.heapreplace(heap, item)


def write_counter(path: Path, counter: Counter, key_name: str, total_rows: int | None = None) -> None:
    rows = []
    for key, count in counter.most_common():
        row = {key_name: key, "rows": count}
        if total_rows:
            row["row_fraction"] = count / float(total_rows)
        rows.append(row)
    write_rows(path, rows, fieldnames=[key_name, "rows", "row_fraction"] if total_rows else [key_name, "rows"])


def write_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    keys.append(key)
                    seen.add(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def inspect_one(label: str, path: Path, out_dir: Path, top_n: int) -> dict[str, object]:
    run_dir = out_dir / label
    run_dir.mkdir(parents=True, exist_ok=True)

    evidence_counts = Counter()
    source_counts = Counter()
    formalism_counts = Counter()
    metric_counts = Counter()
    source_metric_counts = Counter()
    metric_context_counts = Counter()

    top_abs: list[tuple[float, int, dict[str, object]]] = []
    focus_abs: list[tuple[float, int, dict[str, object]]] = []
    grade4_abs: list[tuple[float, int, dict[str, object]]] = []
    sae_abs: list[tuple[float, int, dict[str, object]]] = []

    row_count = 0
    numeric_count = 0
    blank_primary_count = 0
    serial = 0

    with FinalCsvSource(path) as source:
        selected_name = source.selected_name()
        selected_size = source.selected_size()
        with source.open_text() as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            for row in reader:
                row_count += 1
                serial += 1
                evidence = row.get("evidence_type", "")
                src = row.get("source_file", "")
                formalism = row.get("formalism_class", "")
                metric = row.get("metric", "")
                context = row.get("context", "")
                primary = row.get("primary_value", "")

                evidence_counts[evidence] += 1
                source_counts[src] += 1
                formalism_counts[formalism] += 1
                metric_counts[metric] += 1
                source_metric_counts[(src, metric)] += 1
                metric_context_counts[(metric, context)] += 1

                val = parse_float(primary)
                if val is None:
                    if str(primary).strip() == "":
                        blank_primary_count += 1
                    continue
                numeric_count += 1
                compact = compact_row(row, val)
                score = abs(val)
                heap_push(top_abs, top_n, score, serial, compact)

                search_blob = " | ".join([src, metric, context, formalism, evidence])
                if FOCUS_RE.search(search_blob):
                    heap_push(focus_abs, top_n * 3, score, serial, compact)
                if "grade4" in search_blob.lower() or "x_order" in search_blob.lower() or "x_content" in search_blob.lower():
                    heap_push(grade4_abs, top_n * 3, score, serial, compact)
                if "sae" in search_blob.lower() or "feature" in search_blob.lower() or "reconstruction" in search_blob.lower():
                    heap_push(sae_abs, top_n * 3, score, serial, compact)

    source_metric_rows = [
        {"source_file": src, "metric": metric, "rows": count}
        for (src, metric), count in source_metric_counts.most_common()
    ]
    metric_context_rows = [
        {"metric": metric, "context": context, "rows": count}
        for (metric, context), count in metric_context_counts.most_common()
    ]

    def heap_rows(heap: list[tuple[float, int, dict[str, object]]]) -> list[dict[str, object]]:
        return [row for _, _, row in sorted(heap, key=lambda item: (-item[0], item[1]))]

    write_counter(run_dir / "final_evidence_type_counts.csv", evidence_counts, "evidence_type", row_count)
    write_counter(run_dir / "final_source_file_counts.csv", source_counts, "source_file", row_count)
    write_counter(run_dir / "final_formalism_counts.csv", formalism_counts, "formalism_class", row_count)
    write_counter(run_dir / "final_metric_counts.csv", metric_counts, "metric", row_count)
    write_rows(run_dir / "final_source_metric_counts.csv", source_metric_rows)
    write_rows(run_dir / "final_metric_context_counts.csv", metric_context_rows)
    write_rows(run_dir / "final_top_abs_primary_values.csv", heap_rows(top_abs))
    write_rows(run_dir / "final_focus_top_abs_rows.csv", heap_rows(focus_abs))
    write_rows(run_dir / "final_grade4_top_abs_rows.csv", heap_rows(grade4_abs))
    write_rows(run_dir / "final_sae_top_abs_rows.csv", heap_rows(sae_abs))

    summary = {
        "label": label,
        "input_path": str(path),
        "selected_final_csv": selected_name,
        "selected_final_uncompressed_size_mb": round(selected_size / 1024 / 1024, 3),
        "row_count": row_count,
        "numeric_primary_count": numeric_count,
        "blank_primary_count": blank_primary_count,
        "unique_source_files": len(source_counts),
        "unique_metrics": len(metric_counts),
        "unique_formalism_classes": len(formalism_counts),
        "top_source_file": source_counts.most_common(1)[0][0] if source_counts else "",
        "top_source_file_rows": source_counts.most_common(1)[0][1] if source_counts else 0,
    }
    write_rows(run_dir / "final_inspection_summary.csv", [summary])
    write_summary_md(run_dir / "summary.md", summary, evidence_counts, source_counts, formalism_counts, metric_counts)
    return summary


def write_summary_md(path: Path, summary: dict[str, object], evidence_counts: Counter, source_counts: Counter, formalism_counts: Counter, metric_counts: Counter) -> None:
    lines = [
        f"# FINAL Metric Evidence Inspection: {summary['label']}",
        "",
        "## Selected FINAL CSV",
        "",
        "```text",
        str(summary["selected_final_csv"]),
        "```",
        "",
        "## Size / Shape",
        "",
        "```text",
        f"uncompressed_size_mb = {summary['selected_final_uncompressed_size_mb']}",
        f"row_count = {summary['row_count']}",
        f"numeric_primary_count = {summary['numeric_primary_count']}",
        f"blank_primary_count = {summary['blank_primary_count']}",
        f"unique_source_files = {summary['unique_source_files']}",
        f"unique_metrics = {summary['unique_metrics']}",
        "```",
        "",
        "## Top Evidence Types",
        "",
        "```text",
    ]
    for key, count in evidence_counts.most_common(20):
        lines.append(f"{key}: {count}")
    lines += ["```", "", "## Top Source Files", "", "```text"]
    for key, count in source_counts.most_common(30):
        lines.append(f"{key}: {count}")
    lines += ["```", "", "## Top Formalism Classes", "", "```text"]
    for key, count in formalism_counts.most_common(30):
        lines.append(f"{key}: {count}")
    lines += ["```", "", "## Top Metrics", "", "```text"]
    for key, count in metric_counts.most_common(40):
        lines.append(f"{key}: {count}")
    lines += [
        "```",
        "",
        "## Output Tables",
        "",
        "```text",
        "final_evidence_type_counts.csv",
        "final_source_file_counts.csv",
        "final_formalism_counts.csv",
        "final_metric_counts.csv",
        "final_source_metric_counts.csv",
        "final_metric_context_counts.csv",
        "final_top_abs_primary_values.csv",
        "final_focus_top_abs_rows.csv",
        "final_grade4_top_abs_rows.csv",
        "final_sae_top_abs_rows.csv",
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect huge FINAL_* metric-lab CSVs from zips/directories.")
    parser.add_argument("--input", action="append", required=True, help="Input as label=path or path. Can be repeated.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument("--top-n", type=int, default=300, help="Rows retained for top absolute-value tables.")
    args = parser.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for item in args.input:
        label, path = parse_input_arg(item)
        print(f"Inspecting {label}: {path}")
        summaries.append(inspect_one(label, path, out_dir, top_n=int(args.top_n)))
    write_rows(out_dir / "combined_final_inspection_summary.csv", summaries)
    print(f"Wrote FINAL inspection: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
