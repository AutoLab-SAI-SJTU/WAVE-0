#!/usr/bin/env python
"""Compute action/state quantiles from all local LeRobot Parquet rows.

Migrated from the numeric-only dataset statistics utility. No video decoding,
GPU, Hub access, or source dataset changes are required. Existing statistics
are preserved, with action/state quantiles added to a new output JSON file.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm


QUANTILES = (0.01, 0.10, 0.50, 0.90, 0.99)
FEATURES = ("action", "observation.state")


def compute_quantile_stats(root: Path, output: Path, temp_dir: Path | None = None) -> None:
    root, output = root.expanduser().resolve(), output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new file: {output}")
    stats_path = root / "meta" / "stats.json"
    if not stats_path.is_file():
        raise FileNotFoundError(stats_path)
    data_paths = sorted((root / "data").rglob("*.parquet"))
    if not data_paths:
        raise FileNotFoundError(f"No Parquet files found under {root / 'data'}")

    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    row_counts = [pq.ParquetFile(path).metadata.num_rows for path in data_paths]
    total_rows = sum(row_counts)
    if total_rows == 0:
        raise ValueError("Dataset contains no rows")
    first_nonempty = next(path for path, rows in zip(data_paths, row_counts, strict=True) if rows)
    print(f"Found {len(data_paths)} Parquet files with {total_rows:,} rows", flush=True)

    for feature in FEATURES:
        expected_count = int(np.asarray(stats[feature]["count"]).reshape(-1)[0])
        if expected_count != total_rows:
            raise ValueError(
                f"{feature}: stats count {expected_count:,} does not match Parquet rows {total_rows:,}; "
                "check that the full dataset has been downloaded"
            )
        batch = next(pq.ParquetFile(first_nonempty).iter_batches(batch_size=1, columns=[feature]))
        first_vector = batch.column(0)[0].as_py()
        if not isinstance(first_vector, list) or not first_vector:
            raise ValueError(f"{feature}: expected a non-empty numeric vector")
        dimension = len(first_vector)
        fd, mmap_name = tempfile.mkstemp(prefix="wave0_quantiles_", suffix=".f32", dir=temp_dir)
        os.close(fd)
        mmap_path = Path(mmap_name)
        values = None
        try:
            values = np.memmap(mmap_path, mode="w+", dtype=np.float32, shape=(total_rows, dimension))
            offset = 0
            for path in tqdm(data_paths, desc=f"Reading {feature}"):
                for batch in pq.ParquetFile(path).iter_batches(batch_size=131_072, columns=[feature]):
                    column = batch.column(0)
                    if column.null_count:
                        raise ValueError(f"{feature}: null vectors in {path}")
                    if hasattr(column, "value_lengths") and np.any(
                        column.value_lengths().to_numpy() != dimension
                    ):
                        raise ValueError(f"{feature}: inconsistent vector dimensions in {path}")
                    array = column.flatten().to_numpy(zero_copy_only=False).reshape(len(column), dimension)
                    if not np.isfinite(array).all():
                        raise ValueError(f"{feature}: non-finite values in {path}")
                    values[offset : offset + len(column)] = array
                    offset += len(column)
            if offset != total_rows:
                raise RuntimeError(f"{feature}: read {offset:,} rows, expected {total_rows:,}")
            values.flush()

            quantile_values = np.empty((len(QUANTILES), dimension), dtype=np.float32)
            for index in tqdm(range(dimension), desc=f"Quantiles {feature}"):
                quantile_values[:, index] = np.quantile(values[:, index], QUANTILES, method="linear")
            for quantile, row in zip(QUANTILES, quantile_values, strict=True):
                stats[feature][f"q{int(quantile * 100):02d}"] = row.tolist()
        finally:
            del values
            mmap_path.unlink(missing_ok=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(stats, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Saved {output}; original dataset statistics unchanged.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Local LeRobot dataset root")
    parser.add_argument(
        "--output", type=Path, required=True, help="New statistics JSON file; never overwritten"
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=None,
        help=(
            "Existing scratch directory for a float32 memmap "
            "(~340 MB per feature for RoboTwin; default: /tmp)"
        ),
    )
    args = parser.parse_args()
    compute_quantile_stats(args.root, args.output, args.temp_dir)


if __name__ == "__main__":
    main()
