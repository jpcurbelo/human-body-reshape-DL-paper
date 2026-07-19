"""Batch-generate 3D avatars from centroid measurements.

This script reads web_codes/isobody_code_centroids_v2_jesus.csv and, for each
entry code, generates both a female and a male avatar.

Outputs are written under web_codes/generated_avatars/<gender>/<code>/.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from reshaper.avatar import Avatar
from utils import MEASUREMENTS, M_NUM


INPUT_CSV = Path(__file__).with_name("isobody_code_centroids_v2_jesus.csv")
OUTPUT_ROOT = Path(__file__).with_name("generated_avatars")
GENDERS = ("female", "male")

MEASUREMENT_INDEX = {
    "weight_kg": 0,
    "stature_cm": 1,
    "chest_girth": 3,
    "waist_girth": 4,
    "hips_buttock_girth": 5,
    "shoulder_girth": 6,
    "sleeveoutseam_length": 14,
    "forearm_length": 15,
    "crotchheight_length": 16,
    "waistback_length": 17,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate female and male avatars from centroid measurement codes."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=INPUT_CSV,
        help="Path to the centroid CSV file.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Root folder where generated avatars are written.",
    )
    return parser.parse_args()


def build_measurement_vector(row: pd.Series) -> np.ndarray:
    measurements = np.zeros(M_NUM, dtype=np.float64)
    for column_name, measurement_index in MEASUREMENT_INDEX.items():
        value = row[column_name]
        if pd.notna(value):
            measurements[measurement_index] = float(value)
    return measurements


def generate_avatar(code: str, gender: str, measurements: np.ndarray, output_root: Path) -> None:
    gender_dir = output_root / gender / code
    gender_dir.mkdir(parents=True, exist_ok=True)

    avatar = Avatar(measurements, gender)
    avatar.predict()
    avatar.create_obj_file(ava_dir=str(gender_dir), ava_name=code)
    avatar.measure(out_meas_name=f"{code}_measurements", out_dir=str(gender_dir))


def main() -> None:
    args = parse_args()

    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    centroid_df = pd.read_csv(args.input_csv)
    required_columns = {"code", *MEASUREMENT_INDEX.keys()}
    missing_columns = required_columns.difference(centroid_df.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"CSV is missing required columns: {missing_list}")

    args.output_root.mkdir(parents=True, exist_ok=True)

    for _, row in centroid_df.iterrows():
        code = str(row["code"]).strip()
        if not code:
            continue

        base_measurements = build_measurement_vector(row)
        for gender in GENDERS:
            print(f"Generating {gender} avatar for {code}")
            generate_avatar(code, gender, base_measurements, args.output_root)


if __name__ == "__main__":
    main()