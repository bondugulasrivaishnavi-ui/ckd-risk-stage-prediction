"""Download and clean the UCI Chronic Kidney Disease dataset.

Usage:
    python -m src.data_loader

Writes:
    data/raw/ckd_raw.csv         (as downloaded)
    data/processed/ckd_clean.csv (cleaned, numeric target: 1 = CKD, 0 = not CKD)
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import BINARY_MAP, NUMERIC, TARGET

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "ckd_raw.csv"
CLEAN_PATH = ROOT / "data" / "processed" / "ckd_clean.csv"

UCI_ID = 336  # Chronic Kidney Disease


def download_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Fetch the dataset from the UCI repository via ucimlrepo and save it."""
    from ucimlrepo import fetch_ucirepo

    ds = fetch_ucirepo(id=UCI_ID)
    df = pd.concat([ds.data.features, ds.data.targets], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    if not path.exists():
        return download_raw(path)
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix the known data-quality problems in the UCI CKD file.

    Known issues handled here:
      - stray whitespace / tabs in strings (e.g. '\\tno', ' yes', 'ckd\\t')
      - '?' and empty strings used for missing values
      - numeric columns stored as text
      - inconsistent labels ('notckd' vs 'ckd', 'no' vs '\\tno')
    """
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    # strip whitespace in every text column, then turn junk tokens into NaN
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype("string").str.strip().str.lower()
    df = df.replace({"?": pd.NA, "": pd.NA, "nan": pd.NA})

    # numeric columns -> float
    for col in NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    # binary categoricals -> 0/1, keeping NaN where missing
    for col, mapping in BINARY_MAP.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).astype("float")

    # a few known label typos
    if TARGET in df.columns:
        df[TARGET] = (
            df[TARGET]
            .replace({"ckd\t": "ckd", "notckd": "notckd"})
            .map({"ckd": 1, "notckd": 0})
        )
        df = df.dropna(subset=[TARGET])
        df[TARGET] = df[TARGET].astype(int)

    return df.reset_index(drop=True)


def load_clean(refresh: bool = False) -> pd.DataFrame:
    """Return the cleaned dataset, building it on first use."""
    if CLEAN_PATH.exists() and not refresh:
        return pd.read_csv(CLEAN_PATH)
    df = clean(load_raw())
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False)
    return df


def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column missing count and percentage, sorted high to low."""
    miss = df.isna().sum()
    rep = pd.DataFrame({"missing": miss, "pct": (miss / len(df) * 100).round(1)})
    return rep.sort_values("pct", ascending=False)


if __name__ == "__main__":
    data = load_clean(refresh=True)
    print(f"Saved {CLEAN_PATH}  shape={data.shape}")
    print("\nClass balance:\n", data[TARGET].value_counts())
    print("\nMissingness (top 10):\n", missingness_report(data).head(10))
