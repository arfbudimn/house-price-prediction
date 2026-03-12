"""
Load & time-split the raw dataset.

- Production default writes to data/raw/
- Tests can pass a temp `output_dir` so nothing in data/ is touched.
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw")


def load_and_split_data(
    raw_path: str = "data/raw/HouseTS_origin.csv",
    output_dir: Path | str = DATA_DIR,
):
    """Load raw dataset, split into train/eval/holdout by date, and save to output_dir."""
    df = pd.read_csv(raw_path)

    # Ensure datetime + sort
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # Cutoffs
    cutoff_date_eval = pd.Timestamp("2020-01-01")     # eval starts
    cutoff_date_holdout = pd.Timestamp("2022-01-01")  # holdout starts

    # Splits
    df_train = df[df["date"] < cutoff_date_eval]
    df_eval = df[(df["date"] >= cutoff_date_eval) & (df["date"] < cutoff_date_holdout)]
    df_holdout = df[df["date"] >= cutoff_date_holdout]

    # Save
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(outdir / "train.csv", index=False)
    df_eval.to_csv(outdir / "eval.csv", index=False)
    df_holdout.to_csv(outdir / "holdout.csv", index=False)

    print(f"✅ Data split completed (saved to {outdir}).")
    print(f"   Train: {df_train.shape}, Eval: {df_eval.shape}, Holdout: {df_holdout.shape}")

    return df_train, df_eval, df_holdout


if __name__ == "__main__":
    load_and_split_data()
