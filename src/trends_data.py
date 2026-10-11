"""Optional: build REAL monthly complaint counts for the trends tab.

The training sample is capped per category, so its counts are not real volumes. The full CFPB file
has a date and a product for every complaint, so we count those for trends and spike alerts.

It reads the same data/raw/complaints.csv used for training:   python3 src/trends_data.py
Writes data/trends.parquet (month, label, complaints).
"""
import sys

import pandas as pd

from config import CFPB_META_CSV, DATA, LABEL_MAP, TRENDS_PARQUET
from data import find_col, read_header

CHUNK = 500_000


def to_dates(col):
    """Parse dates quickly. Handles 2024-09-03, 09/03/24, 09/03/2024 and Excel serial numbers."""
    nums = pd.to_numeric(col, errors="coerce")
    if nums.notna().mean() > 0.9:                      # e.g. 45828 saved by Excel
        return pd.to_datetime(nums, unit="D", origin="1899-12-30", errors="coerce")
    for fmt in ("%Y-%m-%d", "%m/%d/%y", "%m/%d/%Y"):
        d = pd.to_datetime(col, format=fmt, errors="coerce")
        if d.notna().mean() > 0.9:
            return d
    return pd.to_datetime(col, errors="coerce", format="mixed")


def main():
    if not CFPB_META_CSV.exists():
        sys.exit(f"Cannot find {CFPB_META_CSV}. Save the CFPB CSV (dates + products) there first.")
    header, enc = read_header(CFPB_META_CSV)
    c_date = find_col(header, "Date received", "date_received")
    c_prod = find_col(header, "Product", "product")
    if not (c_date and c_prod):
        sys.exit(f"Need a date and a product column. Found: {header}")

    parts = []
    for chunk in pd.read_csv(CFPB_META_CSV, usecols=[c_date, c_prod], chunksize=CHUNK, encoding=enc,
                             low_memory=False):
        chunk["label"] = chunk[c_prod].map(LABEL_MAP)
        chunk["month"] = to_dates(chunk[c_date]).dt.to_period("M").dt.to_timestamp()
        chunk = chunk.dropna(subset=["label", "month"])
        parts.append(chunk.groupby(["month", "label"]).size().reset_index(name="complaints"))

    if not parts:
        sys.exit("No rows could be read.")
    out = (pd.concat(parts).groupby(["month", "label"], as_index=False)["complaints"].sum()
           .sort_values(["label", "month"]))
    DATA.mkdir(exist_ok=True)
    out.to_parquet(TRENDS_PARQUET, index=False)
    print(f"Saved {TRENDS_PARQUET}  {out.shape}")
    print("Months covered:", out["month"].min().date(), "to", out["month"].max().date())
    print(out.groupby("label")["complaints"].sum().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
