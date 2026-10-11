"""Step 1: read the CFPB CSV in chunks and build a balanced, labelled sample.

Run from the project folder:   python3 src/data.py
Reads:   data/raw/complaints.csv
Writes:  data/sample.parquet, data/meta.json
"""
import json
import sys

import numpy as np
import pandas as pd

from config import (CAP_PER_LABEL, DATA, LABEL_MAP, LABEL_MODE, MIN_CLASS_ROWS, MIN_WORDS, RAW_CSV,
                    SEED, TEXT_MODE)

CHUNK = 200_000


def norm(s):
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def find_col(columns, *candidates):
    lookup = {norm(c): c for c in columns}
    for cand in candidates:
        if norm(cand) in lookup:
            return lookup[norm(cand)]
    return None


def read_header(path):
    with open(path, "rb") as f:
        magic = f.read(4)
    if magic[:2] == b"PK":
        sys.exit("This file is a zip or an Excel workbook, not a plain CSV. Unzip it, and do not "
                 "open and re-save it in Excel.")
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(path, nrows=0, encoding=enc).columns.tolist(), enc
        except UnicodeDecodeError:
            continue
    sys.exit("Could not read the file's text encoding.")


def choose_label_mode(path, enc, c_prod):
    if LABEL_MODE in ("map", "as_is"):
        return LABEL_MODE
    sample = pd.read_csv(path, usecols=[c_prod], nrows=100_000, encoding=enc, low_memory=False)[c_prod]
    return "map" if sample.isin(list(LABEL_MAP)).mean() >= 0.2 else "as_is"


def choose_mode(path, enc, c_narr):
    if TEXT_MODE == "issue":
        return "issue"
    if TEXT_MODE == "narrative":
        if not c_narr:
            sys.exit("TEXT_MODE is 'narrative' but the file has no narrative column.")
        return "narrative"
    if not c_narr:
        return "issue"
    found = 0
    for i, chunk in enumerate(pd.read_csv(path, usecols=[c_narr], chunksize=CHUNK, encoding=enc,
                                          low_memory=False)):
        found += chunk[c_narr].notna().sum()
        if found >= 50 or i >= 4:
            break
    return "narrative" if found >= 50 else "issue"


def main():
    if not RAW_CSV.exists():
        sys.exit(f"Cannot find {RAW_CSV}. Put the downloaded CSV there and run again.")

    header, enc = read_header(RAW_CSV)
    print("Columns in your file:", header)
    c_date = find_col(header, "Date received", "date_received")
    c_prod = find_col(header, "Product", "product")
    c_issue = find_col(header, "Issue", "issue")
    c_sub = find_col(header, "Sub-issue", "sub_issue")
    c_narr = find_col(header, "Consumer complaint narrative", "consumer_complaint_narrative",
                      "complaint_what_happened", "narrative", "complaint_text")
    if not c_prod:
        sys.exit("No Product column found, so there is nothing to predict.")

    mode = choose_mode(RAW_CSV, enc, c_narr)
    if mode == "issue" and not c_issue:
        sys.exit("No narrative and no Issue column found, so there is no text to classify.")
    print(f"\nText mode: {mode.upper()}  "
          f"({'complaint narratives' if mode == 'narrative' else 'Issue + Sub-issue phrases'})")

    label_mode = choose_label_mode(RAW_CSV, enc, c_prod)
    print(f"Label mode: {label_mode.upper()}  "
          f"({'merging CFPB product names with LABEL_MAP' if label_mode == 'map' else 'using the product names in your file'})")

    usecols = [c for c in [c_date, c_prod, c_issue, c_sub, c_narr] if c]
    rng = np.random.default_rng(SEED)
    kept = None
    seen = {}

    for chunk in pd.read_csv(RAW_CSV, usecols=usecols, chunksize=CHUNK, encoding=enc, low_memory=False):
        out = pd.DataFrame({
            "date": chunk[c_date] if c_date else pd.NaT,
            "product": chunk[c_prod],
        })
        if label_mode == "map":
            out["label"] = out["product"].map(LABEL_MAP)
        else:
            out["label"] = out["product"].astype(str).str.replace("_", " ").str.strip().str.capitalize()
        if mode == "narrative":
            out["text"] = chunk[c_narr]
        else:
            issue = chunk[c_issue].fillna("").astype(str).str.strip()
            sub = chunk[c_sub].fillna("").astype(str).str.strip() if c_sub else pd.Series("", index=chunk.index)
            out["text"] = issue.where(sub == "", issue + ". " + sub)

        out = out.dropna(subset=["label", "text"])
        out = out[out["text"].astype(str).str.count(" ") + 1 >= MIN_WORDS[mode]]   # fast word-count check
        if out.empty:
            continue
        for lab, n in out["label"].value_counts().items():
            seen[lab] = seen.get(lab, 0) + int(n)

        out = out.assign(_r=rng.random(len(out)))
        kept = out if kept is None else pd.concat([kept, out], ignore_index=True)
        if mode == "narrative":      # repeated complaints would leak between train and test
            kept = kept.drop_duplicates(subset=["text"])
        kept = kept.sort_values("_r").groupby("label", sort=False).head(CAP_PER_LABEL)

    if kept is None or kept.empty:
        sys.exit("No usable rows were found. Check the product names, and LABEL_MODE / LABEL_MAP in config.py.")

    df = kept.drop(columns="_r").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce", format="mixed")
    counts = df["label"].value_counts()
    small = counts[counts < MIN_CLASS_ROWS].index.tolist()
    if small:
        print("\nDropping categories with too few rows:", small)
        df = df[~df["label"].isin(small)].reset_index(drop=True)

    print("\nUsable rows per category before capping:")
    print(pd.Series(seen).sort_values(ascending=False).to_string())
    print("\nKept after capping:")
    print(df["label"].value_counts().to_string())

    words = df["text"].astype(str).str.split().str.len()
    print(f"\nMedian words per text: {words.median():.0f}")
    if mode == "narrative" and words.median() < 40:
        print("WARNING: these texts look too short to be narratives. Check your file.")

    DATA.mkdir(exist_ok=True)
    df.to_parquet(DATA / "sample.parquet", index=False)
    with open(DATA / "meta.json", "w") as f:
        json.dump({"text_mode": mode, "source_file": RAW_CSV.name, "rows": int(len(df)),
                   "labels": sorted(df["label"].unique().tolist())}, f, indent=2)
    print(f"\nSaved data/sample.parquet {df.shape} and data/meta.json")


if __name__ == "__main__":
    main()
