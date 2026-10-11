"""Step 2: clean the text and make ONE fixed train / dev / test split.

Run:  python3 src/split.py
Reads data/sample.parquet; writes data/train.parquet, dev.parquet, test.parquet.
Rules: tune on dev, touch test only once at the very end, fit vectorizers on train only.
"""
import json

import pandas as pd
from sklearn.model_selection import train_test_split

from clean import clean
from config import DATA, MIN_WORDS, SEED


def main():
    with open(DATA / "meta.json") as f:
        mode = json.load(f)["text_mode"]
    df = pd.read_parquet(DATA / "sample.parquet")
    df["clean"] = df["text"].apply(clean)
    df = df[df["clean"].str.split().str.len() >= MIN_WORDS[mode]].reset_index(drop=True)

    train, temp = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=SEED)
    dev, test = train_test_split(temp, test_size=0.50, stratify=temp["label"], random_state=SEED)

    for name, part in [("train", train), ("dev", dev), ("test", test)]:
        part.to_parquet(DATA / f"{name}.parquet", index=False)
        print(f"{name}: {len(part):,} rows")
        print(part["label"].value_counts().to_string(), "\n")

    print("Three cleaned examples:")
    print(train[["label", "clean"]].sample(3, random_state=1).to_string(max_colwidth=120))


if __name__ == "__main__":
    main()
