"""Step 3: train the baselines and the final model, save everything the app needs.

    python3 src/train.py            # evaluates on DEV   (use while developing)
    python3 src/train.py --final    # evaluates on TEST  (run ONCE, at the very end)

Writes models/triage.joblib and models/metrics.json
"""
import argparse
import json

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

from config import DATA, MODELS


def make_vec():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=5, sublinear_tf=True, max_features=200_000)


def lookup_baseline(train, evalset):
    """No learning at all: for each exact text, predict the label it had most often in train.
    If this scores close to the real model, the task is mostly a lookup table."""
    majority = train.groupby("clean")["label"].agg(lambda s: s.value_counts().index[0])
    fallback = train["label"].value_counts().index[0]
    pred = evalset["clean"].map(majority).fillna(fallback)
    return f1_score(evalset["label"], pred, average="macro")


def to_jsonable(o):
    return o.item() if hasattr(o, "item") else str(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true", help="evaluate on the test split (do this once)")
    args = ap.parse_args()

    split = "test" if args.final else "dev"
    with open(DATA / "meta.json") as f:
        mode = json.load(f)["text_mode"]
    train = pd.read_parquet(DATA / "train.parquet")
    evalset = pd.read_parquet(DATA / f"{split}.parquet")
    print(f"Text mode: {mode}. Training on {len(train):,}, evaluating on {split} ({len(evalset):,})")

    nb = make_pipeline(make_vec(), MultinomialNB(alpha=0.1))
    lr = make_pipeline(make_vec(), LogisticRegression(max_iter=1000, class_weight="balanced", C=3))

    comparison = {"Exact-text lookup (no learning)": round(float(lookup_baseline(train, evalset)), 4)}
    for name, model in [("Naive Bayes (baseline)", nb), ("Logistic regression (final)", lr)]:
        model.fit(train["clean"], train["label"])
        pred = model.predict(evalset["clean"])
        comparison[name] = round(float(f1_score(evalset["label"], pred, average="macro")), 4)
    for k, v in comparison.items():
        print(f"{k:34s} macro-F1 = {v:.4f}")

    pred = lr.predict(evalset["clean"])
    labels = sorted(train["label"].unique())
    report = classification_report(evalset["label"], pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(evalset["label"], pred, labels=labels)

    MODELS.mkdir(exist_ok=True)
    joblib.dump(lr, MODELS / "triage.joblib")
    metrics = {"split": split, "text_mode": mode, "n_train": int(len(train)), "n_eval": int(len(evalset)),
               "labels": labels, "report": report, "confusion_matrix": cm.tolist(), "comparison": comparison}
    with open(MODELS / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=to_jsonable)
    print("Saved models/triage.joblib and models/metrics.json")


if __name__ == "__main__":
    main()
