"""Step 4: error analysis. Makes the confusion-matrix picture and a file of mistakes to read.

    python3 src/evaluate.py            # on DEV
    python3 src/evaluate.py --final    # on TEST (final numbers for the slides)

Needs models/triage.joblib (run train.py first).
Writes outputs/confusion_matrix.png and outputs/error_examples.csv
"""
import argparse

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score

from config import DATA, MODELS, OUTPUTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args()
    split = "test" if args.final else "dev"

    model = joblib.load(MODELS / "triage.joblib")
    df = pd.read_parquet(DATA / f"{split}.parquet")
    pred = model.predict(df["clean"])

    print(f"Evaluating on {split} ({len(df):,} rows)\n")
    print(classification_report(df["label"], pred, digits=3, zero_division=0))
    print("macro-F1:", round(f1_score(df["label"], pred, average="macro"), 4))

    OUTPUTS.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay.from_predictions(df["label"], pred, normalize="true", xticks_rotation=45,
                                            ax=ax, values_format=".2f", cmap="Blues")
    ax.set_title(f"Confusion matrix ({split} set, rows sum to 1)")
    plt.tight_layout()
    plt.savefig(OUTPUTS / "confusion_matrix.png", dpi=150)

    err = df.assign(pred=pred)
    err = err[err["label"] != err["pred"]]
    pairs = err.groupby(["label", "pred"]).size().sort_values(ascending=False)
    print("\nMost confused pairs (true -> predicted):")
    print(pairs.head(8).to_string())

    rows = []
    for (true, wrong), _ in pairs.head(3).items():
        sub = err[(err["label"] == true) & (err["pred"] == wrong)]
        rows.append(sub.sample(min(10, len(sub)), random_state=1)[["label", "pred", "text"]])
    if rows:
        pd.concat(rows).to_csv(OUTPUTS / "error_examples.csv", index=False)
    print("\nSaved outputs/confusion_matrix.png and outputs/error_examples.csv")
    print("Now read error_examples.csv and write one sentence per confused pair on WHY it happens.")


if __name__ == "__main__":
    main()
