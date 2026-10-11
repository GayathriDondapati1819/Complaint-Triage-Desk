"""Step 5: try a few settings and models, compare on DEV only.

    python3 src/improve.py

Writes outputs/experiments.csv. Pick the best, then run  python3 src/train.py --final  once.
(train.py uses logistic regression C=3. If another setting clearly wins, change it there.)
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from config import DATA, OUTPUTS


def vec(**kw):
    params = dict(ngram_range=(1, 2), min_df=5, sublinear_tf=True, max_features=200_000)
    params.update(kw)
    return TfidfVectorizer(**params)


def main():
    train = pd.read_parquet(DATA / "train.parquet")
    dev = pd.read_parquet(DATA / "dev.parquet")

    experiments = {
        "logreg C=0.5": make_pipeline(vec(), LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5)),
        "logreg C=3": make_pipeline(vec(), LogisticRegression(max_iter=1000, class_weight="balanced", C=3)),
        "logreg C=10": make_pipeline(vec(), LogisticRegression(max_iter=1000, class_weight="balanced", C=10)),
        "linearsvc C=0.3": make_pipeline(vec(), LinearSVC(class_weight="balanced", C=0.3)),
        "linearsvc C=1": make_pipeline(vec(), LinearSVC(class_weight="balanced", C=1)),
        "linearsvc 1-3gram": make_pipeline(vec(ngram_range=(1, 3), min_df=3), LinearSVC(class_weight="balanced", C=0.3)),
    }

    rows = []
    for name, model in experiments.items():
        model.fit(train["clean"], train["label"])
        score = f1_score(dev["label"], model.predict(dev["clean"]), average="macro")
        rows.append({"model": name, "dev_macro_f1": round(float(score), 4)})
        print(f"{name:22s} macro-F1 = {score:.4f}")

    out = pd.DataFrame(rows).sort_values("dev_macro_f1", ascending=False)
    OUTPUTS.mkdir(exist_ok=True)
    out.to_csv(OUTPUTS / "experiments.csv", index=False)
    print("\nBest on dev:", out.iloc[0].to_dict())
    print("If the gap to logistic regression is under ~0.005, keep logistic regression: it gives "
          "probabilities and keywords for the demo.")


if __name__ == "__main__":
    main()
