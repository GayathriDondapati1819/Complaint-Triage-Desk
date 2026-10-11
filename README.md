# Complaint Triage Desk

Routes consumer financial complaints to the right product team, shows a confidence score and the words
behind each decision, and flags unusual jumps in complaint volume.

MSDS 630 Natural Language Processing, Residency NLP Hackathon (Project 2).
University of the Cumberlands, Dallas campus, October 9-11, 2026.

## The problem

A consumer-protection office receives thousands of free-text complaints each week. Staff read each one and
forward it to the right team by hand, and emerging problems are noticed weeks late. This project automates
the routing and surfaces trends early.

## What it does

- **Routing:** paste a complaint and get the predicted team, a confidence score, the top keywords, and a
  "needs human review" warning when the model is unsure.
- **Hardship flag:** complaints mentioning things like eviction or lost income are flagged as possibly urgent
  (a simple word list, not machine learning).
- **Redactions:** CFPB redaction tokens (XXXX) and dollar amounts are masked before classification.
- **Model performance:** per-category precision and recall, a confusion matrix, and the most confused pairs.
- **Trends and spike alerts:** complaint volume by product and month, and an alert when a month is unusually high.

## Setup

Requires Python 3.10 or newer.

```bash
git clone https://github.com/GayathriDondapati1819/Complaint-Triage-Desk.git
cd Complaint-Triage-Desk
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python3 -m pip install -r requirements.txt
```

## Data

The data is not stored in this repository. Download the CFPB Consumer Complaint Database CSV from
https://www.consumerfinance.gov/data-research/consumer-complaints/ and save it as:

```
data/raw/complaints.csv
```

- The file used here has **no consumer narrative column**, so the classifier reads the **Issue + Sub-issue**
  phrases of each complaint (for example "Incorrect information on your report. Information belongs to
  someone else"). This is set by `TEXT_MODE = "issue"` in `src/config.py`.
- Date file used: (fill in the download date and the date range it covers)
- The CFPB renamed its products over the years, so `LABEL_MAP` in `src/config.py` merges old and new names
  into a few clean categories (for example three credit-reporting names become "Credit reporting").
- Each category is capped at 5,000 complaints so classes are balanced and training is fast. The file is
  read in chunks, so it does not need to fit in memory.
- The same file supplies `Date received` and `Product` for the trends tab. Those charts count every
  complaint (not the capped sample), so volumes are real.

**Important limitation.** Issue and Sub-issue phrases come from a fixed list for each product, so predicting
the product from them is partly a lookup table. We therefore also score an exact-text lookup baseline with no
learning. The gap between that baseline and the model is the honest measure of what the model adds.

## Run it, in this order (from the repository root)

```bash
python3 src/data.py          # 1. read the CSV, label, de-duplicate, balance   -> data/sample.parquet
python3 src/split.py         # 2. clean + one fixed train/dev/test split       -> data/*.parquet
python3 src/train.py         # 3. train models and save metrics                -> models/
python3 src/evaluate.py      # 4. confusion matrix + mistakes to read          -> outputs/
python3 src/improve.py       # 5. (optional) compare settings on dev only      -> outputs/experiments.csv
python3 src/trends_data.py   # 6. real monthly counts for the trends tab       -> data/trends.parquet
python3 src/train.py --final      # 7. FINAL numbers on the test set. Run this once.
python3 src/evaluate.py --final   #    final confusion matrix for the slides
streamlit run app/app.py     # 8. the demo
```

The demo works offline once the model is trained.

## Folder structure

```
Complaint-Triage-Desk/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── config.py        settings: paths, label map, text mode, caps
│   ├── clean.py         text cleaning (handles XXXX redactions)
│   ├── data.py          build the balanced sample from the raw CSV
│   ├── split.py         one fixed train / dev / test split
│   ├── train.py         baselines + final model + metrics
│   ├── evaluate.py      confusion matrix + error examples
│   ├── improve.py       compare settings on dev
│   └── trends_data.py   real monthly counts per category from the CFPB file
├── app/
│   └── app.py           Streamlit demo
├── data/raw/            downloaded CSV files go here (not committed)
├── models/              trained model and metrics (generated, not committed)
├── outputs/             confusion_matrix.png, error_examples.csv, experiments.csv
└── docs/                one-page summary, presentation outline, explainer and notes
```

## Method

Text is lowercased, redaction tokens (XXXX) and amounts are replaced by placeholder tokens, and the text is
turned into numbers with TF-IDF (single words and word pairs). A logistic regression classifier with class
weighting predicts the category. We compare it with Naive Bayes and with a no-learning exact-text lookup.
The data is split once into train (70%), dev (15%) and test (15%), stratified by category. We tune on dev and
use the test set once. We report macro-F1 (every category counts equally), per-category precision and
recall, and a confusion matrix, and we read the mistakes to understand why they happen.


Most confused category pairs and why: (fill in from `outputs/error_examples.csv`)

## Limitations

- Some complaints are genuinely ambiguous between categories, so some mix-ups are label overlap, not model error.
- Categories are capped at 5,000 complaints for training, so the sample is balanced, not representative of real volumes.
- The text is a short Issue + Sub-issue phrase, not the consumer's own words, so results do not show how the model would handle free-text complaints.
- English only. Very short or empty text is sent to human review.
- The hardship flag is a keyword list, not a trained model.

## Reproducing our results

Download the same files into `data/raw/`, then run the commands above. The random seed (`SEED` in
`src/config.py`) is fixed, so the sample and split are repeatable for the same files.
