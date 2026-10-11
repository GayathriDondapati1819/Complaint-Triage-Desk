"""Complaint Triage Desk - Streamlit demo.

Run from the project folder:
    streamlit run app/app.py

Needs (created by the pipeline):
    models/triage.joblib, models/metrics.json   <- python3 src/train.py
    data/sample.parquet, data/meta.json         <- python3 src/data.py
"""
import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from clean import clean  # noqa: E402

MODEL_PATH = ROOT / "models" / "triage.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
SAMPLE_PATH = ROOT / "data" / "sample.parquet"
META_PATH = ROOT / "data" / "meta.json"
TRENDS_PATH = ROOT / "data" / "trends.parquet"

st.set_page_config(page_title="Complaint Triage Desk", page_icon="📬", layout="wide")

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
TEAMS = {
    "Credit reporting": "Credit Reporting Team",
    "Debt collection": "Debt Collection Team",
    "Bank account": "Deposits & Banking Team",
    "Credit/prepaid card": "Cards Team",
    "Student loan": "Student Lending Team",
    "Vehicle loan": "Auto Lending Team",
    "Mortgage": "Mortgage Team",
    "Payday/personal loan": "Short-Term Lending Team",
    "Money transfer": "Payments Team",
}

URGENT_PATTERNS = [
    r"evict\w*", r"foreclos\w*", r"homeless\w*", r"repossess\w*", r"garnish\w*",
    r"bankrupt\w*", r"lost (my|his|her) job", r"laid off", r"unemploy\w*",
    r"can'?t (afford|pay)", r"cannot (afford|pay)", r"unable to (afford|pay)",
    r"no (money|income)", r"shut ?off", r"medical (bills|emergency)",
    r"behind on (rent|payments|my mortgage)", r"financial hardship", r"struggling to pay",
]

EXAMPLES_NARRATIVE = {
    "Credit reporting": (
        "There is an account on my credit report that does not belong to me. I disputed it with the "
        "bureau twice and each time they said it was verified without investigating. I never opened "
        "this account and it is dragging down my score."
    ),
    "Debt collection": (
        "A collector keeps calling me at work about a debt of {$1,250.00} that I do not owe. I asked "
        "for written validation and never received it. They called XXXX times last week and "
        "threatened to garnish my wages."
    ),
    "Mortgage": (
        "My servicer misapplied my last three payments and now says I am behind. I am afraid of "
        "foreclosure and nobody will explain my escrow balance. I lost my job in XX/XX/XXXX and "
        "cannot afford the new payment."
    ),
    "Messy / low information": "XXXX XXXX please help with my XXXX",
}
EXAMPLES_ISSUE = {
    "Credit reporting": "Incorrect information on your report. Information belongs to someone else",
    "Debt collection": "Attempts to collect debt not owed. Debt is not yours",
    "Mortgage": "Struggling to pay mortgage",
    "Messy / low information": "XXXX please help",
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def read_text_mode():
    try:
        with open(META_PATH) as f:
            return json.load(f)["text_mode"]
    except Exception:
        return "narrative"


MODE = read_text_mode()
EXAMPLES = EXAMPLES_NARRATIVE if MODE == "narrative" else EXAMPLES_ISSUE
MIN_WORDS_UI = 5 if MODE == "narrative" else 2


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    with open(METRICS_PATH) as f:
        return json.load(f)


@st.cache_data
def load_monthly():
    """Monthly counts per category. Prefers real counts from the CFPB metadata file
    (src/trends_data.py); otherwise falls back to the dates in the training sample, if any."""
    if TRENDS_PATH.exists():
        m = pd.read_parquet(TRENDS_PATH)
        m["month"] = pd.to_datetime(m["month"])
        return m, "cfpb"
    if SAMPLE_PATH.exists():
        df = pd.read_parquet(SAMPLE_PATH)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        if not df.empty:
            df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
            return df.groupby(["month", "label"]).size().reset_index(name="complaints"), "sample"
    return None, None


def top_keywords(vec, clf, x, class_idx, k=8):
    """Words in this text that pushed the model toward the predicted class."""
    x = x.tocsr()
    coefs = clf.coef_[class_idx]
    contrib = x.data * coefs[x.indices]
    names = vec.get_feature_names_out()
    order = np.argsort(contrib)[::-1][:k]
    return [(names[x.indices[i]], float(contrib[i])) for i in order if contrib[i] > 0]


def find_urgency(raw_text):
    low = raw_text.lower()
    return sorted({m.group(0) for pat in URGENT_PATTERNS if (m := re.search(pat, low))})


def set_example(name):
    st.session_state["complaint_text"] = EXAMPLES[name]


def find_spikes(series, window, k):
    """Flag months above rolling mean + k * std of the previous `window` months."""
    past = series.shift(1).rolling(window, min_periods=window)
    threshold = past.mean() + k * past.std()
    return threshold, series > threshold


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
st.title("📬 Complaint Triage Desk")
st.caption("Routes consumer financial complaints to the right team and flags emerging problems.")
if MODE == "issue":
    st.info("This version reads the complaint's **Issue and Sub-issue** phrases, because the data "
            "available has no free-text narratives. Paste a phrase such as the examples below.")

tab_route, tab_perf, tab_trends = st.tabs(["Route a complaint", "Model performance", "Trends & spike alerts"])

# ------------------------------- Tab 1: routing ----------------------------- #
with tab_route:
    if not MODEL_PATH.exists():
        st.error("No trained model found. Run `python3 src/train.py` from the project folder first.")
    else:
        model = load_model()
        vec, clf = model.steps[0][1], model.steps[-1][1]

        left, right = st.columns([3, 2], gap="large")

        with left:
            st.subheader("Paste a complaint" if MODE == "narrative" else "Enter issue text")
            st.text_area("Complaint text", key="complaint_text", height=200,
                         placeholder="Paste the complaint here..." if MODE == "narrative"
                         else "Example: Incorrect information on your report. Information belongs to someone else",
                         label_visibility="collapsed")
            st.write("Try an example:")
            cols = st.columns(len(EXAMPLES))
            for col, name in zip(cols, EXAMPLES):
                col.button(name, on_click=set_example, args=(name,), width="stretch")
            threshold = st.slider("Send to human review below this confidence", 0.10, 0.90, 0.50, 0.05)
            go_btn = st.button("Route complaint", type="primary")

        with right:
            st.subheader("Result")
            raw = st.session_state.get("complaint_text", "") or ""

            if not go_btn:
                st.info("Enter a complaint and press **Route complaint**.")
            elif not raw.strip():
                st.warning("The box is empty. Enter a complaint first.")
            else:
                n_redacted = len(re.findall(r"x{2,}", raw.lower()))
                n_words = len(raw.split())
                letters = [c for c in raw if c.isalpha()]
                non_ascii = (sum(ord(c) > 127 for c in letters) / len(letters)) if letters else 0

                if n_words < MIN_WORDS_UI:
                    st.warning("That is too short to route reliably. Please enter a little more text.")
                else:
                    if non_ascii > 0.3:
                        st.warning("This looks like non-English text. The model was trained on English.")
                    cleaned = clean(raw)
                    x = vec.transform([cleaned])
                    if x.nnz == 0:
                        st.warning("None of the words here are known to the model. Send it to a human reviewer.")
                    else:
                        proba = clf.predict_proba(x)[0]
                        order = np.argsort(proba)[::-1]
                        best, conf = clf.classes_[order[0]], float(proba[order[0]])
                        second, conf2 = clf.classes_[order[1]], float(proba[order[1]])

                        if conf < threshold:
                            st.error("⚠️ Needs human review")
                            st.write(f"Top guesses: **{best}** ({conf:.0%}) or **{second}** ({conf2:.0%}).")
                        else:
                            st.success(f"Route to: **{TEAMS.get(best, best + ' Team')}**")
                            m1, m2 = st.columns(2)
                            m1.metric("Predicted category", best)
                            m2.metric("Confidence", f"{conf:.0%}")
                            st.caption(f"Runner-up: {second} ({conf2:.0%})")

                        urgent = find_urgency(raw)
                        if urgent:
                            st.error("🚨 Possible financial hardship: " + ", ".join(f"“{u}”" for u in urgent)
                                     + ". Consider prioritising this complaint.")

                        kws = top_keywords(vec, clf, x, int(order[0]))
                        if kws:
                            st.write("**Top keywords behind this decision**")
                            kdf = pd.DataFrame(kws, columns=["keyword", "weight"]).iloc[::-1]
                            fig = px.bar(kdf, x="weight", y="keyword", orientation="h", height=260)
                            fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), yaxis_title=None)
                            st.plotly_chart(fig, width="stretch")

                        with st.expander("All category probabilities"):
                            pdf = pd.DataFrame({"category": clf.classes_, "probability": proba})
                            st.dataframe(pdf.sort_values("probability", ascending=False)
                                         .style.format({"probability": "{:.1%}"}), hide_index=True)

                        if n_redacted:
                            st.caption(f"ℹ️ {n_redacted} redacted token(s) (XXXX) were masked before classification.")

# ------------------------------ Tab 2: performance -------------------------- #
with tab_perf:
    if not METRICS_PATH.exists():
        st.info("Run `python3 src/train.py` to generate evaluation metrics.")
    else:
        mt = load_metrics()
        report, labels = mt["report"], mt["labels"]
        st.subheader("Classifier evaluation")
        st.caption(f"Evaluated on the **{mt['split']}** split ({mt['n_eval']:,} complaints); "
                   f"trained on {mt['n_train']:,}. Text mode: {mt.get('text_mode', 'narrative')}.")

        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Macro-F1", f"{report['macro avg']['f1-score']:.3f}")
            st.metric("Accuracy", f"{report['accuracy']:.3f}")
            comp = pd.DataFrame({"Model": list(mt["comparison"].keys()),
                                 "Macro-F1": list(mt["comparison"].values())})
            st.write("**Baselines vs final**")
            st.dataframe(comp, hide_index=True)
            st.caption("The exact-text lookup does no learning. If the real model barely beats it, "
                       "the task is mostly a lookup table.")
        with c2:
            rows = [{"Category": l, "Precision": report[l]["precision"], "Recall": report[l]["recall"],
                     "F1": report[l]["f1-score"], "Support": int(report[l]["support"])} for l in labels]
            tbl = pd.DataFrame(rows)
            st.write("**Per-class precision and recall**")
            st.dataframe(tbl.style.format({"Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}"}),
                         hide_index=True, width="stretch")

        cm = np.array(mt["confusion_matrix"])
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        st.write("**Confusion matrix** (each row sums to 1: of all true complaints in a category, "
                 "where did they go?)")
        fig = px.imshow(cm_norm, x=labels, y=labels, text_auto=".2f", color_continuous_scale="Blues",
                        labels=dict(x="Predicted", y="True", color="Share"), height=560, aspect="auto")
        st.plotly_chart(fig, width="stretch")

        pairs = [(labels[i], labels[j], int(cm[i, j])) for i in range(len(labels))
                 for j in range(len(labels)) if i != j and cm[i, j] > 0]
        pairs.sort(key=lambda p: p[2], reverse=True)
        st.write("**Most confused category pairs**")
        st.dataframe(pd.DataFrame(pairs[:8], columns=["True category", "Predicted as", "Count"]),
                     hide_index=True)

# ------------------------------- Tab 3: trends ------------------------------ #
with tab_trends:
    monthly_all, source = load_monthly()
    if monthly_all is None:
        st.info("No dated data found. Run `python3 src/trends_data.py` to build real monthly counts "
                "from the CFPB file.")
    else:
        st.subheader("Complaint volume by product and month")
        if source == "cfpb":
            st.caption("Real monthly counts from the full CFPB file, grouped into the same categories as the classifier.")
        else:
            st.caption("⚠️ Counts come from the **sampled** training data, where each category was capped, so "
                       "absolute volumes are not real. Month-to-month shape is more reliable.")

        cats = sorted(monthly_all["label"].unique())
        chosen = st.multiselect("Products", cats, default=cats)
        view = monthly_all[monthly_all["label"].isin(chosen)]

        st.plotly_chart(px.line(view, x="month", y="complaints", color="label", markers=True, height=380),
                        width="stretch")
        totals = view.groupby("label", as_index=False)["complaints"].sum()
        st.plotly_chart(px.bar(totals, x="label", y="complaints", height=300), width="stretch")

        st.divider()
        st.subheader("🔔 Spike alerts")
        st.caption("A month is flagged when its volume exceeds the mean of the previous N months "
                   "plus K standard deviations.")
        a, b, c = st.columns(3)
        product = a.selectbox("Product to monitor", cats)
        window = b.slider("Look-back window (months)", 3, 12, 6)
        kstd = c.slider("K (standard deviations)", 1.0, 4.0, 2.0, 0.5)

        s_ = (monthly_all[monthly_all["label"] == product].set_index("month")["complaints"]
              .sort_index().asfreq("MS", fill_value=0))
        if len(s_) <= window:
            st.info("Not enough months of data for this look-back window. Try a smaller window.")
        else:
            thr, flags = find_spikes(s_, window, kstd)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=s_.index, y=s_.values, mode="lines+markers", name="Complaints"))
            fig.add_trace(go.Scatter(x=thr.index, y=thr.values, mode="lines", name="Alert threshold",
                                     line=dict(dash="dash")))
            fig.add_trace(go.Scatter(x=s_.index[flags], y=s_[flags], mode="markers", name="Spike",
                                     marker=dict(size=14, symbol="x", color="red")))
            fig.update_layout(height=380, margin=dict(t=20))
            st.plotly_chart(fig, width="stretch")
            if flags.any():
                st.error("Spikes detected in: " + ", ".join(d.strftime("%b %Y") for d in s_.index[flags]))
            else:
                st.success("No spikes detected with these settings.")
