"""Simple color-coded Streamlit interface for sentiment predictions."""

from pathlib import Path

import joblib
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "sentiment_model.pkl"

st.set_page_config(
    page_title="Sentiment Analyst",
    page_icon="💬",
    layout="centered",
)
st.title("Sentiment Analyst")
st.caption("Paste a sentence or review to see the model's sentiment prediction.")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not MODEL_PATH.exists():
    st.warning(
        "Run the Colab notebook with a GITHUB_TOKEN secret, or start the Train sentiment model workflow from the repository’s Actions tab; then refresh this app.",
        title="Model not found",
    )
    st.stop()

model = load_model()
classifier = model.named_steps["classifier"]
classes = list(classifier.classes_)

with st.form("sentiment_form"):
    text = st.text_area(
        "Text to analyze",
        placeholder="Example: I really enjoyed the movie and would watch it again!",
        height=140,
        max_chars=5000,
    )
    submitted = st.form_submit_button("Analyze sentiment", type="primary", width="stretch")

if submitted:
    if not text.strip():
        st.info("Enter a little text first, then select Analyze sentiment.")
    else:
        prediction = model.predict([text])[0]
        probabilities = model.predict_proba([text])[0]
        confidence = float(probabilities[classes.index(prediction)])
        if prediction == "positive":
            st.success("Positive sentiment", title="Prediction")
        elif prediction == "negative":
            st.error("Negative sentiment", title="Prediction")
        else:
            st.warning("Neutral sentiment", title="Prediction")

        st.metric("Model confidence", f"{confidence:.0%}")
        st.progress(confidence, text="Confidence")
        with st.expander("See scores for each sentiment"):
            for label, score in sorted(zip(classes, probabilities), key=lambda item: item[1], reverse=True):
                st.write(f"**{label.title()}** — {score:.1%}")
                st.progress(float(score), text=label.title())

st.caption("A simple TF-IDF and Logistic Regression model. Its prediction is an estimate, not a guarantee.")

