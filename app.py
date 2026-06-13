import io
from typing import List, Dict

import pandas as pd
import streamlit as st
from transformers import pipeline


MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

EXAMPLE_TEXTS = [
    "I really enjoyed this product. It works exactly as expected.",
    "The service was slow and the experience was disappointing.",
    "The movie was okay, but I probably would not watch it again.",
    "Amazing support team and very easy to use.",
    "I am not happy with the quality of this item.",
]


st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="💬",
    layout="wide",
)


@st.cache_resource
def load_sentiment_pipeline():
    """
    Load the Hugging Face sentiment analysis pipeline once and cache it.

    The model will be downloaded the first time the app runs.
    """
    return pipeline(
        task="sentiment-analysis",
        model=MODEL_NAME,
    )


def normalize_label(label: str) -> str:
    """
    Convert model labels into user-friendly labels.
    """
    label = label.upper()

    if label == "POSITIVE":
        return "Positive"
    if label == "NEGATIVE":
        return "Negative"

    return label.title()


def predict_sentiment(text: str, classifier) -> Dict[str, object]:
    """
    Run sentiment prediction for a single text.
    """
    cleaned_text = text.strip()

    if not cleaned_text:
        return {
            "text": text,
            "label": "Empty input",
            "confidence": None,
        }

    prediction = classifier(cleaned_text)[0]

    return {
        "text": cleaned_text,
        "label": normalize_label(prediction["label"]),
        "confidence": round(float(prediction["score"]), 4),
    }


def predict_batch(texts: List[str], classifier) -> pd.DataFrame:
    """
    Run sentiment prediction for multiple texts.
    """
    results = []

    for text in texts:
        if text.strip():
            results.append(predict_sentiment(text, classifier))

    return pd.DataFrame(results)


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """
    Convert a DataFrame to CSV bytes for Streamlit download.
    """
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def main() -> None:
    st.title("💬 Sentiment Analysis App")
    st.caption("NLP inference with Hugging Face Transformers and Streamlit")

    with st.sidebar:
        st.header("Project Info")
        st.write("This app uses a pretrained Hugging Face model for sentiment analysis.")
        st.write("No fine-tuning or model training is performed in this version.")

        st.markdown("**Model**")
        st.code(MODEL_NAME)

        st.markdown("**Supported language**")
        st.write("English text is recommended.")

        st.markdown("**Output**")
        st.write("The app returns a sentiment label and confidence score.")

    try:
        classifier = load_sentiment_pipeline()
    except Exception as exc:
        st.error("The sentiment model could not be loaded.")
        st.exception(exc)
        st.stop()

    tab_single, tab_batch = st.tabs(["Single Text", "Batch Analysis"])

    with tab_single:
        st.subheader("Analyze one text")

        selected_example = st.selectbox(
            "Choose an example or write your own text below:",
            options=[""] + EXAMPLE_TEXTS,
            index=0,
        )

        default_text = selected_example if selected_example else ""

        user_text = st.text_area(
            "Input text",
            value=default_text,
            height=150,
            placeholder="Type a sentence or short paragraph in English...",
        )

        analyze_button = st.button("Analyze Sentiment", type="primary")

        if analyze_button:
            if not user_text.strip():
                st.warning("Please enter some text before analyzing.")
            else:
                result = predict_sentiment(user_text, classifier)

                col_label, col_confidence = st.columns(2)

                with col_label:
                    st.metric("Predicted Sentiment", result["label"])

                with col_confidence:
                    st.metric("Confidence Score", result["confidence"])

                st.markdown("### Result")
                st.dataframe(pd.DataFrame([result]), use_container_width=True)

    with tab_batch:
        st.subheader("Analyze multiple texts")

        st.write(
            "Enter one text per line, or upload a CSV file with a column named `text`."
        )

        batch_text = st.text_area(
            "Batch input",
            value="\n".join(EXAMPLE_TEXTS),
            height=220,
            placeholder="Enter one text per line...",
        )

        uploaded_file = st.file_uploader(
            "Optional: upload a CSV file with a `text` column",
            type=["csv"],
        )

        texts_to_analyze: List[str] = []

        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_file)

                if "text" not in uploaded_df.columns:
                    st.error("The uploaded CSV must contain a column named `text`.")
                else:
                    texts_to_analyze = uploaded_df["text"].dropna().astype(str).tolist()
                    st.success(f"Loaded {len(texts_to_analyze)} texts from CSV.")
            except Exception as exc:
                st.error("Could not read the uploaded CSV file.")
                st.exception(exc)
        else:
            texts_to_analyze = [
                line.strip()
                for line in batch_text.splitlines()
                if line.strip()
            ]

        analyze_batch_button = st.button("Analyze Batch", type="primary")

        if analyze_batch_button:
            if not texts_to_analyze:
                st.warning("Please enter at least one text or upload a valid CSV file.")
            else:
                results_df = predict_batch(texts_to_analyze, classifier)

                st.markdown("### Batch Results")
                st.dataframe(results_df, use_container_width=True)

                csv_bytes = dataframe_to_csv_bytes(results_df)

                st.download_button(
                    label="Download results as CSV",
                    data=csv_bytes,
                    file_name="sentiment_analysis_results.csv",
                    mime="text/csv",
                )

                sentiment_counts = results_df["label"].value_counts()

                st.markdown("### Sentiment Summary")
                st.bar_chart(sentiment_counts)


if __name__ == "__main__":
    main()