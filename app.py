import numpy as np
import streamlit as st

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="IMDb Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_sentiment_model():
    return load_model("imdb_simple_rnn.h5")

model = load_sentiment_model()

# ----------------------------
# Load Word Index
# ----------------------------
word_index = imdb.get_word_index()

# ----------------------------
# Preprocessing Function
# ----------------------------
def preprocess_text(text):
    words = text.lower().split()

    max_features = 10000
    encoded_review = []

    for word in words:
        base_index = word_index.get(word, 2)
        shifted_index = base_index + 3

        if shifted_index < max_features:
            encoded_review.append(shifted_index)
        else:
            encoded_review.append(2)

    padded_review = sequence.pad_sequences(
        [encoded_review],
        maxlen=500
    )

    return padded_review


# ----------------------------
# Header
# ----------------------------
st.title("🎬 IMDb Movie Review Sentiment Analyzer")

st.markdown("""
Analyze movie reviews using a **Simple RNN with Embedding Layers**.

The model predicts whether a review expresses a:

- 😊 Positive sentiment
- 😠 Negative sentiment
""")

st.divider()

# ----------------------------
# Examples
# ----------------------------
with st.expander("📝 Example Reviews"):

    st.markdown("""
**Positive Example**

> This movie was absolutely fantastic with brilliant acting and an amazing storyline.

**Negative Example**

> The movie was boring, predictable and a complete waste of time.
""")

# ----------------------------
# User Input
# ----------------------------
review = st.text_area(
    "✍️ Enter Movie Review",
    height=180,
    placeholder="Example: This movie was absolutely amazing..."
)

# ----------------------------
# Prediction Button
# ----------------------------
if st.button(
    "🔍 Analyze Sentiment",
    use_container_width=True
):

    if review.strip() == "":
        st.warning("⚠️ Please enter a movie review.")
        st.stop()

    with st.spinner("Analyzing review sentiment..."):

        processed_input = preprocess_text(review)

        prediction = model.predict(
            processed_input,
            verbose=0
        )[0][0]

        sentiment = (
            "Positive"
            if prediction > 0.5
            else "Negative"
        )

        confidence = (
            prediction
            if prediction > 0.5
            else 1 - prediction
        )

    st.divider()

    st.subheader("📊 Analysis Result")

    if sentiment == "Positive":

        st.success(
            f"😊 Positive Review Detected"
        )

        st.progress(
            float(confidence)
        )

        st.metric(
            "Confidence Score",
            f"{confidence*100:.2f}%"
        )

    else:

        st.error(
            f"😠 Negative Review Detected"
        )

        st.progress(
            float(confidence)
        )

        st.metric(
            "Confidence Score",
            f"{confidence*100:.2f}%"
        )

    st.write("### Prediction Probability")

    st.write(
        f"Positive Probability: **{prediction*100:.2f}%**"
    )

    st.write(
        f"Negative Probability: **{(1-prediction)*100:.2f}%**"
    )

st.divider()

st.caption(
    "Built using TensorFlow, Keras, Simple RNN, Embedding Layer and Streamlit 🚀"
)
