import streamlit as st
import fasttext

@st.cache_resource
def load_model():
    return fasttext.load_model("fasttext_supervised.bin")

model = load_model()

st.title("🌟 Star Rating Predictor with FastText")

user_input = st.text_area("Enter your review text:")

if st.button("Predict Rating"):
    if not user_input.strip():
        st.warning("Please enter some review text!")
    else:
        label, prob = model.predict(user_input)
        rating = int(label[0].replace("__label__", ""))
        st.success(f"Predicted Star Rating: ⭐ {rating} stars")
        st.caption(f"Confidence Score: {prob[0]:.2f}")
