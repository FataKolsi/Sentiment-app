import streamlit as st
import joblib
import re

# Preprocessing function
def preprocess(text):
    text = re.sub(r'[^\w\s\']',' ', text)
    text = re.sub(' +', ' ', text)
    return text.strip().lower()

@st.cache_resource
def load_model():
    return joblib.load("nb_pipeline_model.joblib")

model = load_model()

st.title("⭐ Star Rating Predictor (Naive Bayes + TF-IDF)")

user_input = st.text_area("Enter your review text:")

if st.button("Predict Rating"):
    if not user_input.strip():
        st.warning("Please enter some review text!")
    else:
        processed_text = preprocess(user_input)
        prediction = model.predict([processed_text])[0]
        st.success(f"Predicted Star Rating: ⭐ {prediction} stars")
