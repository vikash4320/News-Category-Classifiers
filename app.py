import streamlit as st
import pickle
from preprocess import clean_text

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

labels = ["World", "Sports", "Business", "Sci/Tech"]

st.title("📰 News Category Classifier")

news = st.text_area("Enter news headline or description:")

if st.button("Predict"):
    if news.strip() == "":
        st.warning("Please enter text")
    else:
        cleaned = clean_text(news)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        st.success(f"Category: {labels[prediction]}")
