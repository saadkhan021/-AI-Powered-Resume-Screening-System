# app.py

import streamlit as st
import joblib
import fitz  # PyMuPDF
import re

# Load trained model and vectorizer
model = joblib.load("resume_classifier_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Clean text function (must match training)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Extract text from uploaded PDF file
def extract_text_from_pdf(file):
    try:
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f" Error reading PDF: {e}")
        return ""

# Streamlit UI
st.set_page_config(page_title="Resume Classifier", layout="centered")
st.title(" AI-Powered Resume Screening System")
st.write("Upload a resume in PDF format and get an AI-predicted job category!")

# File uploader
uploaded_file = st.file_uploader(" Upload Resume (PDF only)", type=["pdf"])

if uploaded_file is not None:
    with st.spinner(" Extracting text from PDF..."):
        resume_text = extract_text_from_pdf(uploaded_file)

    if resume_text.strip() == "":
        st.warning(" Could not extract any text from this PDF. Please try another file.")
    else:
        # Clean and vectorize
        cleaned = clean_text(resume_text)
        vectorized = vectorizer.transform([cleaned])

        # Predict
        prediction = model.predict(vectorized)[0]
        confidence = model.predict_proba(vectorized).max()

        # Display result
        st.success(f" **Predicted Job Category:** {prediction}")
        st.info(f" **Confidence Score:** {confidence:.2f}")

        # Optional: Show extracted resume text
        with st.expander(" Show Extracted Resume Text"):
            st.text_area("Resume Preview", resume_text, height=300)
