# ============================================
# 💬 Sentiment Analysis Web App (Streamlit)
# ============================================

# ============================================
# 📌 Import Libraries
# ============================================
import streamlit as st
import numpy as np
import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# ============================================
# 📌 Page Setup
# ============================================
st.set_page_config(page_title="Sentiment Analysis App", page_icon="💬", layout="centered")

st.title("💬 Sentiment Analysis NLP App")
st.write("Analyze the sentiment of text using Machine Learning 🚀")

# ============================================
# 📌 Download NLTK Data
# ============================================
nltk.download('stopwords')

# ============================================
# 📌 Load Dataset
# ============================================
@st.cache_data
def load_data():
    return pd.read_csv("dataset.csv")

dataset = load_data()

# 👉 Show dataset preview (for debugging - optional)
st.write("### 📊 Dataset Preview")
st.dataframe(dataset.head())

# ============================================
# 📌 Preprocessing Function
# ============================================
def preprocess_text(text):
    text = re.sub(r'\W', ' ', str(text))
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'^[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    text = re.sub(r'^b\s+', '', text)
    text = text.lower()
    return text

# ============================================
# 📌 Prepare Data (IMPORTANT FIX)
# ============================================
# ⚠️ Change these indexes if needed
features = dataset.iloc[:, 0].values   # Text column
labels = dataset.iloc[:, 1].values     # Sentiment column

# Clean text
processed_features = [preprocess_text(text) for text in features]

# ============================================
# 📌 TF-IDF Vectorization
# ============================================
vectorizer = TfidfVectorizer(
    max_features=2500,
    stop_words=stopwords.words('english')
)

X = vectorizer.fit_transform(processed_features).toarray()

# ============================================
# 📌 Train Model (Cached for speed)
# ============================================
@st.cache_resource
def train_model(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=0)
    model.fit(X, y)
    return model

model = train_model(X, labels)

st.success("✅ Model Loaded Successfully!")

# ============================================
# 📌 User Input Section
# ============================================
st.subheader("✍️ Enter Text to Analyze")

user_input = st.text_area("Type your text here...")

# ============================================
# 📌 Prediction
# ============================================
if st.button("🔍 Analyze Sentiment"):
    
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text!")
    else:
        # Preprocess
        processed_input = preprocess_text(user_input)
        
        # Vectorize
        vectorized_input = vectorizer.transform([processed_input]).toarray()
        
        # Predict
        prediction = model.predict(vectorized_input)[0]
        
        # Show Result
        st.success(f"Predicted Sentiment: {prediction}")

        # Emoji Display
        if str(prediction).lower() == "positive":
            st.markdown("### 😊 Positive Sentiment")
        elif str(prediction).lower() == "negative":
            st.markdown("### 😡 Negative Sentiment")
        else:
            st.markdown("### 😐 Neutral Sentiment")

# ============================================
# 📌 Footer
# ============================================
st.markdown("---")
st.caption("Built with ❤️ using Streamlit & Machine Learning")