# ============================================
# 📌 Sentiment Analysis using NLP
# ============================================

# ============================================
# 📌 Install Required Libraries (Run Once)
# ============================================
# !pip install nltk

# ============================================
# 📌 Import Libraries
# ============================================
import numpy as np
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# ============================================
# 📌 Download NLTK Stopwords
# ============================================
nltk.download('stopwords')

# ============================================
# 📌 Load Dataset
# ============================================
# 👉 If using Google Colab
# from google.colab import files
# uploaded = files.upload()

dataset = pd.read_csv('dataset.csv')

# Display dataset info
print("Dataset Shape:", dataset.shape)
print(dataset.head())

# ============================================
# 📌 Separate Features & Labels
# ============================================
# ⚠️ Change column index if needed
features = dataset.iloc[:, 10].values   # Text column
labels = dataset.iloc[:, 1].values      # Sentiment column

print("Sample Labels:", labels[:5])

# ============================================
# 📌 Text Preprocessing
# ============================================
processed_features = []

for sentence in range(len(features)):
    
    # Remove special characters
    processed_feature = re.sub(r'\W', ' ', str(features[sentence]))
    
    # Remove single characters
    processed_feature = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)
    
    # Remove single characters from start
    processed_feature = re.sub(r'^[a-zA-Z]\s+', ' ', processed_feature)
    
    # Replace multiple spaces with single space
    processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)
    
    # Remove prefixed 'b'
    processed_feature = re.sub(r'^b\s+', '', processed_feature)
    
    # Convert to lowercase
    processed_feature = processed_feature.lower()
    
    processed_features.append(processed_feature)

# ============================================
# 📌 Feature Extraction (TF-IDF)
# ============================================
vectorizer = TfidfVectorizer(
    max_features=2500,
    min_df=7,
    max_df=0.8,
    stop_words=stopwords.words('english')
)

X = vectorizer.fit_transform(processed_features).toarray()

print("Feature Vector Shape:", X.shape)

# ============================================
# 📌 Train-Test Split
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=0
)

# ============================================
# 📌 Model Training (Random Forest)
# ============================================
model = RandomForestClassifier(n_estimators=200, random_state=0)
model.fit(X_train, y_train)

# ============================================
# 📌 Predictions
# ============================================
y_pred = model.predict(X_test)

# ============================================
# 📌 Model Evaluation
# ============================================
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# ============================================
# 📌 Test with Custom Input
# ============================================
new_input = ["@virginamerica Well, I didn't…but NOW I DO! :-D"]

# Transform input
new_input_vectorized = vectorizer.transform(new_input).toarray()

# Predict
prediction = model.predict(new_input_vectorized)

print("Predicted Sentiment:", prediction[0])