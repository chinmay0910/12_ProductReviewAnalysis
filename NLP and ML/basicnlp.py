import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Function to process the uploaded Excel file and extract reviews
def process_file(file):
    df = pd.read_excel(file)
    if 'Reviews' in df.columns:
        reviews = df['Reviews'].dropna()
    else:
        st.error("The file must have a 'Reviews' column.")
        return None
    return reviews

# Sentiment Analysis using Logistic Regression and TF-IDF
def sentiment_analysis_logistic(reviews):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(reviews)
    
    # Mock labels for demonstration (replace with actual labels)
    y = [1 if i % 2 == 0 else 0 for i in range(len(reviews))]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Streamlit App
st.title("Logistic Regression Sentiment Analysis")
uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx'])

if uploaded_file is not None:
    reviews = process_file(uploaded_file)
    if reviews is not None:
        sentiment_analysis_logistic(reviews)