import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Function to process the uploaded Excel file and extract reviews
def process_file(file):
    df = pd.read_excel(file)
    if 'Reviews' in df.columns:
        reviews = df['Reviews'].dropna()
    else:
        st.error("The file must have a 'Reviews' column.")
        return None
    return reviews

# Sentiment Analysis using LSTM
def sentiment_analysis_lstm(reviews):
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(reviews)
    sequences = tokenizer.texts_to_sequences(reviews)
    X = pad_sequences(sequences, maxlen=100)

    # Mock labels for demonstration
    y = np.array([1 if i % 2 == 0 else 0 for i in range(len(reviews))])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = Sequential([
        Embedding(input_dim=5000, output_dim=64, input_length=100),
        LSTM(64, return_sequences=False),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))
    
    st.write(f"Model Accuracy: {model.evaluate(X_test, y_test)[1]}")

# Streamlit App
st.title("Deep Learning with LSTM for Sentiment Analysis")
uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx'])

if uploaded_file is not None:
    reviews = process_file(uploaded_file)
    if reviews is not None:
        sentiment_analysis_lstm(reviews)
