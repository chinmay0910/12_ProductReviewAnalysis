import streamlit as st
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import plotly.graph_objects as go
import numpy as np
import plotly.express as px


# Load Pre-trained BERT Tokenizer and Model
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    return tokenizer, model

tokenizer, model = load_model()

# Function to preprocess text (batch processing)
def preprocess_batch(texts):
    inputs = tokenizer(
        texts,
        max_length=512,
        truncation=True,
        padding=True,  # Pad to the longest sequence in batch
        add_special_tokens=True,
        return_tensors='pt'
    )
    return inputs

# Function to predict sentiment (batch processing)
def predict_sentiments(texts):
    inputs = preprocess_batch(texts)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_classes = logits.argmax(dim=1).numpy()
    
    # Map predicted classes to sentiments (0: Very Negative, 4: Very Positive)
    sentiment_map = {0: 'Very Negative', 1: 'Negative', 2: 'Neutral', 3: 'Positive', 4: 'Very Positive'}
    sentiments = [sentiment_map[pred] for pred in predicted_classes]
    return sentiments

# Function to process the uploaded Excel file and extract reviews
def process_file(file):
    df = pd.read_excel(file)
    
    # Assuming the reviews are in a column named 'Reviews' or 'Ratings'
    if 'Reviews' in df.columns:
        reviews = df['Reviews'].dropna().tolist()
    elif 'Ratings' in df.columns:
        reviews = df['Ratings'].dropna().tolist()
    else:
        st.error("The file must have a 'Reviews' or 'Ratings' column.")
        return None
    
    return reviews

# Function to plot interactive sentiment distribution
def plot_sentiment_distribution(sentiment_counts):
    fig = go.Figure(data=[
        go.Bar(x=sentiment_counts.index, 
               y=sentiment_counts.values,
               marker=dict(color=px.colors.qualitative.Vivid),
               text=sentiment_counts.values, 
               textposition='outside')  # Display total count on top of each bar
    ])
    
    fig.update_layout(
        title="Sentiment Distribution",
        xaxis_title="Sentiment",
        yaxis_title="Count",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=14),
        showlegend=False
    )
    
    st.plotly_chart(fig)

# Function to analyze reviews and provide summary
def analyze_reviews(reviews):
    # Process reviews in batches for speed
    batch_size = 16  # You can adjust this based on your system capacity
    sentiments = []
    positive_reviews = []
    negative_reviews = []

    # Split reviews into batches and process each batch
    for i in range(0, len(reviews), batch_size):
        batch_reviews = reviews[i:i+batch_size]
        batch_sentiments = predict_sentiments(batch_reviews)
        sentiments.extend(batch_sentiments)

        for review, sentiment in zip(batch_reviews, batch_sentiments):
            if 'Positive' in sentiment:
                positive_reviews.append(review)
            elif 'Negative' in sentiment:
                negative_reviews.append(review)

    sentiment_counts = pd.Series(sentiments).value_counts()

    # Display Sentiment Distribution
    st.subheader("Sentiment Distribution")
    plot_sentiment_distribution(sentiment_counts)

    # Display Top Positive and Negative Reviews
    st.subheader("Top Positive Reviews")
    if positive_reviews:
        for review in positive_reviews[:2]:
            st.write(f"- {review}")
    else:
        st.write("None")

    st.subheader("Top Negative Reviews")
    if negative_reviews:
        for review in negative_reviews[:2]:
            st.write(f"- {review}")
    else:
        st.write("None")
    
    # Final Recommendation
    if sentiment_counts.get('Positive', 0) + sentiment_counts.get('Very Positive', 0) > sentiment_counts.get('Negative', 0) + sentiment_counts.get('Very Negative', 0):
        st.success("Recommendation: Based on the reviews, this product is likely a good buy!")
    else:
        st.warning("Recommendation: Based on the reviews, this product may not be a good buy.")

# Streamlit App
st.title("Product Review Sentiment Analysis")
st.write("Upload an Excel file with product reviews or ratings to analyze sentiments and get recommendations.")

uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx'])

if uploaded_file is not None:
    reviews = process_file(uploaded_file)
    
    if reviews is not None:
        st.write(f"Analyzing {len(reviews)} reviews...")
        analyze_reviews(reviews)
