Final Version - Sentiment Analysis with BERT

In this version, we have implemented an advanced sentiment analysis approach using BERT (Bidirectional Encoder Representations from Transformers). The model is fine-tuned to analyze product reviews and classify them into five sentiment categories: Very Negative, Negative, Neutral, Positive, and Very Positive.

Key features of this version include:

Sentiment Analysis with BERT: Leveraging a pre-trained BERT model from HuggingFace’s nlptown/bert-base-multilingual-uncased-sentiment, this version enhances the quality of sentiment classification with state-of-the-art deep learning techniques.
Preprocessing of Reviews: Reviews are processed with tokenization, truncation, and padding using BERT’s tokenizer for better text encoding and handling large review datasets.
Interactive Sentiment Distribution: A Plotly-based interactive bar chart visualizes the distribution of sentiment classes with total counts displayed on top of each bar for better insights.
Top Positive and Negative Reviews: The application extracts and displays the top two positive and negative reviews, offering users a quick glimpse into the most impactful feedback.
Final Recommendation: Based on the overall sentiment distribution, the app provides a recommendation on whether the product is worth purchasing or not, adding real-world usability.
Optimized for Performance: The app employs Streamlit's st.cache_resource to efficiently load the BERT model and reduce re-loading time while maintaining high-quality sentiment analysis.
With these additions, this version offers a comprehensive tool for product review sentiment analysis with cutting-edge natural language processing (NLP) technology.

