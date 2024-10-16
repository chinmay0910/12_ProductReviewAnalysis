# 🛒 Product Review Sentiment Analysis Application

## 📄 Overview
This application performs **sentiment analysis** on product reviews using a deep learning model (BERT). Users can upload an Excel file containing product reviews or ratings, and the sentiment of each review is analyzed and displayed. The app also provides insights through a sentiment distribution graph and highlights top positive and negative reviews.

The application integrates with a **blockchain-based user login** system to ensure secure access. After logging in, users can analyze product reviews, receive recommendations, and export results.

## ✨ Features

1. **🔐 Blockchain-based Login**:
   - Secure login via **Metamask** ensuring a decentralized and safe authentication process.

2. **🧠 Sentiment Analysis Using Deep Learning (BERT)**:
   - **BERT** (Bidirectional Encoder Representations from Transformers) for high-accuracy sentiment analysis, handling a wide range of review data.

3. **📝 Natural Language Processing (NLP)**:
   - Tokenization and preprocessing of review text using NLP techniques to extract patterns and insights.

4. **📊 Interactive Sentiment Distribution Visualization**:
   - Dynamic sentiment distribution graph (Very Positive, Positive, Neutral, Negative, Very Negative) using **Plotly** for interactive visualization.

5. **⭐ Top Positive and Negative Reviews**:
   - Highlights the top 2 positive and negative reviews to give a quick overview of product feedback.

6. **✅ Final Recommendation**:
   - Based on the sentiment analysis, the app provides a recommendation on whether the product is likely a good purchase.

7. **📁 Excel File Upload**:
   - Users can upload Excel files (.xlsx) containing product reviews or ratings, which the system will process for sentiment analysis.

8. **🚀 Performance Optimizations**:
   - Optimized for fast processing of large datasets through caching mechanisms, allowing quicker sentiment analysis without compromising accuracy.

## 🛠 Version History

### 🥇 Version 1: Basic Sentiment Analysis
- **Text Preprocessing**: Basic cleaning and sentiment analysis using simple lexical approaches.
- **ML Model**: Logistic regression for classifying reviews into positive, negative, or neutral.

### 🥈 Version 2: Advanced NLP and Machine Learning
- **Word Embeddings and TF-IDF**: For text preprocessing.
- **Machine Learning Models**: Implemented **Random Forest** and **SVM** for sentiment classification.

### 🥉 Version 3: Sentiment Analysis with LSTM
- **LSTM (Long Short-Term Memory)**: Used deep learning to improve sentiment analysis accuracy by processing sequential data.
- **Word Embeddings**: Combined with LSTM layers for capturing long-term dependencies in text.

### 🏅 Version 4: BERT and Blockchain Integration
- **BERT for Sentiment Analysis**: State-of-the-art NLP model for highly accurate and nuanced sentiment analysis.
- **Blockchain-based User Authentication**: Secure decentralized login via **Metamask**.
- **Performance Optimizations**: Caching and optimized model deployment for quick and efficient processing.

## 🛠 Technologies Used
- **Python**: Backend logic and sentiment analysis
- **Streamlit**: Interactive web framework for the frontend
- **Transformers (Hugging Face)**: For BERT model integration
- **Blockchain (Metamask, Solidity)**: Secure user login via blockchain
- **Plotly**: For interactive visualizations
- **Pandas**: Data processing and manipulation
- **OpenPyXL**: Read and write Excel files
- **PyTorch**: For handling deep learning models like BERT

## 📝 How to Use

1. **Login with Metamask** 🔐:
   - Ensure **Metamask** is installed and logged in. Use the Metamask login page to authenticate and access the app.

2. **Upload an Excel File** 📁:
   - Upload an Excel file (.xlsx) with product reviews or ratings. The file should contain a column labeled "Reviews" or "Ratings".

3. **Analyze Reviews** 🧠:
   - Once the file is uploaded, the app will process the reviews and display the sentiment distribution (positive, negative, etc.) and top positive/negative reviews.

4. **Get Recommendations** ✅:
   - Based on sentiment analysis, the app will provide a recommendation on whether the product is likely a good purchase.

5. **Export Results** 💾:
   - You can export the sentiment predictions and visualizations for further analysis.

## ⚙️ Installation and Setup

1. Clone the repository to your local machine:
    ```bash
    git clone https://github.com/chinmay0910/12_ProductReviewAnalysis
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the application:
    ```bash
    streamlit run final_code.py
    ```

4. Open the application in your browser at `https://reviewsentimentanalysis.streamlit.app/`.

## 🌟 Future Enhancements
- **Multilingual Support**: Extend support for other languages by fine-tuning the BERT model for different language datasets.
- **Advanced Analytics**: Improve visualizations with more detailed insights and interactive charts.
- **Enhanced Blockchain Integration**: Add more decentralized features for increased user privacy and security.

## 🤝 Contributing
We welcome contributions! If you'd like to contribute:
1. Fork the repository.
2. Create a new branch for your feature.
3. Submit a pull request.



