🛒 Product Review Sentiment Analysis Application
📄 Overview
This application performs sentiment analysis on product reviews using a deep learning model (BERT). Users can upload an Excel file with product reviews or ratings, and the sentiment of each review is analyzed and displayed. The app also provides insights through a sentiment distribution graph and highlights top positive and negative reviews.

It integrates with a blockchain-based user login system to ensure secure access. After logging in, users can analyze product reviews, receive recommendations, and export results.

✨ Features
🔐 Blockchain-based Login:

A secure login page built with blockchain technology. Users authenticate via Metamask, ensuring decentralized and secure access.
🧠 Sentiment Analysis Using Deep Learning (BERT):

Utilizes BERT for high-accuracy sentiment analysis, handling diverse product review data across multiple languages.
📝 Natural Language Processing (NLP):

NLP techniques are used for text tokenization and analysis. The BERT model ensures accurate sentiment prediction.
📊 Interactive Sentiment Distribution Visualization:

Provides a dynamic bar chart using Plotly to display sentiment distribution (Very Positive, Positive, Neutral, Negative, Very Negative).
⭐ Top Positive and Negative Reviews:

Highlights the top 2 positive and negative reviews, offering a quick overview of the best and worst feedback.
✅ Final Recommendation:

Based on the sentiment analysis, the app recommends whether the product is likely a good purchase.
📁 Excel File Upload:

Users can upload Excel files containing reviews or ratings for sentiment analysis.
🚀 Performance Optimizations:

Caching mechanisms for faster processing of large datasets, improving performance without compromising accuracy.
🛠 Version History
🥇 Version 1: Basic Sentiment Analysis
Initial Text Preprocessing.
ML Model for basic positive, negative, or neutral classification.
🥈 Version 2: Advanced NLP and Machine Learning
Added Word Embeddings and TF-IDF.
Improved with Random Forest and SVM models.
🥉 Version 3: Sentiment Analysis with LSTM
Transition to LSTM for enhanced sentiment prediction.
Utilized word embeddings with sequential models.
🏅 Version 4: BERT and Blockchain Integration
BERT for state-of-the-art sentiment analysis.
Integrated Blockchain User Authentication with Metamask.
Performance optimizations with caching.
🛠 Technologies Used
Python: Backend and logic
Streamlit: Interactive UI
Transformers (Hugging Face): BERT model for sentiment analysis
Blockchain (Metamask, Solidity): Secure decentralized login
Plotly: For sentiment distribution graphs
Pandas: Handling and processing Excel data
OpenPyXL: Read/write Excel files
Torch (PyTorch): Deep learning models
📝 How to Use
Login with Metamask 🔐:

Use Metamask to securely log in.
Upload Excel File 📁:

Upload a .xlsx file containing reviews or ratings.
Analyze Reviews 🧠:

View sentiment distribution and top reviews after uploading the file.
Get Recommendations ✅:

Based on the analysis, the app will provide a product recommendation.
Export Results 💾:

Export sentiment predictions and visualizations for further analysis.
⚙️ Installation and Setup
Clone this repository:

bash
Copy code
git clone https://github.com/chinmay0910/12_ProductReviewAnalysis
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt
Run the application:

bash
Copy code
streamlit run final_code.py
Access the app at https://reviewsentimentanalysis.streamlit.app/.

🚀 Future Enhancements
Support for additional languages through BERT fine-tuning.
Improved visualizations with advanced analytics.
More blockchain features to enhance security.
🤝 Contributing
Contributions are welcome! Fork the repository, create a branch, and submit a pull request. Follow the project’s coding guidelines.

