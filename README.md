# Product Review Sentiment Analysis Application

## Overview
This application performs **sentiment analysis** on product reviews using a deep learning model (BERT). The system allows users to upload an Excel file containing product reviews or ratings, and the sentiment of each review is analyzed and displayed. It also provides insights by showcasing a sentiment distribution graph, along with top positive and negative reviews.

The application integrates with a **blockchain-based user login** system to ensure secure access. After logging in, users can analyze product reviews, receive recommendations, and export the results as needed. 

## Features
1. **Blockchain-based Login**:
    - The application includes a **secure login page** built using blockchain technology. Users can log in using their **Metamask** wallet, ensuring a decentralized and secure authentication process.
  
2. **Sentiment Analysis Using Deep Learning (BERT)**:
    - For sentiment analysis, the system leverages **BERT (Bidirectional Encoder Representations from Transformers)**. BERT's language understanding capabilities enable the system to predict sentiment with high accuracy, handling a wide range of review data.

3. **Natural Language Processing (NLP)**:
    - **NLP** techniques are used to preprocess the reviews, tokenize the text, and extract meaningful patterns. The model performs multi-lingual sentiment analysis, making it flexible for various kinds of textual data.

4. **Interactive Sentiment Distribution Visualization**:
    - The application provides a **dynamic, interactive bar chart** to display the sentiment distribution (Very Positive, Positive, Neutral, Negative, Very Negative). It uses **Plotly** for visualization, which includes user-friendly tooltips and color-coded graphs for easy interpretation. Each bar shows the total number of reviews for each sentiment category.

5. **Top Positive and Negative Reviews**:
    - The application automatically highlights the top 2 positive and negative reviews, helping users quickly identify the best and worst feedback on the product.
    
6. **Final Recommendation**:
    - Based on the sentiment analysis, the application provides an **overall recommendation** on whether the product is likely a good purchase.

7. **Excel File Upload**:
    - Users can upload an Excel file with reviews or ratings, and the application will process and analyze the data. The system supports files containing columns labeled "Reviews" or "Ratings".

8. **Performance Optimizations**:
    - The application has been optimized for **fast processing** of large datasets without compromising the quality of sentiment analysis. Various caching mechanisms have been implemented to speed up repetitive tasks.

## Version History

### Version 1: Basic Sentiment Analysis
- **Basic Text Preprocessing**:
  - Initial version of the application included basic text cleaning and sentiment analysis using simple rule-based or lexical approaches.
  
- **ML Model for Sentiment**:
  - Implemented a basic **machine learning model** (such as logistic regression) to classify reviews as positive, negative, or neutral.

### Version 2: Advanced NLP and Machine Learning
- **Word Embeddings and TF-IDF**:
  - Incorporated **word embedding techniques** and **TF-IDF** vectorization for text preprocessing.
  
- **Machine Learning Models**:
  - Implemented machine learning algorithms like **Random Forest** and **Support Vector Machines (SVM)** to improve the sentiment classification.
  
### Version 3: Sentiment Analysis with LSTM
- **Sentiment Analysis Using LSTM**:
  - Transitioned to a **deep learning approach** by using **LSTM (Long Short-Term Memory)** to handle sequential data and improve sentiment analysis accuracy.
  
- **Word Embeddings and Sequential Model**:
  - Employed word embeddings with LSTM layers to capture long-term dependencies in the text data, providing better predictions.

### Version 4: Final Version with BERT and Blockchain Integration
- **BERT for Sentiment Analysis**:
  - The current version employs **BERT**, a state-of-the-art model in natural language processing, for more accurate and nuanced sentiment analysis.
  
- **Blockchain User Authentication**:
  - Integrated a secure, decentralized **blockchain-based login** system using **Metamask**, ensuring user privacy and security.
  
- **Optimized Performance**:
  - Significant improvements in performance and speed, with **caching** mechanisms and optimized model deployment for quick and efficient analysis.

## Technologies Used
- **Python**: Backend logic and sentiment analysis
- **Streamlit**: Web framework for building the interactive user interface
- **Transformers (Hugging Face)**: For using **BERT** model for sentiment analysis
- **Blockchain (Metamask, Solidity)**: Used for creating a secure, decentralized login system
- **Plotly**: For interactive sentiment distribution graphs
- **Pandas**: For handling and processing Excel files
- **OpenPyXL**: To read and write Excel files
- **Torch (PyTorch)**: For handling deep learning models such as BERT

## How to Use
1. **Login with Metamask**:
    - Ensure you have the **Metamask** extension installed and are logged into your wallet.
    - Use the Metamask login page to authenticate and access the app.

2. **Upload an Excel File**:
    - Upload an Excel file (.xlsx) containing a column of product reviews or ratings.
    
3. **Analyze Reviews**:
    - Once the file is uploaded, the app will process the reviews and display the sentiment distribution and top positive/negative reviews.

4. **Get Recommendations**:
    - Based on the sentiment analysis, the application will give you a recommendation on whether the product is likely a good purchase.

5. **Export Results**:
    - You can export the results, such as sentiment predictions and visualizations, for further analysis.

## Installation and Setup
1. Clone this repository to your local machine:
    ```bash
    git clone https://github.com/your-username/repo-name.git
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the application:
    ```bash
    streamlit run app2.py
    ```

4. Access the app on `http://localhost:8501/`.

## Future Enhancements
- Expand support for other languages by fine-tuning the BERT model.
- Improve the visualizations with more advanced analytics.
- Integrate more decentralized features using blockchain.
  
## Contributing
We welcome contributions from the community. If you'd like to contribute, please fork the repository, create a branch for your feature, and submit a pull request. Make sure to follow the project's coding guidelines.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
