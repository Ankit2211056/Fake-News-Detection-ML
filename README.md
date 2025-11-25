#Fake News Detection using Logistic Regression

A Machine Learning project to classify news articles as Fake (1) or Real (0) using NLP techniques and Logistic Regression.

Project Overview : 
-Fake news is a growing problem in the digital world.
-This ML model detects whether a given news article is real or fake based on its content.

This project uses:
-Text Preprocessing
-TF-IDF Vectorization
-Logistic Regression Classifier

Dataset Information: 
The dataset contains the following columns-
title – Headline of the news
text – Full news article
label – 1 → Fake News, 0 → Real News

Dataset was cleaned using:
✔ Removing punctuation
✔ Lowercasing
✔ Removing stopwords
✔ Stemming
✔ Combining title + text into a single content column

 Machine Learning Model Used Algorithm: Logistic Regression
Vectorizer: TF-IDF (Term Frequency – Inverse Document Frequency)

Model Performance:
Metric	Score:
-Training Accuracy	0.89
-Testing Accuracy	0.87
The model performs well and generalizes effectively without overfitting.

Technologies & Libraries Used:
Python
NumPy
Pandas
NLTK
Scikit-learn
TF-IDF Vectorizer
Logistic Regression
Google Colab

How the Model Works:
-Load Dataset
-Fill Missing Values
-Combine Title + Text
-Text Preprocessing
-Remove symbols
-Convert to lowercase
-Remove stopwords
-Perform stemming
-Convert text into numerical features using TF-IDF
-Train Logistic Regression Model
-Predict Real (0) / Fake (1)

Sample Prediction Code:
X_new = X_test[0]
prediction = model.predict(X_new)
if prediction[0] == 1:
    print("The news is Fake")
else:
    print("The news is Real")

Future Improvements:
-Use advanced NLP models (LSTM, BERT)
-Add GUI or Streamlit Web App
-Deploy using Flask / FastAPI

Contributions
Contributions are welcome! Feel free to fork and improve the project.

Author:
Ankit Kumar
Machine Learning & Data Science Enthusiast
