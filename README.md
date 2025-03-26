# Text-Sentiment-Analysis

Build a sentiment analysis model using a dataset such as IMDB Reviews.

IMDB Movie Reviews Sentiment Analysis Project

>Project Overview:

This project focuses on performing sentiment analysis on IMDB movie reviews. It uses Natural Language Processing (NLP) techniques to clean the text and machine learning models to classify the sentiment of reviews as positive or negative.

>We are using Python libraries like Pandas, NumPy, NLTK, and Scikit-learn to:

Preprocess text data

Vectorize text using TF-IDF

Train a Logistic Regression model

Evaluate model performance

>Project Steps:

Install Required Libraries:
pip install numpy pandas scikit-learn nltk

>These libraries handle:

Data manipulation (pandas, numpy)

NLP tasks (nltk)

Machine learning model (scikit-learn)

Import Libraries and Download NLTK Resources:

import pandas as pd

import numpy as np

import re

import nltk

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, accuracy_score

nltk.download('punkt')

nltk.download('stopwords')

nltk.download('wordnet')

>Load Dataset: 

df = pd.read_csv("IMDB Dataset.csv")

print(df.head())


>Text Preprocessing:

stop_words = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer()

def preprocess_text(text)
    
  text = text.lower()
    
  text = re.sub(r'[^a-zA-Z\s]', '', text)
    
  tokens = word_tokenize(text)
    
  tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
  return ' '.join(tokens)

df['cleaned_review'] = df['review'].apply(preprocess_text)

>Cleans text by:

Converting to lowercase

Removing punctuation/numbers

Tokenizing words

Removing stopwords

Lemmatizing words

>Split Dataset:

X_train, X_test, y_train, y_test = train_test_split(df['cleaned_review'], df['sentiment'], test_size=0.2, random_state=42)

>Splits the dataset into:

80% Training

20% Testing

>Convert Text to TF-IDF Features:

vectorizer = TfidfVectorizer();

X_train_tfidf = vectorizer.fit_transform(X_train);

X_test_tfidf = vectorizer.transform(X_test);

Transforms the cleaned text into numerical form so that machine learning models can process it.

>Train Logistic Regression Model

lr_model = LogisticRegression()

lr_model.fit(X_train_tfidf, y_train)

The Logistic Regression model is trained on the TF-IDF vectors.

>Make Predictions & Evaluate Model

y_pred_lr = lr_model.predict(X_test_tfidf)

print("Logistic Regression Performance:")

print(classification_report(y_test, y_pred_lr))

print("Accuracy:", accuracy_score(y_test, y_pred_lr))

Predicts sentiment and prints the model's precision, recall, F1-score, and accuracy.

> Observations

After running the script, the model will output an accuracy percentage, usually around 85%-88% depending on the dataset and data split.

Classification Report shows:

Precision: How accurate the positive/negative predictions are.

Recall: How well the model identifies all positive/negative instances.

F1-Score: Harmonic mean of precision and recall.

The model performs well and is capable of differentiating between positive and negative reviews based on textual content.

TF-IDF vectorization significantly improves text representation for the model.

>Dataset Source: Kaggle - IMDB Movie Reviews Dataset
