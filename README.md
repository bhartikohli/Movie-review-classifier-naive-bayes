Introduction:
This project aims to develop a movie review classification system utilizing the Naive Bayes classifier. 
The primary objective is to classify movie reviews as positive or negative based on their sentiment.
The Naive Bayes algorithm is chosen due to its simplicity and efficiency in text classification tasks.

Dataset:
The dataset comprises movie reviews sourced from Kaggle(IMDB Dataset), consisting of 2 features the review and sentiment labeled as positive or negative to denote the sentiment. 
Preprocessing steps, such as removing irrelevant information like HTML tags, punctuation, and stopwords, were applied to enhance data quality.

Methodology:

Preprocessing: Initial preprocessing involved tokenization, removal of stopwords, and stemming to extract meaningful features.
Feature Extraction: We adopted a bag-of-words model to convert reviews into numerical feature vectors.
Naive Bayes Classifier: Trained on the preprocessed dataset, the Naive Bayes classifier calculates conditional probabilities of each class (positive or negative) given the input features.
Model Evaluation: Performance metrics including accuracy, precision, recall, and F1-score were used to assess the classifier's effectiveness.

Implementation:

Programming Language: 
Python
Libraries Used:
scikit-learn: for Naive Bayes classifier implementation and evaluation metrics.
NLTK (Natural Language Toolkit): for text preprocessing.
Steps:
Data loading and preprocessing
Feature extraction (bag-of-words)
Dataset split into training and testing sets
Training Naive Bayes classifier
Evaluation on test set



