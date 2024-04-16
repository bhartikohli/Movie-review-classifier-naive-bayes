import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Load data (replace 'your_data.csv' with your actual data path)
data = pd.read_csv('C:\\Users\\hp\\OneDrive\\Desktop\IMDB Dataset\\IMDB Dataset.csv')
reviews = data['review']
sentiment = data['sentiment']

print("Wait a few minute then you can add the review")

# Data Preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum()]
    tokens = [token for token in tokens if token not in stop_words]
    return " ".join(tokens)

reviews_preprocessed = reviews.apply(preprocess_text)

# Feature extraction (Bag-of-Words)
vectorizer = CountVectorizer(max_features=2000)
features = vectorizer.fit_transform(reviews_preprocessed)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, sentiment, test_size=0.2, random_state=42)

# Model training (Naive Bayes)
model = MultinomialNB()
model.fit(X_train, y_train)

# Prediction on test data
y_pred = model.predict(X_test)

# Model evaluation
accuracy = (y_pred == y_test).mean()
print("Accuracy:", accuracy)

# Example usage for new review classification
while True:
    try:
        new_review = input("Enter your review (or type 'exit' to stop): ")

        if new_review.lower() == 'exit':
            print("Exiting...")
            break

        # Preprocess the new review
        new_review_preprocessed = preprocess_text(new_review)

        # Convert the preprocessed review into features
        new_review_features = vectorizer.transform([new_review_preprocessed])

        # Print the new review and its corresponding features
        print("New Review:", new_review)
        print("New Review Features:", new_review_features)

        # Predict sentiment for the new review
        prediction = model.predict(new_review_features)
        print("Predicted sentiment:", prediction[0])

        # Ask user if sentiment prediction is correct
        update_sentiment = input("Is the predicted sentiment correct? (yes/no): ")
        if update_sentiment.lower() == 'no':
            new_sentiment = input("Enter the correct sentiment (positive/negative): ")
            # Update sentiment in the dataset
            new_row = pd.Series([new_review, new_sentiment], index=['review', 'sentiment'])
            reviews = pd.concat([reviews, new_row], ignore_index=True)

            print("please wait while we update it")
            reviews_preprocessed = reviews.apply(preprocess_text)
            features = vectorizer.fit_transform(reviews_preprocessed)
            # Retrain the model with updated data
            X_train, X_test, y_train, y_test = train_test_split(features, sentiment, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            print("Model updated with new review and sentiment.")
    except Exception as e:
        print("An error occurred:", e)
