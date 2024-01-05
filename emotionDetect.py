import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
import string
from sklearn.metrics import precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score  
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load the data
data = pd.read_csv("C:/Users/HP/Desktop/yzodev/emotion/emotion-labels.csv")

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer() 
# Preprocessing functions

def clean_text(text):
  cleaned_text = ""
  tokens = word_tokenize(text.lower()) # Tokenize and lowercase words
  for word in tokens:
    word = lemmatizer.lemmatize(word)
    stemmer.stem(word) # Lemmatize ve stemming
    if word not in stopwords_set and word not in punctuation_set:
      cleaned_text += word + " "
  return cleaned_text.strip()

analyzer = SentimentIntensityAnalyzer()

def calculate_sentiment(text):
  sentiment_scores = analyzer.polarity_scores(text)
  return sentiment_scores['compound'] # Return a single sentiment score

def extract_features(text):
  text_length = len(text.split())
  sentiment_score = calculate_sentiment(text) # Replace with your sentiment analysis function
  return clean_text, text_length, sentiment_score

# Load stopwords and punctuation
stopwords_set = set(stopwords.words("english"))
punctuation_set = set(string.punctuation)

# Clean the text data
data["Text"] = data["Text"].apply(clean_text)

# Define hyperparameter grids for TfidfVectorizer and LogisticRegression
tfidf_params = {
  'ngram_range': [(1, 1), (1, 2), (2, 2)], # Explore different n-gram ranges
  'min_df': [2, 5], # Try different minimum document frequencies
  'max_df': [0.85, 0.9], # Experiment with maximum document frequencies
  'max_features': [5000, 10000],
}

lr_params = {
  'penalty': ['l1', 'l2'], # Test different regularization techniques
  'C': [0.1, 1, 10], # Explore different regularization strengths
  'solver': ['newton-cg', 'lbfgs', 'liblinear'] # Experiment with solvers
}

pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', LogisticRegression())
])

grid_search = GridSearchCV(pipeline, param_grid={'vectorizer__max_df': [0.8, 0.9], 'vectorizer__ngram_range': [(1, 1), (1, 2), (2, 2)], 'vectorizer__min_df': [2, 5], 'classifier__C': [0.1, 1, 10], 'classifier__penalty': ['l1', 'l2']}, cv=5)
grid_search.fit(data['Text'], data['Emotion'])

# Access the best model and its hyperparameters
best_model = grid_search.best_estimator_
print("Best hyperparameters:", grid_search.best_params_)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data['Text'], data['Emotion'], test_size=0.10, random_state=1000)

# Fit the pipeline to the training data
pipeline.fit(x_train, y_train)

# Get user input
user_input = input("Enter a sentence: ")
cleaned_user_input = clean_text(user_input)

y_pred = pipeline.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Predict the emotion for the input sentence
user_prediction = pipeline.predict([cleaned_user_input])[0]
print(f"Predicted Emotion: {user_prediction}")

# Comprehensive accuracy assessment
print("Accuracy on the entire test set:")
accuracy = accuracy_score(y_test, pipeline.predict(x_test))
print(accuracy)

print("Cross-validation accuracy:")
cv_scores = cross_val_score(pipeline, data['Text'], data['Emotion'], cv=5)
print(cv_scores.mean())

print("Classification report on the test set:")
print(classification_report(y_test, pipeline.predict(x_test)))

# Calculate additional metrics
y_pred = pipeline.predict(x_test)
precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')