import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download NLTK resources if they aren't already available
nltk.download('stopwords')
nltk.download('wordnet')

# Define lemmatizer for TF-IDF
def lemmatized_words(doc):
    lemmatizer = WordNetLemmatizer()
    words = doc.split()
    return [lemmatizer.lemmatize(word) for word in words]

# Attempt to load a pre-trained model or train a new one if not found
model_path = 'sentiment_model.joblib'
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    # Function to load data from CSV files
    def load_data(filenames):
        data_frames = []
        for filename in filenames:
            data_frames.append(pd.read_csv(os.path.join('datasets', filename)))
        combined_data = pd.concat(data_frames)
        return combined_data['text'], combined_data['label']
    
    # Define dataset files and train model if needed
    dataset_files = ['positive.csv', 'negative.csv', 'neutral.csv']
    X, y = load_data(dataset_files)
    model = make_pipeline(
        TfidfVectorizer(tokenizer=lemmatized_words, stop_words=stopwords.words('english')),
        MultinomialNB()
    )
    
    # Train-test split and model training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    
    # Save the model to the specified path
    joblib.dump(model, model_path)
    
    # Print the classification report to verify model accuracy
    print(classification_report(y_test, model.predict(X_test)))

# Function to get sentiment prediction
def get_sentiment(text):
    prediction = model.predict([text])[0]
    return prediction
