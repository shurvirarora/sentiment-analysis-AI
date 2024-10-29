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

nltk.download('stopwords')
nltk.download('wordnet')

def lemmatized_words(doc):
    lemmatizer = WordNetLemmatizer()
    words = doc.split()
    return [lemmatizer.lemmatize(word) for word in words]

# Attempt to load the trained model or train if not found
try:
    model = joblib.load('sentiment_model.joblib')
except FileNotFoundError:
    # Load and combine data from multiple files, then train the model
    def load_data(filenames):
        data_frames = []
        for filename in filenames:
            data_frames.append(pd.read_csv('datasets/' + filename))
        combined_data = pd.concat(data_frames)
        return combined_data['text'], combined_data['label']
    
    dataset_files = ['positive.csv', 'negative.csv', 'neutral.csv']
    X, y = load_data(dataset_files)
    model = make_pipeline(
        TfidfVectorizer(tokenizer=lemmatized_words, stop_words=stopwords.words('english')),
        MultinomialNB()
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, 'sentiment_model.joblib')
    print(classification_report(y_test, model.predict(X_test)))

def get_sentiment(text):
    prediction = model.predict([text])[0]
    return prediction
