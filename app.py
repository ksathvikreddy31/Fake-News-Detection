from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Load the model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Load the dataset with specified encoding
fake_df = pd.read_csv(r"E:\projects\fake_news_prediction\archive (4)\News _dataset\Fake.csv", encoding='ISO-8859-1')
true_df = pd.read_csv(r"E:\projects\fake_news_prediction\archive (4)\News _dataset\True.csv", encoding='ISO-8859-1')

# Combine the datasets
df = pd.concat([fake_df, true_df], ignore_index=True)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = ""
    if request.method == 'POST':
        news_text = request.form['news_text']
        prediction = predict_news(news_text)
    return render_template('index.html', prediction=prediction)

def predict_news(news_text):
    # Check if the news text is present in the dataset
    if news_text not in df['text'].values:
        return "The data is not available in the dataset. Please include the data in the dataset."
    
    # Transform the news text using the vectorizer
    news_tfidf = vectorizer.transform([news_text])
    # Predict using the model
    prediction = model.predict(news_tfidf)
    return "Real News" if prediction[0] == 1 else "Fake News"

if __name__ == "__main__":
    app.run(debug=True)
