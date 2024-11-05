import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the data
fake_df = pd.read_csv(r"E:\projects\fake_news_prediction\archive (4)\News _dataset\Fake.csv")  # Use raw string
true_df = pd.read_csv(r"E:\projects\fake_news_prediction\archive (4)\News _dataset\True.csv")  # Use raw string

# Add a label column (0 for fake, 1 for true)
fake_df['label'] = 0
true_df['label'] = 1

# Combine the datasets
df = pd.concat([fake_df, true_df], ignore_index=True)

# Shuffle the dataset to mix fake and true news
df = df.sample(frac=1).reset_index(drop=True)

# Separate features and target variable
X = df['text']  # Adjust this if your news content column name is different
y = df['label']

# Split into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and transform the training data, then transform the test data
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Initialize and train the model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Predict on test data
y_pred = model.predict(X_test_tfidf)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix and Classification Report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Prediction function
def predict_news(news_text):
    # Check if the news text is present in the dataset
    if news_text not in df['text'].values:
        return "The data is not available in the dataset. Please include the data in the dataset."

    # Transform the text to match the trained vectorizer
    news_tfidf = vectorizer.transform([news_text])
    # Predict using the model
    prediction = model.predict(news_tfidf)
    return "Real News" if prediction[0] == 1 else "Fake News"

# Example usage
news_text = "On Christmas day, Donald Trump announced that he would be back to work the following day, but he is golfing for the fourth day in a row..."  # (Your long text here)
result = predict_news(news_text)
print(result)
