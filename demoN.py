from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from textblob import TextBlob

# Sample texts
texts = ["This is a real news article.", "Breaking: Alien invasion confirmed!"]

# Labels: 1 = Real, 0 = Fake (for demo purposes only)
labels = [1, 0]

# Vectorization & Model
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(texts)
model = LogisticRegression().fit(X, labels)

# Prediction demo
for text in texts:
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    sentiment = TextBlob(text).sentiment.polarity
    print(f"\nText: {text}")
    print(f"Prediction: {'Real' if pred else 'Fake'} | Sentiment: {'Positive' if sentiment > 0 else 'Negative' if sentiment < 0 else 'Neutral'}")
