import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Download NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Load dataset (use a larger subset of the data)
df = pd.read_csv('tweets.csv', encoding='latin-1', names=['sentiment', 'id', 'date', 'query', 'user', 'text'], nrows=900000)  # Modified: Load 900,000 rows to include both classes

# Map sentiment: 0 = negative, 4 = positive, else neutral
df['sentiment'] = df['sentiment'].map({0: 'negative', 4: 'positive'}).fillna('neutral')

# Check the class distribution
print("Class distribution in the dataset:")
print(df['sentiment'].value_counts())

# Clean text
stop_words = set(stopwords.words('english'))
def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'@\w+|\#', '', text)  # Remove mentions and hashtags
    text = text.lower()  # Lowercase
    tokens = word_tokenize(text)  # Tokenize
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]  # Remove stopwords
    return ' '.join(tokens)

df['clean_text'] = df['text'].apply(clean_text)

# Convert text to numerical features (keep it sparse)
tfidf = TfidfVectorizer(max_features=2000)
X = tfidf.fit_transform(df['clean_text'])  # Sparse matrix
y = df['sentiment']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model and vectorizer
pickle.dump(model, open('sentiment_model.pkl', 'wb'))
pickle.dump(tfidf, open('tfidf_vectorizer.pkl', 'wb'))
