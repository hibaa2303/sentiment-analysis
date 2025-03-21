from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('sentiment_model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

def clean_text(text):
    import re
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|\#', '', text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(tokens)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    cleaned_text = clean_text(text)
    vectorized_text = tfidf.transform([cleaned_text])
    prediction = model.predict(vectorized_text)[0]
    # Get probability scores
    probabilities = model.predict_proba(vectorized_text)[0]
    prob_dict = {model.classes_[i]: round(prob * 100, 2) for i, prob in enumerate(probabilities)}
    return render_template('index.html', prediction=prediction, text=text, probabilities=prob_dict)

if __name__ == '__main__':
    app.run(debug=True)
