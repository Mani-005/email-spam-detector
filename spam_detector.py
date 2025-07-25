# spam_detector.py

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle

# Download stopwords
nltk.download('stopwords')

# 1. Load data
df = pd.read_csv('data/spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']

# 2. Label encoding
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# 3. Preprocessing
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = re.sub(r'\W', ' ', text)          # Remove punctuation
    text = text.lower()                      # Lowercase
    text = text.split()                      # Tokenize
    text = [stemmer.stem(word) for word in text if word not in stop_words]
    return ' '.join(text)

df['cleaned'] = df['message'].apply(preprocess)

# 4. TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df['cleaned']).toarray()
y = df['label_num']

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# 7. Accuracy
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# 8. Save model and vectorizer
pickle.dump(model, open('model/spam_model.pkl', 'wb'))
pickle.dump(vectorizer, open('model/vectorizer.pkl', 'wb'))
