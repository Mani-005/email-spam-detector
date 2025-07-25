import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import pickle

# Sample dataset — You can replace this with a larger spam dataset
data = {
    'text': [
        'Congratulations! You have won a $1000 Walmart gift card. Click here to claim now.',
        'Hi, please find attached the project files for your review.',
        'Lowest price luxury watches available NOW. Click to buy!',
        'Reminder: your appointment is scheduled for tomorrow at 3 PM.',
        'You have been selected for a free vacation. Call now to book!',
        'Let me know if you’re free to discuss tomorrow’s meeting agenda.'
    ],
    'label': [1, 0, 1, 0, 1, 0]  # 1 = spam, 0 = ham
}

df = pd.DataFrame(data)

# Step 1: Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])

# Step 2: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, df['label'], test_size=0.2, random_state=42)

# Step 3: Train classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 4: Save vectorizer and model
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model and vectorizer saved successfully!")
