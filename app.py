'''import streamlit as st
import pandas as pd
import numpy as np
import pickle
import string
import nltk
import smtplib
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import speech_recognition as sr
import pyttsx3
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os

nltk.download('stopwords')
ps = PorterStemmer()

# ========== Helper Functions ========== #

def text_preprocess(text):
    text = text.lower()
    text = "".join([ch for ch in text if ch not in string.punctuation])
    tokens = text.split()
    tokens = [ps.stem(word) for word in tokens if word not in stopwords.words('english')]
    return " ".join(tokens)

def load_model():
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return vectorizer, model

def classify_email(text, vectorizer, model):
    processed = text_preprocess(text)
    vector = vectorizer.transform([processed])
    prediction = model.predict(vector)[0]
    return prediction

def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def get_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Speak Now...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            st.success(f"Recognized: {text}")
            return text
        except:
            st.error("Could not understand audio.")
            return ""

def send_spam_alert(message):
    try:
        sender = "manitejasai123@gmail.com"
        receiver = "zebexred@gmail.com"
        password = "Redzebex123@"  # Use App Password if 2FA enabled

        msg = MIMEMultipart("alternative")
        msg["Subject"] = "🚨 Spam Alert Detected!"
        msg["From"] = sender
        msg["To"] = receiver

        body = f"⚠️ A spam message was detected:\n\n{message}"
        msg.attach(MIMEText(body, "plain"))

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, receiver, msg.as_string())
        server.quit()
        st.success("Email alert sent!")
    except Exception as e:
        st.warning(f"Email not sent. Error: {e}")

def save_history(message, result):
    df = pd.DataFrame([[message, result]], columns=["Message", "Prediction"])
    if os.path.exists("prediction_history.csv"):
        df.to_csv("prediction_history.csv", mode='a', header=False, index=False)
    else:
        df.to_csv("prediction_history.csv", index=False)

def load_history():
    if os.path.exists("prediction_history.csv"):
        return pd.read_csv("prediction_history.csv")
    return pd.DataFrame(columns=["Message", "Prediction"])

def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=300, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

# ========== Streamlit UI ========== #

st.set_page_config(page_title="Email Spam Detector", layout="centered")
st.title("📧 Email Spam Detector with AI 🔍")
st.markdown("---")

vectorizer, model = load_model()

option = st.radio("Choose Input Mode", ["📝 Manual Text", "🎙️ Voice Input"])

if option == "📝 Manual Text":
    email_text = st.text_area("Enter email message:")
else:
    if st.button("🎙️ Start Voice Input"):
        email_text = get_voice_input()
    else:
        email_text = ""

if st.button("🔍 Analyze Email"):
    if email_text.strip() == "":
        st.warning("Please enter or speak a message.")
    else:
        result = classify_email(email_text, vectorizer, model)
        result_label = "🚫 Spam" if result == 1 else "✅ Not Spam"
        result_color = "red" if result == 1 else "green"
        st.markdown(f"### Prediction: <span style='color:{result_color}'>{result_label}</span>", unsafe_allow_html=True)

        # Speak output
        speak_text(result_label)

        # Save prediction history
        save_history(email_text, result_label)

        # Send email alert if spam
        if result == 1:
            send_spam_alert(email_text)

        # WordCloud
        with st.expander("☁️ WordCloud Visualization"):
            generate_wordcloud(email_text)

if st.button("🗑️ Clear"):
    st.rerun()

# ========== History Table ========== #
with st.expander("📜 View Prediction History"):
    history_df = load_history()
    st.dataframe(history_df)'''
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import string
import nltk
import smtplib
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import speech_recognition as sr
import pyttsx3
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os

# Download stopwords if not already present
nltk.download('stopwords', quiet=True)
ps = PorterStemmer()

# ========== Helper Functions ========== #

def text_preprocess(text):
    text = text.lower()
    text = "".join([ch for ch in text if ch not in string.punctuation])
    tokens = text.split()
    tokens = [ps.stem(word) for word in tokens if word not in stopwords.words('english')]
    return " ".join(tokens)

def load_model():
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return vectorizer, model

def classify_email(text, vectorizer, model):
    processed = text_preprocess(text)
    vector = vectorizer.transform([processed])
    prediction = model.predict(vector)[0]
    return prediction

def speak_text(text):
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        st.warning(f"Text-to-speech error: {e}")

def get_voice_input():
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.info("🎤 Listening... Speak now.")
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            st.success(f"Recognized: {text}")
            return text
    except sr.WaitTimeoutError:
        st.error("⏰ No speech detected. Try again.")
    except sr.UnknownValueError:
        st.error("❌ Could not understand the audio.")
    except Exception as e:
        st.error(f"🎙️ Voice input error: {e}")
    return ""

def send_spam_alert(message):
    try:
        sender = "manitejasai123@gmail.com"
        receiver = "zebexred@gmail.com"
        password = "Redzebex123@"  # ⚠️ Consider using a secure app password!

        msg = MIMEMultipart("alternative")
        msg["Subject"] = "🚨 Spam Alert Detected!"
        msg["From"] = sender
        msg["To"] = receiver

        body = f"⚠️ A spam message was detected:\n\n{message}"
        msg.attach(MIMEText(body, "plain"))

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, receiver, msg.as_string())
        server.quit()
        st.success("📧 Email alert sent!")
    except Exception as e:
        st.warning(f"⚠️ Email not sent. Error: {e}")

def save_history(message, result):
    df = pd.DataFrame([[message, result]], columns=["Message", "Prediction"])
    if os.path.exists("prediction_history.csv"):
        df.to_csv("prediction_history.csv", mode='a', header=False, index=False)
    else:
        df.to_csv("prediction_history.csv", index=False)

def load_history():
    if os.path.exists("prediction_history.csv"):
        return pd.read_csv("prediction_history.csv")
    return pd.DataFrame(columns=["Message", "Prediction"])

def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=300, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

# ========== Streamlit UI ========== #

st.set_page_config(page_title="Email Spam Detector", layout="centered")
st.title("📧 Email Spam Detector with AI 🔍")
st.markdown("---")

# Load model and vectorizer
vectorizer, model = load_model()

# Input Mode Selection
option = st.radio("Choose Input Mode", ["📝 Manual Text", "🎙️ Voice Input"])

email_text = ""

if option == "📝 Manual Text":
    email_text = st.text_area("Enter email message:")
else:
    if st.button("🎙️ Start Voice Input"):
        email_text = get_voice_input()

# Analyze Email
if st.button("🔍 Analyze Email"):
    if email_text.strip() == "":
        st.warning("⚠️ Please enter or speak a message.")
    else:
        result = classify_email(email_text, vectorizer, model)
        result_label = "🚫 Spam" if result == 1 else "✅ Not Spam"
        result_color = "red" if result == 1 else "green"

        st.markdown(f"### Prediction: <span style='color:{result_color}'>{result_label}</span>", unsafe_allow_html=True)
        
        # Speak result
        speak_text(result_label)

        # Save history
        save_history(email_text, result_label)

        # Send email alert if spam
        if result == 1:
            send_spam_alert(email_text)

        # WordCloud
        with st.expander("☁️ WordCloud Visualization"):
            generate_wordcloud(email_text)

# Clear Button
if st.button("🗑️ Clear"):
    st.rerun()

# History Table
with st.expander("📜 View Prediction History"):
    history_df = load_history()
    st.dataframe(history_df)


