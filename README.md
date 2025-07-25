# 📧 Email Spam Detector 🔍

A powerful machine learning-based web application that detects whether an email is **Spam** or **Not Spam**, built using **Python**, **Scikit-learn**, and **Streamlit**.

> 🚀 **Live Demo**: [Click here to try it out](https://your-streamlit-link.streamlit.app)  
> *(Replace with your actual Streamlit Cloud link)*

---

## 📌 Features

- ✅ Real-time spam prediction
- 🧠 Trained using Multinomial Naive Bayes
- 🔠 TF-IDF vectorization for text preprocessing
- 🧹 NLP: tokenization, stopword removal, stemming
- 📊 User-friendly Streamlit UI
- 💬 Voice input & prediction history (optional features)

---

## 🧠 How It Works

1. Email text is processed and vectorized using TF-IDF.
2. A Naive Bayes model predicts if it’s spam or not.
3. Streamlit displays the result interactively.

---

## 🛠️ Tech Stack

- Python
- Scikit-learn
- Pandas, NumPy
- NLTK
- Streamlit

---

## 📂 Project Structure

email-spam-detector/
│
├── app.py # Main Streamlit application
├── train_model.py # Model training script
├── model.pkl # Trained ML model
├── vectorizer.pkl # TF-IDF vectorizer
├── requirements.txt # Python dependencies
└── README.md # Project overview


---

## 🧪 Sample Emails

### ✅ Not Spam
Subject: Meeting Reminder

Hi Team,
This is a reminder for today’s 4 PM meeting. Please join on time.
Regards,
Manager

### 🚫 Spam
Subject: You've Won a Free iPhone!

Congratulations! Click the link to claim your iPhone now before it expires!
Offer valid only for 24 hours.



---

## 🚀 Run the App Locally

```bash
# Clone the repo
git clone https://github.com/your-github-username/email-spam-detector.git
cd email-spam-detector

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

---

You're all set! ✅ Let me know if you want:

- A project banner/cover image for GitHub
- A YouTube/LinkedIn post caption
- A PDF report or resume project section writeup for this

Happy deploying!
