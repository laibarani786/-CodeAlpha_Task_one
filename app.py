# ===============================================
# 🏦 Smart Banking FAQ Chatbot (Premium Dark Edition)
# Enhanced by AI ✨
# ===============================================

import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# 🧠 NLTK Setup (Safe Download)
# -------------------------------
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# -------------------------------
# 📘 Banking FAQs (Enhanced Answers)
# -------------------------------
faqs = {
    "How can I reset my online banking password?": 
    "You can reset your password by visiting the login page and selecting ‘Forgot Password’. Follow the secure steps sent to your registered email or phone to create a new password safely.",
    
    "How can I check my account balance?": 
    "You can instantly check your account balance through our mobile app, online banking portal, or at any nearby ATM using your debit card.",
    
    "How do I block my lost debit or credit card?": 
    "Please contact our 24/7 customer helpline immediately or log in to the mobile app to block your card. This helps prevent any unauthorized use or fraudulent activity on your account.",
    
    "What are the interest rates for savings accounts?": 
    "Our savings account interest rates vary by tier and account type. Visit our official website or nearest branch to view the latest rates and offers.",
    
    "How can I apply for a personal loan?": 
    "You can conveniently apply for a personal loan through our website or mobile app by submitting your details and required documents. Approval is quick and fully digital.",
    
    "How do I update my registered mobile number or email?": 
    "Log in to your online banking account, go to ‘Profile Settings’, and update your registered contact details in a few easy steps.",
    
    "What is the process for international money transfer?": 
    "You can transfer money abroad via online banking or by visiting your nearest branch. Transfer fees and delivery times vary based on the country and currency.",
    
    "How do I set up automatic bill payments?": 
    "Go to ‘Bill Pay’ in your online banking portal and enable automatic payments for recurring bills like utilities, credit cards, or subscriptions.",
    
    "Can I open a fixed deposit online?": 
    "Yes! You can open a fixed deposit in just minutes through our mobile app or online banking dashboard — with flexible tenures and competitive interest rates.",
    
    "How do I contact customer support?": 
    "Our dedicated support team is available 24/7 via live chat, helpline, and email. Visit our official website for all contact details."
}

faq_questions = list(faqs.keys())
faq_answers = list(faqs.values())

# -------------------------------
# 🧹 Preprocessing
# -------------------------------
def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stopwords.words('english') and t not in string.punctuation]
    return " ".join(tokens)

processed_questions = [preprocess(q) for q in faq_questions]

# -------------------------------
# 🔢 TF-IDF + Cosine Similarity
# -------------------------------
vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(processed_questions)

def get_best_answer(user_query):
    user_query_processed = preprocess(user_query)
    user_vector = vectorizer.transform([user_query_processed])
    similarity = cosine_similarity(user_vector, faq_vectors)
    best_index = similarity.argmax()
    best_score = similarity[0, best_index]

    if best_score < 0.3:
        return "🤔 I'm not completely sure about that. Could you please rephrase or ask in a different way?"
    else:
        answer = faq_answers[best_index]
        return f"💡 {answer} If you need more details, I’d be happy to help further."

# -------------------------------
# 💬 Streamlit UI
# -------------------------------
st.set_page_config(page_title="Banking FAQ Chatbot", page_icon="🏦", layout="wide")

# 🌙 Dark Theme Styling
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #0d1117;
    color: #f0f6fc;
}
h1, h2, h3, p, label {
    color: #f0f6fc !important;
}
.chat-bubble {
    border-radius: 16px;
    padding: 14px 18px;
    margin: 10px 0;
    max-width: 85%;
    font-size: 17px;
    line-height: 1.5;
}
.user-bubble {
    background-color: #1f6feb;
    color: white;
    text-align: right;
    margin-left: auto;
    box-shadow: 0 4px 12px rgba(31,111,235,0.3);
}
.bot-bubble {
    background-color: #161b22;
    color: #f0f6fc;
    text-align: left;
    margin-right: auto;
    box-shadow: 0 4px 10px rgba(0,0,0,0.4);
    border: 1px solid #30363d;
}
input[type="text"] {
    background-color: #161b22 !important;
    color: #f0f6fc !important;
    border: 1px solid #30363d !important;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# 🏦 Header
st.markdown("<h1 style='text-align:center;'>🏦 Smart Banking FAQ Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>💬 Ask me anything about our banking services, security, or policies</p>", unsafe_allow_html=True)
st.markdown("---")

# Initialize Chat
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# User Input
user_input = st.text_input("💬 Type your question here:")

if user_input:
    response = get_best_answer(user_input)
    st.session_state["chat_history"].append(("🧑‍💻 You", user_input))
    st.session_state["chat_history"].append(("🤖 Bot", response))

# Display Conversation
for role, message in st.session_state["chat_history"]:
    if role == "🧑‍💻 You":
        st.markdown(f"<div class='chat-bubble user-bubble'><b>{role}:</b> {message}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-bubble bot-bubble'><b>{role}:</b> {message}</div>", unsafe_allow_html=True)
