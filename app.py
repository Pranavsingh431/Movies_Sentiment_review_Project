# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Load word index and model
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}
model = load_model('simple_RNN_imdb.h5')

# Function to decode review
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# --- Streamlit App UI ---
st.markdown(
    """
    <style>
    body {
        background-color: #f4f4f4;
        color: #333;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 12px 20px;
        font-size: 16px;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        color: #333;
    }
    .info-text {
        font-size: 18px;
        text-align: center;
        color: #666;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App Title
st.markdown('<p class="title">🎬 IMDB Movie Review Sentiment Analysis 🎭</p>', unsafe_allow_html=True)
st.markdown('<p class="info-text">Enter a movie review to classify it as Positive or Negative.</p>', unsafe_allow_html=True)

# User Input
user_input = st.text_area("✍️ Enter your movie review below:")

# Classify Button
if st.button('🚀 Classify Sentiment'):

    preprocessed_input = preprocess_text(user_input)

    # Make prediction
    prediction = model.predict(preprocessed_input)
    sentiment = '😊 Positive' if prediction[0][0] > 0.7 else '😞 Negative'

    # Display Results with Formatting
    st.markdown(
        f"""
        <div style="border-radius:10px; padding:15px; background-color:#fafafa; text-align:center; font-size:22px;">
            <b>Predicted Sentiment:</b> <span style="color:#4CAF50;">{sentiment}</span> <br>
            <b>Confidence Score:</b> <span style="color:#FF5733;">{prediction[0][0]:.4f}</span>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.info('⌨️ Please enter a movie review and click "Classify Sentiment".')
