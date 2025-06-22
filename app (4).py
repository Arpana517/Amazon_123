import streamlit as st
import pickle
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import os

# Download necessary NLTK data
nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Function to preprocess text
def preprocess_text(text):
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if not word in STOPWORDS]
    return ' '.join(review)

# Load the CountVectorizer, Scaler, and Model
# Ensure these files exist in the 'Models' directory
try:
    cv = pickle.load(open('Models/countVectorizer.pkl', 'rb'))
    scaler = pickle.load(open('Models/scaler.pkl', 'rb'))
    # Assuming the model saved was a RandomForestClassifier
    model = pickle.load(open('Models/random_forest_model.pkl', 'rb')) # Assuming you saved your model as 'random_forest_model.pkl'
except FileNotFoundError:
    st.error("Error: Model files not found. Please ensure 'countVectorizer.pkl', 'scaler.pkl', and 'random_forest_model.pkl' are in the 'Models' directory.")
    st.stop() # Stop the app if files are not found

st.title("Amazon Alexa Review Sentiment Analysis")
st.write("Enter a review to predict its sentiment (Positive/Negative).")

# Text input from user
user_input = st.text_area("Enter your review here:")

if st.button("Predict Sentiment"):
    if user_input:
        # Preprocess the input
        processed_input = preprocess_text(user_input)

        # Vectorize the input using the loaded CountVectorizer
        # Need to handle potential new words not in the original vocabulary
        input_vector = cv.transform([processed_input]).toarray()

        # Scale the vectorized input using the loaded Scaler
        input_scaled = scaler.transform(input_vector)

        # Make prediction
        prediction = model.predict(input_scaled)

        # Display prediction
        if prediction[0] == 1:
            st.success("Prediction: Positive Feedback")
        else:
            st.error("Prediction: Negative Feedback")
    else:
        st.warning("Please enter a review to predict.")
