import nltk
import os
import streamlit as st
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from PIL import Image
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB

import string

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [i for i in text if i.isalnum()]
    text = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]
    text = [ps.stem(i) for i in text]
    return " ".join(text)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
abc = pickle.load(open('model.pkl', 'rb'))

st.title('SMS Spam Classifier')

input_sms = st.text_area('Enter the message')

if st.button('Predict'):
    # 1. Preprocess
    transformed_sms = transform_text(input_sms)
    # 2. Vectorize
    X = tfidf.transform([transformed_sms])
    # 3. Predict
    y = abc.predict(X)[0]
    # 4. Display
    col1 = st.columns(1)
    if y == 0:
        st.success('Not Spam 🎉')
    else:
        st.error('Spam 🚫')
