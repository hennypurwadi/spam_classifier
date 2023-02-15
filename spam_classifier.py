
import numpy as np
import pandas as pd
import csv
import joblib
import streamlit as st
import unicodedata
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import *
from sklearn.preprocessing import LabelEncoder

def clean_text(text):
    import re
    from string import punctuation
    text=re.sub(r'(http|ftp|https):\/\/([\w\-_]+(?:(?:\.[\w\-_]+)+))([\w\-\.,@?^=%&:/~\+#]*[\w\-\@?^=%&/~\+#])?', 
                ' ', text)
    text=re.sub(r'['+punctuation+']',' ',text)
    text=re.sub(r'#(\w+)',' ',text)
    text=re.sub(r'@(\w+)',' ',text)
    text = text.lower() # Convert  to lowercase

    token=RegexpTokenizer(r'\w+')
    tokens = token.tokenize(text)

    lemmatizer = WordNetLemmatizer()
    stems = [lemmatizer.lemmatize(t) for t in tokens]
    stemmer = PorterStemmer()
    stems = [stemmer.stem(t) for t in stems]
    
    return ' '.join(stems)

def tokenize(text):
    token=RegexpTokenizer(r'\w+')
    tokens = token.tokenize(text)
    
    return tokens    

def load_models():     
    # Load the vectorizer.
    file = open('vectorizer.pkl', 'rb')
    vectorizer = joblib.load(file)
    file.close()
    
    # Load the LR Model.
    file = open('model.pkl', 'rb')
    model = joblib.load(file)
    file.close()
    
    return vectorizer, model

df = pd.read_csv('SMSSpamColl.csv', encoding='utf-8')
cv=TfidfVectorizer(lowercase=True,preprocessor=clean_text,stop_words='english',ngram_range=(1,3),tokenizer=tokenize)
text_counts=cv.fit_transform(df['text'].values.astype('U'))
x_train, x_test, y_train, y_test = train_test_split(text_counts,df['label'],test_size=0.3)

def main():    
    st.title("Spam Classifier")
    st.write("Enter your message to check if it's spam or not.")
    user_input = st.text_input("Enter message here:")
    
    if st.button("Check"):                  
    # Make predictions
        vectorizer, model = load_models()
        model.fit(x_train, y_train) 
        clean_input = clean_text(user_input)
        input_counts = vectorizer.transform([clean_input])
        prediction = model.predict(input_counts)[0]
        st.write("Prediction: ", prediction)

if __name__ == '__main__':
    main()
