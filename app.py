import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

ps=PorterStemmer()

def transform_text(text):
    text=text.lower() #1 convert to lowercase
    text=nltk.word_tokenize(text)#2 word tokenization
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    
    text=y[:]
    y.clear()
    
    ps=PorterStemmer()
    for i in text:
        y.append(ps.stem(i))
        
    return " ".join(y)

cv=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

st.title("Email-spam-classifier")
input_text=st.text_area("Enter the email_text")

if st.button('Predict'):
    #now we work as
    #preprocess
    transformed_text=transform_text(input_text)
    #vectorize
    vector_input=cv.transform([transformed_text]).toarray()
    #predict
    result=model.predict(vector_input)
    #display
    if result==1:
        st.header("spam")
    else:
        st.header("Not spam")


