from flask import Flask,render_template,url_for,request

import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import json
import pickle
from keras.backend import clear_session
clear_session()



from sklearn.feature_extraction.text import CountVectorizer #For Bag of words
from sklearn.feature_extraction.text import TfidfVectorizer #For TF-IDF
from gensim.models import Word2Vec                          #For Word2Vec

from bs4 import BeautifulSoup

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score

from keras.models import Sequential, Model, load_model

from keras.callbacks import ModelCheckpoint, EarlyStopping

from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform


with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        model = load_model('model-ngram.h5')
model._make_predict_function()

from keras.layers import Input, Dense, Activation, Dropout, LSTM, Flatten
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.preprocessing import text, sequence
from keras import utils

import itertools
import os

import re
import string


from midi2audio import FluidSynth


app = Flask(__name__)


# def get_model():
# 	global model
# 	model = load_model('model-ngram.h5')
# 	print(" * Model loaded!")

# print(" * Loading Keras model...")
# get_model()


def predict_class(input_x, model):
  y_probs = model.predict(input_x) 
  y_classes = y_probs.argmax(axis=-1)
  return y_probs

uri_re = r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))'

def stripTagsAndUris(x):
    if x:
        # BeautifulSoup on content
        soup = BeautifulSoup(x, "html.parser")
        # Stripping all <code> tags with their content if any
        if soup.code:
            soup.code.decompose()
        # Get all the text out of the html
        text =  soup.get_text()
        # Returning text stripping out all uris
        return re.sub(uri_re, "", text)
    else:
        return ""

def removePunctuation(x):
    # Lowercasing all words
    x = x.lower()
    # Removing non ASCII chars
    x = re.sub(r'[^\x00-\x7f]',r' ',x)
    # Removing (replacing with empty spaces actually) all the punctuations
    return re.sub("["+string.punctuation+"]", " ", x)

snow = nltk.stem.SnowballStemmer('english')
stops = set(stopwords.words("english"))
def stemAndRemoveStopwords(x):
    # Removing all the stopwords
    filtered_words = [snow.stem(word) for word in x.split() if word not in stops]
    return " ".join(filtered_words)
    
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)    
    return input_txt
  
def preprocess(df):
    df["content"] = np.vectorize(remove_pattern)(df["content"], "@[\w]*")
    df["content"] = df["content"].map(stripTagsAndUris)
    df["content"] = df["content"].map(removePunctuation)
    df["content"] = df["content"].map(stemAndRemoveStopwords)
    return df
  
def insert_text(input_text, dataframe):
    dataframe = dataframe.append({'content' : input_text}, ignore_index=True)
    return dataframe
	

def TextProcessing(df):

	if request.method == 'POST':
		InputText = request.form['InputText']
		# df = pd.DataFrame(columns=['content', 'sentiment'])
		print(InputText)
		df = insert_text(InputText, df)
		df = preprocess(df)
		
	return df

# @app.route("/play", methods=["POST"])
# def playPage():
# 	if request.method == 'POST':
# 		if request.form['nextButton'] == 'nextOne':
# 			return render_template('FYP_FINAL_PAGE.html') 


@app.route("/play", methods=["POST"])
def play():

	fs = FluidSynth()
	#fs.midi_to_audio('sample1.mid', 'output.wav')



	if request.method == 'POST':
		return render_template('FYP_FINAL_PAGE.html')
		# if request.form['nextButton'] == 'Next':
		# 	print("haha")




@app.route("/predict", methods=["POST"])
def predict():
	df = pd.DataFrame(columns=['content', 'sentiment'])

	df = TextProcessing(df)
	test_text = df['content']
	picklefile = open("tfidf.pickle", 'rb')
	tfidfvectorizer = pickle.load(picklefile)

	x_text = tfidfvectorizer.transform(test_text.values.astype('U')).astype('float32')

	predictions = np.argmax(predict_class(x_text, model), axis = 1)
	print(predictions)
	


	if request.method == 'POST':
		if request.form['nameSubmit'] == 'Submit':
			InputText = request.form['InputText']
			return render_template('FYP_Result_page.html', prediction = predictions)

		# if request.form['nextButton'] == 'Next':
		# 	print("haha")
		# 	return render_template('FYP_FINAL_PAGE.html') 


	# if request.method == 'POST':

	


# def playPage():
# 	if request.method == 'POST':
# 		if request.form['nextButton'] == 'nextOne':
# 			return render_template('FYP_FINAL_PAGE.html') 



@app.route('/')
def FYPWebpage():
	return render_template('FYPWebpage.html')

if __name__ == '__main__':
	app.run(debug=True)