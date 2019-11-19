# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 15:29:55 2019

@author: liuyu
"""
#IMPORT
from flask import Flask, request, render_template
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.models import load_model
import pandas as pd 

#DATA
df_v1 = pd.read_json('Sarcasm_Headlines_Dataset.json', lines=True)
df_v2 = pd.read_json('Sarcasm_Headlines_Dataset_v2.json', lines=True)
df = df_v1.append(df_v2, sort = False)
tokenizer = Tokenizer(num_words = 5000, filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower = True, split = ' ')
tokenizer.fit_on_texts(list(df['headline']))

#METHODS
def preprocess(text):
    seq = tokenizer.texts_to_sequences([text])
    seq = pad_sequences(seq, maxlen = 106)
    seq = seq.reshape(1, 106)
    return seq


#WEBAPP
app = Flask(__name__)

@app.route("/", methods =['POST'])
def webapp():
    #MODEL
    global graph
    graph = tf.get_default_graph()
    model = load_model('weights/model.h5')
    with graph.as_default():
        txt = preprocess(request.form['headline'])
        y_pred = model.predict(txt.reshape(1,106),batch_size=1,verbose = 2)
    pred_int = np.argmax(y_pred)
    if pred_int == 1:
        prediction = "SARCASTIC"
    else:
        prediction = "NOT SARCASTIC"
    return render_template('templates.html', prediction = prediction)

@app.route('/', methods=['GET'])
def load():
    return render_template('templates.html', prediction=None)
if __name__ == '__main__':
    app.run(debug=True)
