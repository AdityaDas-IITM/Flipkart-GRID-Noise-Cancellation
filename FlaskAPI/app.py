# -*- coding: utf-8 -*-
"""
Script for creating and loading contents to the server
"""
import flask
from flask import Flask, jsonify, request
import json
import tensorflow as tf
import librosa
import numpy as np

def load_model():
    model = tf.keras.models.load_model('gbl_model.h5')
    return model

def inputProcess(file, A=2000, L=110):
    arr, _ = librosa.load(file, sr=22000)
    arr_pad = np.pad(arr, (0, A*L - len(arr)), 'constant', constant_values=(0,0))
    arr_reshaped = arr_pad.reshape(1, A, L, 1)
    arr_pad = np.reshape(arr_pad, (1, -1))

    return arr_reshaped

def wavCreator(path, arr):
    arr = np.array(arr).T
    librosa.output.write_wav(path, arr, sr=22000)

app = Flask(__name__)

@app.route('/predict', methods = ['GET', 'POST'])
def predict():
    print("Request: ", request)
    request_json = request.

    file = request_json['file']
    path = request_json['path']

    arr_reshaped = inputProcess(file)

    model = load_model()
    denoised_arr = model.predict([arr_reshaped, np.zeros((1, 2000*110))])

    wavCreator(path, denoised_arr)
    
    response = json.dumps({1:2})

    return response, 200

if __name__ == "__main__":
    app.run(debug=True)
    