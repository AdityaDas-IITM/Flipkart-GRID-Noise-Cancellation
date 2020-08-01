# -*- coding: utf-8 -*-
"""
Script for creating and loading contents to the server
"""
import flask
from flask import Flask, jsonify, request
import json
import tensorflow as tf
#import numpy as np
#from tensorflow.keras.models import Sequential, Model
#from tensorflow.keras.layers import Dense, Conv1D, BatchNormalization, Dropout, LeakyReLU, Input, Flatten, Multiply, Conv2D, Reshape, PReLU, Add
#import keras.backend as K

def load_model():
    model = tf.keras.models.load_model('gbl_model.h5')
    return model

app = Flask(__name__)

@app.route('/predict', methods = ['GET'])
def predict():
    request_json = request.get_json()
    x = request_json['input']
    