import tensorflow as tf
import librosa
import numpy as np
import os

def load_model():
    model = tf.keras.models.load_model('D:/Github Repos/Flipkart-GRID-Noise-Cancellation2/FlaskAPI/Model/gbl_model.h5', compile=False)
    return model

def inputProcess(filepath, A=2000, L=110):
    arr, _ = librosa.load(filepath, sr=22000)
    arr_pad = np.pad(arr, (0, A*L - len(arr)), 'constant', constant_values=(0,0))
    arr_reshaped = arr_pad.reshape(1, A, L, 1)
    arr_pad = np.reshape(arr_pad, (1, -1))

    return arr_reshaped

def wavCreator(path, arr):
    arr = np.array(arr).T
    librosa.output.write_wav(path, arr, sr=22000)


