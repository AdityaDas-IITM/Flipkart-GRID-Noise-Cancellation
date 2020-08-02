import librosa
import tensorflow as tf
import numpy as np

def inputProcess(path, A = 2000, L = 110):
    arr, _ = librosa.load(path, sr=22000)
    arr_pad = np.pad(arr, (0, A*L - len(arr)), 'constant', constant_values=(0,0))
    arr_reshaped = arr_pad.reshape(1, A, L, 1)
    arr_pad = np.reshape(arr_pad, (1, -1))

    return arr_pad, arr_reshaped

def wavCreator(path, arr):
    arr = np.array(arr).T
    librosa.output.write_wav(path, arr, sr=22000)

model_path = 'D:/Github Repos/Flipkart-GRID-Noise-Cancellation2/gbl_model.h5'
weights_path = 'D:/Github Repos/Flipkart-GRID-Noise-Cancellation2/weights(3).h5'
model = tf.keras.models.load_model(model_path, compile = False)

input_file_path = input(" Enter input file path")
output_file_path = input(" Enter download path")

arr_pad, arr_reshaped = inputProcess(input_file_path)

predict = model.predict([arr_reshaped, arr_pad])

wavCreator(output_file_path, predict)