## Imports
import librosa
import numpy as np

def inputProcess(path, A, L):
    arr, _ = librosa.load(path, sr=22000)
    arr_pad = np.pad(arr, (0, A*L - len(arr)), 'constant', constant_values=(0,0))
    arr_reshaped = arr_pad.reshape(1, A, L, 1)
    arr_pad = np.reshape(arr_pad, (1, -1))

    return arr_pad, arr_reshaped

def wavCreator(path, arr):
    librosa.output.write_wav(path, arr, sr=22000)