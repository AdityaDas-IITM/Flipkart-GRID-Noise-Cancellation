# %% [code]
## Imports
import librosa
import os
import numpy as np

def initialize_counter():
    global counter1
    global counter2
    counter1 = 0
    counter2 = 0

def inputProcess1(path, A, L, batch_size):
    global counter1
    print(counter1)
    file_paths = os.listdir(path)
    try:
        required_paths = file_paths[counter1: counter1+batch_size]
    except:
        required_paths = file_paths[counter1:]
    arr_pad = []
    arr_reshaped = []
    for f in required_paths:
        single_arr, _ = librosa.load(path + f, sr=22000)
        single_arr_pad = np.pad(single_arr, (0, A*L - len(single_arr)), 'constant', constant_values=(0,0))
        single_arr_reshaped = single_arr_pad.reshape(1, A, L, 1)
        #single_arr_reshaped = np.expand_dims(single_arr_reshaped, axis=0)
        #print(single_arr_reshaped.shape)
        single_arr_pad = np.reshape(single_arr_pad, (1, -1))
        arr_pad.append(single_arr_pad)
        arr_reshaped.append(single_arr_reshaped)
    actual_batch_size = len(required_paths)
    counter1 = counter1 + actual_batch_size
    arr_reshaped = np.array(arr_reshaped).reshape(actual_batch_size,A,L,1)
    arr_pad = np.array(arr_pad).reshape(actual_batch_size, A*L)
    return arr_pad, arr_reshaped, actual_batch_size

def inputProcess2(path, A, L, batch_size):
    global counter2
    print(counter2)
    file_paths = os.listdir(path)
    try:
        required_paths = file_paths[counter2: counter2+batch_size]
    except:
        required_paths = file_paths[counter2:]
    arr_pad = []
    arr_reshaped = []
    for f in required_paths:
        single_arr, _ = librosa.load(path + f, sr=22000)
        single_arr_pad = np.pad(single_arr, (0, A*L - len(single_arr)), 'constant', constant_values=(0,0))
        single_arr_reshaped = single_arr_pad.reshape(1, A, L, 1)
        #single_arr_reshaped = np.expand_dims(single_arr_reshaped, axis=0)
        #print(single_arr_reshaped.shape)
        single_arr_pad = np.reshape(single_arr_pad, (1, -1))
        arr_pad.append(single_arr_pad)
        arr_reshaped.append(single_arr_reshaped)
    actual_batch_size = len(required_paths)
    counter2 = counter2 + actual_batch_size
    arr_reshaped = np.array(arr_reshaped).reshape(actual_batch_size,A,L,1)
    arr_pad = np.array(arr_pad).reshape(actual_batch_size, A*L)
    return arr_pad, arr_reshaped, actual_batch_size

def inputProcesstest(path, A, L):
    single_arr, _ = librosa.load(path, sr=22000)
    single_arr_pad = np.pad(single_arr, (0, A*L - len(single_arr)), 'constant', constant_values=(0,0))
    single_arr_reshaped = single_arr_pad.reshape(1, A, L, 1)
        #single_arr_reshaped = np.expand_dims(single_arr_reshaped, axis=0)
        #print(single_arr_reshaped.shape)
    single_arr_pad = np.reshape(single_arr_pad, (1, -1))
    return single_arr_pad, single_arr_reshaped

def wavCreator(path, arr):
    arr = np.array(arr).T
    librosa.output.write_wav(path, arr, sr=22000)