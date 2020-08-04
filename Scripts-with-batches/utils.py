# %% [code]
## Imports
import librosa
import os
import numpy as np
from sklearn.model_selection import train_test_split

def initialize_counter(train_path, target_path):
    global counter
    global file_paths
    global train_files
    global valid_files
    
    train_files = np.sort(os.listdir(train_path))
    target_files = np.sort(os.listdir(target_path))
    
    file_paths = {}
    
    for i in range(len(train_files)):
        file_paths[train_files[i]] = target_files[i]
    
    valid_files = train_files[-200:]
    train_files = train_files[:-200]
    
    np.random.shuffle(train_files)

    counter = 0

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

def get_validation_set(train_path, target_path, A, L):
    global valid_files
    global file_paths
    
    arr_pad_valid = []
    arr_pad_valid_target = []
    arr_reshaped_valid = []
    
    for f in valid_files:
        #print(f, file_paths[f])
        single_arr_valid, _ = librosa.load(train_path + f , sr=22000)
        single_arr_pad_valid = np.pad(single_arr_valid, (0, A*L - len(single_arr_valid)), 'constant', constant_values=(0,0))
        single_arr_reshaped_valid = single_arr_pad_valid.reshape(1, A, L, 1)
        
        single_arr_target_valid, _ = librosa.load(target_path + file_paths[f] , sr=22000)
        single_arr_pad_target_valid = np.pad(single_arr_target_valid, (0, A*L - len(single_arr_target_valid)), 'constant', constant_values=(0,0))
        
        single_arr_pad_valid = np.reshape(single_arr_pad_valid, (1, -1))
        arr_pad_valid.append(single_arr_pad_valid)
        
        single_arr_pad_target_valid = np.reshape(single_arr_pad_target_valid, (1, -1))
        arr_pad_valid_target.append(single_arr_pad_target_valid)
        
        arr_reshaped_valid.append(single_arr_reshaped_valid)
        
    arr_reshaped_valid = np.array(arr_reshaped_valid).reshape(200,A,L,1)
    
    arr_pad_valid = np.array(arr_pad_valid).reshape(200, A*L)
    arr_pad_valid_target = np.array(arr_pad_valid_target).reshape(200, A*L)
    
    return arr_pad_valid, arr_pad_valid_target, arr_reshaped_valid

def inputProcess(train_path,target_path, A, L, batch_size):
    global counter
    global file_paths
    global train_files
    
    required_train = train_files[counter:counter+batch_size]
     
    arr_pad_train = []
    arr_pad_target = []
    arr_reshaped = []
    
    for f in required_train:
        #print(f, file_paths[f])
        single_arr_train, _ = librosa.load(train_path + f , sr=22000)
        single_arr_pad_train = np.pad(single_arr_train, (0, A*L - len(single_arr_train)), 'constant', constant_values=(0,0))
        single_arr_reshaped = single_arr_pad_train.reshape(1, A, L, 1)
        
        single_arr_target, _ = librosa.load(target_path + file_paths[f] , sr=22000)
        single_arr_pad_target = np.pad(single_arr_target, (0, A*L - len(single_arr_target)), 'constant', constant_values=(0,0))
        #single_arr_reshaped = single_arr_pad_target.reshape(1, A, L, 1)
        
        single_arr_pad_train = np.reshape(single_arr_pad_train, (1, -1))
        arr_pad_train.append(single_arr_pad_train)
        
        single_arr_pad_target = np.reshape(single_arr_pad_target, (1, -1))
        arr_pad_target.append(single_arr_pad_target)
        
        arr_reshaped.append(single_arr_reshaped)
        
    actual_batch_size = len(required_train)
    counter = counter + actual_batch_size
    arr_reshaped = np.array(arr_reshaped).reshape(actual_batch_size,A,L,1)
    
    arr_pad_train = np.array(arr_pad_train).reshape(actual_batch_size, A*L)
    arr_pad_target = np.array(arr_pad_target).reshape(actual_batch_size, A*L)
    
    return arr_pad_train, arr_pad_target, arr_reshaped, actual_batch_size