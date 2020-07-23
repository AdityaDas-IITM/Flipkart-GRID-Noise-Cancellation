import numpy as np
import pandas as pd
import os
import librosa
import warnings
warnings.filterwarnings('ignore')

## Loading the original files
original_filepath = '../input/flipkart-grid-20-round-3-original'
files = librosa.util.find_files(original_filepath)

original_audio = pd.DataFrame()

for f in files:
    audio = librosa.load(f)
    #newpoint  = pd.DataFrame({'File_Path':f.split('/')[-1], 'Audio_Array':audio[0]})
    #original_audio = original_audio.append(newpoint)
    librosa.output.write_wav(f.split('/')[-1], np.asarray(audio[0]), sr = 22050)