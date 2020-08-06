# -*- coding: utf-8 -*-
"""
Script for interacting with the flipkart ASR API
"""
import requests
import os
import pandas as pd

datafile = pd.DataFrame()
filenames = []
transcriptions = []
headers = {'Authorization' : 'Token 3715119fd7753d33bedbd3c2832752ee7b0a10c7'}
data = {'user' : '310' ,'language' : 'HI'}
url = 'https://dev.liv.ai/liv_transcription_api/recordings/'

folder_path = 'C:/Users/nihal/Downloads/FlipkartNoiseCancel/Round3/Key Guidelines and Audio Recording Updated/Flipkart-GRID-Noise-Cancellation/predictions/'
for music in sorted(os.listdir(folder_path)):
    print(music)
    '''
    files = {'audio_file' : open(folder_path + music,'rb')}
    res = requests.post(url, headers = headers, data = data, files = files)
    content = (res.json()['transcriptions'])[0]['utf_text']
    filenames.append(music)
    transcriptions.append(content)

datafile['Filename'] = filenames
datafile['ASR'] = transcriptions
datafile.to_excel('C:/Users/nihal/Downloads/FlipkartNoiseCancel/Round3/Key Guidelines and Audio Recording Updated/Flipkart-GRID-Noise-Cancellation/ASR-API/ASRTranscriptions.xlsx', index = False)

ASR_data = pd.read_excel("C:/Users/nihal/Downloads/FlipkartNoiseCancel/Round3/Key Guidelines and Audio Recording Updated/Flipkart-GRID-Noise-Cancellation/ASR-API/ASRTranscriptions.xlsx")
true_data = pd.read_excel("C:/Users/nihal/Downloads/FlipkartNoiseCancel/Round3/Key Guidelines and Audio Recording Updated/Flipkart-GRID-Noise-Cancellation/ASR-API/TrueTranscriptions.xlsx")

all_data = pd.concat([ASR_data['Filename'], true_data['Transcription '] , ASR_data['ASR']], axis=1)
all_data.to_excel('C:/Users/nihal/Downloads/FlipkartNoiseCancel/Round3/Key Guidelines and Audio Recording Updated/Flipkart-GRID-Noise-Cancellation/ASR-API/AllData.xlsx', index = False)
print(len(all_data))
'''