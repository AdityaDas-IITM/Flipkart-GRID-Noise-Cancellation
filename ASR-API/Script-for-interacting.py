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

folder_path = 'D:\Github Repos\Flipkart-GRID-Noise-Cancellation2\ASR-API\Outputs\\'
for music in os.listdir(folder_path):

    files = {'audio_file' : open(folder_path + music,'rb')}
    res = requests.post(url, headers = headers, data = data, files = files)
    content = (res.json()['transcriptions'])[0]['utf_text']
    filenames.append(music)
    transcriptions.append(content)

datafile['Filename'] = filenames
datafile['Transcriptions'] = transcriptions
datafile.to_excel('D:\Github Repos\Flipkart-GRID-Noise-Cancellation2\ASR-API\\transcriptions.xlsx', index = False)
