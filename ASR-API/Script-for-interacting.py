# -*- coding: utf-8 -*-
"""
Script for interacting with the flipkart ASR API
"""
import requests

headers = {'Authorization' : 'Token 3715119fd7753d33bedbd3c2832752ee7b0a10c7'}
data = {'user' : '310' ,'language' : 'HI'}
files = {'audio_file' : open('C:/Projects/Flipkart-GRID-Noise-Cancellation/Audio-Recordings/259.mp3','rb')}
url = 'https://dev.liv.ai/liv_transcription_api/recordings/'
res = requests.post(url, headers = headers, data = data, files = files)
content = res.json()['transcriptions']