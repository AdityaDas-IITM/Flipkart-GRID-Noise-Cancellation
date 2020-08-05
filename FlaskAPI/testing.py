import requests
import json
from GUI import smallGUI
import os

headers = {'Content-type' : 'application/json'}
url = 'http://127.0.0.1:5000/predict'
#data = {'input_path' : 'D:\\Github Repos\\Flipkart-GRID-Noise-Cancellation2\\WAV-Inputs\\71.wav', 'output_path' : 'D:\\Github Repos\\Flipkart-GRID-Noise-Cancellation2\\FlaskAPI\\'}
guiobj = smallGUI()
inputdir = guiobj.inputdirpath + '/'
outputdir = guiobj.outputdirpath + '/'


for music in os.listdir(inputdir):
    data = {'input_path' : inputdir + music, 'output_path' : outputdir + 'pred_'+music}
    res = requests.post(url, headers=headers, json=data)
    print(res)
