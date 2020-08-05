import requests
import json

headers = {'Content-type' : 'application/json'}
data = {'input_path' : 'D:\\Github Repos\\Flipkart-GRID-Noise-Cancellation2\\WAV-Inputs\\71.wav', 'output_path' : 'D:\\Github Repos\\Flipkart-GRID-Noise-Cancellation2\\FlaskAPI\\'}

url = 'http://127.0.0.1:5000/predict'
res = requests.post(url, headers=headers, json=data)
print(res)

#open(data["input_path"], "rb")