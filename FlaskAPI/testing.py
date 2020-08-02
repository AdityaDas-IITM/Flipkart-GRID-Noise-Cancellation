import requests
import json

headers = {'Content-type' : 'application/json'}
data = {'input_path' : 'C:/Users/nihal/Downloads/FlipkartNoiseCancel/Round3/Key Guidelines and Audio Recording Updated/Flipkart-GRID-Noise-Cancellation/WAV-Inputs/69.wav', 'output_path' : "C:/Users/nihal/Downloads/FlipkartNoiseCancel/Round3/Key Guidelines and Audio Recording Updated/Flipkart-GRID-Noise-Cancellation/"}
url = 'http://127.0.0.1:5000/predict'
res = requests.post(url, headers=headers, json=data)
print(res)