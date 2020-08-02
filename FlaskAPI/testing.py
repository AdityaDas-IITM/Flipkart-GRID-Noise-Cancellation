import requests

#headers = 
data = {'path' : "C:/Users/nihal/Downloads/FlipkartNoiseCancel/Round3/Key Guidelines and Audio Recording Updated/Flipkart-GRID-Noise-Cancellation/"}
files = {'file' : open('C:/Users/nihal/Downloads/FlipkartNoiseCancel/Round3/Key Guidelines and Audio Recording Updated/Flipkart-GRID-Noise-Cancellation/WAV-Inputs/69.wav', 'rb')}
url = 'http://127.0.0.1:5000/predict'
res = requests.post(url, data = data)
print(res)