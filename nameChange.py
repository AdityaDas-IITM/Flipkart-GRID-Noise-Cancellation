import os
import shutil

path1 = "D:/Github Repos/Flipkart-GRID-Noise-Cancellation2/results/" 
path2 = "D:/Github Repos/Flipkart-GRID-Noise-Cancellation2/results2/" 

for doc in os.listdir(path1):
    name = doc.split('.')[0]
    shutil.move(path1+doc, path2+name+'.wav')