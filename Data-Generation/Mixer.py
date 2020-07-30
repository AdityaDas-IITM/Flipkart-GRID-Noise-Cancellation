"""
Script for making the audio files with pydub
"""
import pydub
import os
import numpy as np
import random

clean_path = 'C:/Users/nihal/Downloads/FlipkartNoiseCancel/Round3/Key Guidelines and Audio Recordings/Flipkart-GRID-Noise-Cancellation/Clean_Audio/'
noise_path = 'C:/Users/nihal/Downloads/FlipkartNoiseCancel/Round3/Key Guidelines and Audio Recordings/Flipkart-GRID-Noise-Cancellation/Noise_Audio/'

def scale_audio(clean_arr, noise_arr):
        desired = random.uniform(0.6, 0.8)
        clean_avg = clean_arr.dBFS
        noise_avg = noise_arr.dBFS

        noise_arr += noise_avg - desired*clean_avg

        return clean_arr, noise_arr

counter = 0
for i in os.listdir(noise_path):
    for j in os.listdir(clean_path):
        print(j)
        counter = counter + 1
        sound1 = pydub.AudioSegment.from_file(clean_path + j)
        sound2 = pydub.AudioSegment.from_file(noise_path + i)
        
        #sound1, sound2 = scale_audio(sound1, sound2)
        
        if len(sound1)<len(sound2):
            combined = sound2.overlay(sound1)
        else:
            combined = sound1.overlay(sound2)
            
        combined.export('C:/Users/nihal/Downloads/FlipkartNoiseCancel/Round3/Key Guidelines and Audio Recordings/Flipkart-GRID-Noise-Cancellation/Data-Generation/Mixed/' + str(counter)+'.wav', format = 'wav')
        sound1.export('C:/Users/nihal/Downloads/FlipkartNoiseCancel/Round3/Key Guidelines and Audio Recordings/Flipkart-GRID-Noise-Cancellation/Data-Generation/Target/' + str(counter) + ' target.wav', format = 'wav')