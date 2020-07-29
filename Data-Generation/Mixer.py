"""
Script for making the audio files with pydub
"""
import pydub
import os
import numpy as np
import random

clean_path = 'C:/Projects/Flipkart-GRID-Noise-Cancellation/Clean_Audio/'
noise_path = 'C:/Projects/Flipkart-GRID-Noise-Cancellation/Noise_Audio/'

def scale_audio(clean_arr, noise_arr):
        desired = random.uniform(0.6, 0.8)
        clean_avg = clean_arr.dBFS
        noise_avg = noise_arr.dBFS

        noise_arr += noise_avg - desired*clean_avg

        return clean_arr, noise_arr

counter = 0
for i in os.listdir(noise_path):
    for j in os.listdir(clean_path):
        counter = counter + 1
        sound1 = pydub.AudioSegment.from_file(clean_path + j)
        sound2 = pydub.AudioSegment.from_file(noise_path + i)
        
        sound1, sound2 = scale_audio(sound1, sound2)
        
        if len(sound1)<len(sound2):
            combined = sound2.overlay(sound1)
        else:
            combined = sound1.overlay(sound2)
            
        combined.export('Mixed/' + str(counter)+'.wav', format = 'wav')
        sound1.export('Target/' + str(counter) + ' target.wav', format = 'wav')