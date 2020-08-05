from tkinter import *
import tkinter.filedialog as filedialog
import tensorflow as tf
import librosa
import numpy as np
import os

def load_model():
    model = tf.keras.models.load_model('D:/Github Repos/Flipkart-GRID-Noise-Cancellation2/FlaskAPI/Model/gbl_model.h5', compile=False)
    return model

def inputProcess(filepath, A=2000, L=110):
    arr, _ = librosa.load(filepath, sr=22000)
    arr_pad = np.pad(arr, (0, A*L - len(arr)), 'constant', constant_values=(0,0))
    arr_reshaped = arr_pad.reshape(1, A, L, 1)
    arr_pad = np.reshape(arr_pad, (1, -1))

    return arr_reshaped

def wavCreator(path, arr):
    arr = np.array(arr).T
    librosa.output.write_wav(path, arr, sr=22000)

class smallGUI():

    def __init__(self):
        self.check = False
        self.first = Tk()  
        self.first.title(string = "Third Degree Burn")
        self.first.geometry("400x300")
        self.button1 = Button(self.first, text = "Run Single Audio FIle", command = self.singleinput, width = 30, height = 2).place(x = 90, y = 90)
        self.button2 = Button(self.first, text = "Run Multiple Audio files", command = self.multipleinput, width = 30, height = 2).place(x = 90, y = 190)
        self.first.mainloop() 

    def singleinput(self):
        self.check = True
        self.first.destroy()
        self.tk = Tk()  
        self.tk.title(string = "Third Degree Burn")
        self.tk.geometry("400x300")
        self.button3 = Button(self.tk, text = "Select Input File", command = self.inputfile, width = 30, height = 2).place(x = 90, y = 90)
        self.button4 = Button(self.tk, text = "Select Directory to Save Outputs", command = self.outputdir, width = 30, height = 2).place(x = 90, y = 190)
        self.tk.mainloop() 

    def multipleinput(self):
        self.first.destroy()
        self.tk = Tk()  
        self.tk.title(string = "Third Degree Burn")
        self.tk.geometry("400x300")
        self.button3 = Button(self.tk, text = "Select Directory with Input Files", command = self.inputdir, width = 30, height = 2).place(x = 90, y = 90)
        self.button4 = Button(self.tk, text = "Select Directory to Save Outputs", command = self.outputdir, width = 30, height = 2).place(x = 90, y = 190)
        self.tk.mainloop() 

    def inputfile(self):
        self.inputfile = filedialog.askopenfilename(title = "Audio file")

    def inputdir(self):
        self.inputdirpath = filedialog.askdirectory(title="Open Input folder")

    def outputdir(self):
        self.outputdirpath = filedialog.askdirectory(title="Open Output folder")
        self.tk.destroy()

guiobj = smallGUI()
model = load_model()

if(guiobj.check):
    arr_reshaped = inputProcess(guiobj.inputfile)
    denoised_arr = model.predict([arr_reshaped, np.zeros((1, 2000*110))])
    wavCreator(guiobj.outputdirpath + '/TDB_Prection.wav', denoised_arr)

else:
    for filename in os.listdir(guiobj.inputdirpath):
        filepath = guiobj.inputdirpath+'/'+filename
        arr_reshaped = inputProcess(filepath)
        denoised_arr = model.predict([arr_reshaped, np.zeros((1, 2000*110))])
        wavCreator(guiobj.outputdirpath + '/pred_'+filename, denoised_arr)
