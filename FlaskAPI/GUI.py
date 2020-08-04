from tkinter import *
import tkinter.filedialog as filedialog

class smallGUI():

    def __init__(self):
        self.tk = Tk()  
        self.tk.title(string = "Third Degree Burn")
        self.tk.geometry("400x300")
        self.button1 = Button(self.tk, text = "Select Directory with Input Files", command = self.inputdir, width = 30, height = 2).place(x = 90, y = 90)
        self.button2 = Button(self.tk, text = "Select Directory to Save Outputs", command = self.outputdir, width = 30, height = 2).place(x = 90, y = 190)
        self.tk.mainloop() 


    def inputdir(self):
        self.inputdirpath = filedialog.askdirectory(title="Open Input folder")

    def outputdir(self):
        self.outputdirpath = filedialog.askdirectory(title="Open Output folder")
