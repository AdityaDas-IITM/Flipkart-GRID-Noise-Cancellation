# Flipkart-GRID-Noise-Cancellation-Solution

This repository contains team Third Degree Burn's solution for Round 3 of the Flipkart GRiD 2.0.

## API Usage
### Without a GUI
We have provided an API which uses Flask to take in the path to an input file or a directory with multiple input files and the path to an output directory where the files will be stored in WAV format. To make use of our scripts, run [this](FlaskNoGUI/wsgi.py) wsgi script on a terminal and then run [this](FlaskNoGUI/testing.py) interacting script separately to start interacting with the server.

The scripts can be found here:
- Web Server Gateway Interface Script : [WSGI Script](FlaskNoGUI/wsgi.py)
- The Script for the App : [App Script](FlaskNoGUI/app.py) 
- Script for interacting with the server : [Script for Interacting](FlaskNoGUI/testing.py)

### With a GUI
We have also made the scripts for having a small GUI incorporated with Flask using tkinter. The order to run the scripts is the same. You first run [this](FlaskGUI/wsgi_GUI.py) wsgi script on a terminal and then run [this](FlaskGUI/testing_GUI.py) interacting script separately to start interacting with the server. The only difference here is that On running the interacting script, a pop-up window shows in which you can select whether you want to input a single file or a whole directory. In any case, you need to select the input directory before the output directory, else the code will not run.

The scripts for this can be found here:
- Web Server Gateway Interface Script : [WSGI Script](FlaskGUI/wsgi_GUI.py)
- The Script for the App : [App Script](FlaskGUI/app_GUI.py) 
- Script for interacting with the server : [Script for Interacting](FlaskGUI/testing_GUI.py)

## Model Building
Following are the scripts that we have used for building our model:
- Utility Script: [Utils](Scripts-with-batches/utils.py)
