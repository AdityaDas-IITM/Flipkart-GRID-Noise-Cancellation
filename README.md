# Flipkart-GRID-Noise-Cancellation-Solution

This repository contains team Third Degree Burn's solution for Round 3 of the Flipkart GRiD 2.0.

As of now we have uploaded the dataset and are working on producing the code for our solution which implements the Conv-Tasnet 
to separate the primary speaker from background noise in a given audio file.

# The Dataset
## Original Audio
The Original Audio Recordings provided by Flipkart for this competition can be found [here](Audio Recordings/)

## Target Audio
We modified 30 of the above files to represent what we think would be ideal target audio for the model to learn from. 
We did this modification using Audacity where we recognized the segment where the primary speaker speaks and replaced all the remaining
parts of the audio with silence.

The modfied audio files can be found [here](Targets/)
