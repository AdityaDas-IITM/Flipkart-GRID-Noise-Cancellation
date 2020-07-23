## Imports

import tensorflow as tf
import utils
from model import models

#tf.compat.v1.disable_eager_execution()
tf.config.experimental_run_functions_eagerly(True)

A = 1000
L = 220
N = 500
B = 300
H = 500
Sc = 300
vr = 3
bl = 5

'''
TODO
1. Get paths of target and non target <<<DONE>>>
2. Process them <<<DONE>>>
3. Make the model
4. Fit it
5. Save weights
'''

out_arr_pad, _ = utils.inputProcess("C:/Users/nihal/Downloads/2nd Cross Road 2 target.wav", A, L)
in_arr_pad, in_arr_reshaped = utils.inputProcess("C:/Users/nihal/Downloads/2nd Cross Road 2.wav", A, L)

model = models(A, L, N, B, H, Sc, vr, bl).gbl_model

model.fit(x=[in_arr_reshaped, in_arr_pad], y=out_arr_pad, epochs=1)
model.summary()
