## Imports

import tensorflow as tf
import utils
from model import models
import keras.backend as K

#tf.compat.v1.disable_eager_execution()
#tf.config.experimental_run_functions_eagerly(True)
K.clear_session()
A = 1000
L = 220
N = 128
B = 128
H = 128
Sc = 128
vr = 3
bl = 8

'''
TODO
1. Get paths of target and non target <<<DONE>>>
2. Process them <<<DONE>>>
3. Make the model
4. Fit it
5. Save weights
'''

out_arr_pad, _ = utils.inputProcess("C:/Users/Aditya Das/OneDrive/Desktop/2nd Cross Road 2 target.wav", A, L)
in_arr_pad, in_arr_reshaped = utils.inputProcess("C:/Users/Aditya Das/OneDrive/Desktop/2nd Cross Road 2.wav", A, L)

model = models(A, L, N, B, H, Sc, vr, bl)

#model.fit(x=[in_arr_reshaped, in_arr_pad], y=out_arr_pad, epochs=1)
#model.fit(x=in_arr_reshaped, y=out_arr_pad, epochs=1)
model.gbl_model.summary()
model.train(in_arr_reshaped, in_arr_pad, out_arr_pad, 5)
predict = model.gbl_model.predict(in_arr_reshaped)
utils.wavCreator("C:/Users/Aditya Das/OneDrive/Desktop/a.wav", predict)
print(predict)
