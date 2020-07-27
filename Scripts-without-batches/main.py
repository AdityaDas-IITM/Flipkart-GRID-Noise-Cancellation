## Imports

import tensorflow as tf
import utils
from model import models
import keras.backend as K

#tf.compat.v1.disable_eager_execution()
#tf.config.experimental_run_functions_eagerly(True)
K.clear_session()
A = 2000
L = 110
N = 512
B = 128
H = 512
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

Input_PATH = ''
Target_PATH = ''
Save_PATH = ''

out_arr_pad, _ = utils.inputProcess(Target_PATH, A, L)
in_arr_pad, in_arr_reshaped = utils.inputProcess(Input_PATH, A, L)

model = models(A, L, N, B, H, Sc, vr, bl)

#model.fit(x=[in_arr_reshaped, in_arr_pad], y=out_arr_pad, epochs=1)
#model.fit(x=in_arr_reshaped, y=out_arr_pad, epochs=1)
model.gbl_model.summary()
model.train(in_arr_reshaped, in_arr_pad, out_arr_pad, 10, 5)
predict = model.gbl_model.predict(in_arr_reshaped)
utils.wavCreator(Save_PATH, predict)
print(predict)
