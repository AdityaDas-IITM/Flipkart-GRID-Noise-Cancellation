# %% [code]
## Imports

import tensorflow as tf
import utilswithbatches as utils
import os
from modelwithbatches import models
import numpy as np
import keras.backend as K

tf.compat.v1.disable_eager_execution()
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

model = models(A, L, N, B, H, Sc, vr, bl)

batch_size = 8

train_path = '../input/flipkart-testing/'
target_path = '../input/flipkart-grid-20-round-3-targets/'
model.encoder_decoder_model.summary()
model.gbl_model.summary()
for i in range(500):
    print(i)
    utils.initialize_counter()
    for f in range(int(np.ceil(len(os.listdir(train_path))/batch_size))):
        out_arr_pad, _, updated_batch_size = utils.inputProcess1(target_path, A, L, batch_size)
        in_arr_pad, in_arr_reshaped, _ = utils.inputProcess2(train_path, A, L, batch_size)
        #print(np.array(in_arr_reshaped).shape)
        #print(np.array(in_arr_pad).shape)
        #print(out_arr_pad.shape)
        model.train(in_arr_reshaped, in_arr_pad, out_arr_pad, 15, 8, updated_batch_size)

#model.fit(x=[in_arr_reshaped, in_arr_pad], y=out_arr_pad, epochs=1)
#model.fit(x=in_arr_reshaped, y=out_arr_pad, epochs=1)
#model.gbl_model.summary()
model.gbl_model.save_weights('weights.h5')
pred, to_pred = utils.inputProcesstest(train_path+'2nd Cross Road 9.m4a.wav', A, L)
predict = model.gbl_model.predict([to_pred, pred])
utils.wavCreator("a.wav", predict)
#print(predict[predict<0])