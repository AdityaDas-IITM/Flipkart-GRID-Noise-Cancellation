# %% [code]
## Imports

import tensorflow as tf
import utils as utils
import os
from model import models
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
model.gbl_model.load_weights('../input/weights-files/weights Pass-2(1).h5')
batch_size = 32

train_path = '../input/flipkart-train-loud/Mixed/'
target_path = '../input/flipkart-target-loud/Target/'
#model.encoder_decoder_model.summary()
#model.gbl_model.summary()
for i in range(5):
    utils.initialize_counter(train_path, target_path)
    valid, valid_target, valid_reshaped = utils.get_validation_set(train_path, target_path, A, L)
    print(int(np.ceil((len(os.listdir(train_path)) - 200)/batch_size)))
    for f in range(int(np.ceil((len(os.listdir(train_path)) - 200)/batch_size))):
        print(i, f)
        in_arr_pad, out_arr_pad, in_arr_reshaped, updated_batch_size = utils.inputProcess(train_path, target_path, A, L, batch_size)
        #print(np.array(in_arr_reshaped).shape)
        #print(np.array(in_arr_pad).shape)
        #print(out_arr_pad.shape)
        model.train(in_arr_reshaped, in_arr_pad, out_arr_pad, 10, 5, updated_batch_size, valid, valid_target, valid_reshaped)
    model.gbl_model.save_weights('weights: Pass-' + str(i + 1) + '.h5')

#model.fit(x=[in_arr_reshaped, in_arr_pad], y=out_arr_pad, epochs=1)
#model.fit(x=in_arr_reshaped, y=out_arr_pad, epochs=1)
#model.gbl_model.summary()

pred, to_pred = utils.inputProcesstest('../input/flipkart-round-3-original/0.mp3.wav', A, L)
predict = model.gbl_model.predict([to_pred, pred])
utils.wavCreator("0_output.wav", predict)
#print(predict[predict<0])

model.gbl_model.save('gbl_model.h5')
