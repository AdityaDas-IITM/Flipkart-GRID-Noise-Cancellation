# Imports
from tensorflow.keras.layers import Dense, Conv1D, BatchNormalization, Dropout, LeakyReLU, Input, Flatten, Multiply, Conv2D, Reshape, PReLU, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.activations import sigmoid
import tensorflow.compat.v1 as tf
import numpy as np
import keras.backend as K

#Defining a models class
class models():
    def __init__(self, A, L, N, B, H, Sc, vr, bl):
        self.L = L
        self.N = N
        self.B = B
        self.H = H
        self.A = A
        self.Sc = Sc

        self.whole_audio = Input(shape = (A*L,))
        
        self.encoder_model = self.encoder()
        gbl_model = Model(inputs = self.encoder_model.input, outputs = [self.encoder_model.output,BatchNormalization()(self.encoder_model.output)])

        encoded_values = gbl_model.output[0]
        scale_conv1 = self.ChannelChanger(A, B)
        gbl_model = Model(inputs = gbl_model.input, outputs = scale_conv1(gbl_model.output[1]))
        
        new_block = self.ConvBlock(0,0,0)[0]
        gbl_model = Model(inputs = gbl_model.input, outputs = new_block(gbl_model.output))
        Sc_layer = gbl_model.output[0]

        for blocks in range(bl-1):
            new_block = self.ConvBlock(blocks+1, 0, blocks+1)[0]
            gbl_model = Model(inputs = gbl_model.input, outputs = new_block(gbl_model.output[1]))
            Sc_layer = Add()([Sc_layer, gbl_model.output[0]])

        for verticals in range(vr-2):
            for blocks in range(bl):
                new_block = self.ConvBlock(blocks, verticals+1, blocks)[0]
                gbl_model = Model(inputs = gbl_model.input, outputs = new_block(gbl_model.output[1]))
                Sc_layer = Add()([Sc_layer, gbl_model.output[0]])
        
        for blocks in range(bl-1):
            new_block = self.ConvBlock(blocks+1, vr-1, blocks)[0]
            gbl_model = Model(inputs = gbl_model.input, outputs = new_block(gbl_model.output[1]))
            Sc_layer = Add()([Sc_layer, gbl_model.output[0]])

        new_block = self.ConvBlock(bl-1, vr-1, bl-1)[1]
        gbl_model = Model(inputs = gbl_model.input, outputs = new_block(gbl_model.output[1]))
        Sc_layer = Add()([Sc_layer, gbl_model.output])

        Sc_layer = PReLU()(Sc_layer)

        scale_conv2 = self.ChannelChanger(self.Sc, self.A)
        Sc_layer = scale_conv2(Sc_layer)
        mask = sigmoid(Sc_layer)

        mult = Multiply()([mask,encoded_values])
        '''
        mult_dash = encoded_values - mult
        mult_flatten = Flatten()(mult)
        mult_dash_flatten = Flatten()(mult_dash)
        '''
        self.decoder_model = self.decoder()
        

        final_output = self.decoder_model(mult)

        self.gbl_model = Model(inputs = [gbl_model.input, self.whole_audio], outputs = final_output)
        
        def gbl_model_loss(ytrue, ypred, whole_audio = self.whole_audio):
            
            non_primary = whole_audio - ypred # (2,220000)
            sim = tf.math.divide(K.batch_dot(non_primary, ypred, axes = (1,1)),tf.math.multiply(tf.norm(non_primary, axis = [-2,-1]),tf.norm(ypred, axis = [-2,-1]))) # (2,2)
            sim = K.mean(sim)
            #sim = tf.math.divide(K.batch_dot(mult_flatten, mult_dash_flatten, axes = (1,1)),tf.math.multiply(tf.norm(mult_flatten, axis = [-2,-1]),tf.norm(mult_dash_flatten, axis = [-2,-1])))
            #sim = K.mean(sim)
            
            ypred = ypred - K.mean(ypred)
            ytrue = ytrue - K.mean(ytrue)
            s_target = tf.math.multiply(tf.math.divide(K.batch_dot(ytrue, ypred, axes = (1,1)),K.batch_dot(ytrue,ytrue, axes = (1,1))),ytrue)
            e_noise = ypred - s_target
            SNR = 10*(K.log(tf.math.divide(K.batch_dot(s_target, s_target, axes = (1,1)),K.batch_dot(e_noise, e_noise, axes = (1,1))))/K.log(10.0))
            SNR = K.mean(SNR)
            return sim - 10*SNR
            #return sim
        
        #self.gbl_model.add_loss(gbl_model_loss(self.y_true, final_output, self.whole_audio))
        self.gbl_model.compile(optimizer = Adam(lr=0.001), loss = gbl_model_loss)
        
        self.encoder_decoder_model = Model(inputs = self.encoder_model.input, outputs = self.decoder_model(self.encoder_model.output))
        def encoder_decoder_model_loss(ytrue, ypred):
            ypred = ypred - K.mean(ypred) #(2,220000)
            ytrue = ytrue - K.mean(ytrue) #(2,220000)             
            s_target = tf.math.multiply(tf.math.divide(K.batch_dot(ytrue, ypred, axes = (1,1)),K.batch_dot(ytrue,ytrue, axes = (1,1))),ytrue) # (2,220000)
            e_noise = ypred - s_target # (2,220000)
            SNR = 10*(K.log(tf.math.divide(K.batch_dot(s_target, s_target, axes = (1,1)),K.batch_dot(e_noise, e_noise, axes = (1,1))))/K.log(10.0))
            SNR = K.mean(SNR)
            return -SNR
        self.encoder_decoder_model.compile(Adam(lr = 0.001), loss = encoder_decoder_model_loss)
        
    def encoder(self):
        #input dimension 1xL, output dimension 1xN
        '''
        The encoder takes in an input of dimension AxL where A is the number of segments the audio is been broken to with 0 padding if necessary and
        L is the length of each segment. The output is of dimension AxN where each segment of length L is converted to a vector representation of length N
        '''
        input_layer = Input(shape = (self.A, self.L, 1))
        #self.whole_audio = Input(shape = (self.A*self.L))
        #self.whole_audio = tf.placeholder(dtype = 'float32', shape = (self.A*self.L))
        layer1 = Conv2D(self.N, (1,self.L), input_shape = (self.A, self.L,1), activation = 'relu')(input_layer)
        layer2 = Flatten()(layer1)
        layer3 = Reshape(target_shape = (self.A, self.N, 1))(layer2)


        layer4 = Conv2D(self.N, (1,self.L), input_shape = (self.A, self.L,1), activation = 'sigmoid')(input_layer)
        layer5 = Flatten()(layer4)
        layer6 = Reshape(target_shape = (self.A, self.N, 1))(layer5)

        '''
        The two blocks take the same input layer and performs the same convolution operation. One block is relu activated and the other is sigmoid.
        these outputs are then multiplied to finally give the encodings and this architecture is a gated convolution network.
        '''

        layer7 = Multiply()([layer3, layer6])

        #model = Model(inputs = [input_layer, whole_audio], outputs = layer7)
        model = Model(inputs = input_layer, outputs = layer7)
        return model

    def ConvBlock(self, x, vertical, block):
        '''
        This is the building block of a TCN. The input is of dimention BxN and it has two outputs of dimension BxN and ScxN.
        '''
        input_layer = Input(shape = (self.B, self.N, 1))
        layer1 = Conv2D(self.H, (self.B, 1), input_shape = (self.B, self.N, 1), activation = 'relu')(input_layer)
        layer2 = Flatten()(layer1)
        layer3 = Reshape(target_shape = (self.H, self.N, 1))(layer2)
        #layer4 = LeakyReLU(alpha = 0.02)(layer3)
        layer4 = PReLU()(layer3)

        layer5 = BatchNormalization()(layer4)

        layer6 = Conv2D(1, (1,2**(x)), activation = 'linear', padding = 'same')(layer5)
        #layer7 = LeakyReLU(alpha = 0.02)(layer6)
        layer7 = PReLU()(layer6)
         
        layer8 = BatchNormalization()(layer7)

        layer9 = Conv2D(self.Sc, (self.H,1), activation = 'relu')(layer8)
        layer10 = Flatten()(layer9)
        Skip_Connection = Reshape(target_shape = (self.Sc, self.N, 1))(layer10)

        layer11 = Conv2D(self.B, (self.H,1), activation = 'relu')(layer8)
        layer12 = Flatten()(layer11)
        output = Add()([Reshape(target_shape = (self.B, self.N, 1))(layer12), input_layer]) 

        model = Model(inputs = input_layer, outputs = [Skip_Connection, output], name = 'Vertical'+str(vertical)+'block'+str(block))
        model1 = Model(inputs = input_layer, outputs = Skip_Connection)
        return [model, model1]
    
    def decoder(self):
        input_layer = Input(shape = (self.A, self.N, 1))
        layer1 = Conv2D(self.L, (1, self.N), activation = 'linear', input_shape = (self.A, self.N, 1))(input_layer)
        layer2 = Flatten()(layer1)
        #output = Reshape(target_shape = (self.A, self.L, 1))(layer2)

        model = Model(inputs = input_layer, outputs = layer2, name = 'decoder')
        return model

    def ChannelChanger(self, input_channels, output_channels):
        input_layer = Input(shape = (input_channels, self.N, 1))
        layer1 = Conv2D(output_channels, (input_channels, 1), activation = 'relu')(input_layer)
        layer2 = Flatten()(layer1)
        layer3 = Reshape(target_shape = (output_channels, self.N, 1))(layer2)

        model = Model(inputs = input_layer, outputs = layer3)
        return model
    
    def train(self, input_value, whole_audio, output, epochs_gbl_model, epochs_encoder_decoder, batch_size, valid, valid_target, valid_reshaped):
        
        print('...training encoder decoder...')
        self.encoder_decoder_model.fit(x = input_value, y = whole_audio,validation_data = (valid_reshaped,valid_target), batch_size=batch_size, epochs = epochs_encoder_decoder) 
    
        print('...training gbl model...')
        self.gbl_model.fit(x = [input_value, whole_audio], y = output, epochs = epochs_gbl_model, validation_data = ([valid_reshaped,valid], valid_target), batch_size = batch_size)   

        

    def export_model(self, path):
        print('.....Saving Model......')
        self.gbl_model.save(path)