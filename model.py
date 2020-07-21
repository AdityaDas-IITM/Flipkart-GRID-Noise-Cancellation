from tensorflow.keras.layers import Dense, Conv1D, BatchNormalization, Dropout, LeakyReLU, Input, Flatten, Multiply, Conv2D, Reshape
from keras.optimizers import Adam
from keras.models import Sequential, Model
import numpy as np
import tensorflow as tf

class Model():
    def __init__(self, A, L, N, B, H, Sc):
        self.L = L
        self.N = N
        self.B = B
        self.H = H
        self.A = A
        self.Sc = Sc

        #model = Sequential


    def encoder(self):
        #input dimension 1xL, output dimension 1xN
        input_layer = Input(shape = (self.A, self.L, 1))
        layer1 = Conv2D(self.N, (1,self.L), input_shape = (self.A, self.L,1), activation = 'relu')(input_layer)
        layer2 = Flatten()(layer1)
        layer3 = Reshape(target_shape = (self.A, self.N))(layer2)


        layer4 = Conv2D(self.N, (1,self.L), input_shape = (self.A, self.L,1), activation = 'sigmoid')(input_layer)
        layer5 = Flatten()(layer4)
        layer6 = Reshape(target_shape = (self.A, self.N, 1))(layer5)

        layer7 = Multiply([layer3, layer6])

        model = Model(inputs = input_layer, outputs = layer7)
        return model

    def ConvBlock(self, x):
        #input is BxN
        input_layer = Input(shape = (self.B, self.N, 1))
        layer1 = Conv2D(self.H, (self.B, 1), input_shape = (self.B, self.N, 1), activation = 'relu')(input_layer)
        layer2 = Flatten()(layer1)
        layer3 = Reshape(target_shape = (self.H, self.N, 1))(layer2)
        layer4 = LeakyReLU(alpha = 0.02)(layer3)

        layer5 = BatchNormalization()(layer4)

        layer6 = Conv2D(1, (1,2**(x-1)), activation = 'linear', padding = 'same')(layer5)
        layer7 = LeakyReLU(alpha = 0.02)(layer6)
         
        layer8 = BatchNormalization()(layer7)

        layer9 = Conv2D(self.Sc, (self.H,1), activation = 'relu')(layer8)
        layer10 = Flatten()(layer9)
        Skip_Connection = Reshape(target_shape = (self.Sc, self.N, 1))(layer10)

        layer11 = Conv2D(self.B, (self.H,1), activation = 'relu')(layer8)
        layer12 = Flatten()(layer11)
        output = tf.add(Reshape(target_shape = (self.B, self.N, 1))(layer12), input_layer) 

        model = Model(inputs = input_layer, outputs = [Skip_Connection, output])
        return model
    
    def decoder(self):
        input_layer = Input(shape = (self.A, self.N, 1))
        layer1 = Conv2D(self.L, (1, self.N), activation = 'relu', input_shape = (self.A, self.N, 1))(input_layer)
        layer2 = Flatten()(layer1)
        output = Reshape(target_shape = (self.A, self.L, 1))(layer2)

        model = Model(inputs = input_layer, outputs = output)
        return model


    
















