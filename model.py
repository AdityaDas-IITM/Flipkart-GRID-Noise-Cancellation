from keras.layers import Dense, Conv1D, BatchNormalization, Dropout, LeakyReLU, Input, Flatten, Multiply, Conv2D, Reshape, PReLU, Add
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.activations import sigmoid
import numpy as np
import tensorflow as tf

class models():
    def __init__(self, A, L, N, B, H, Sc, vr, bl):
        self.L = L
        self.N = N
        self.B = B
        self.H = H
        self.A = A
        self.Sc = Sc

        gbl_model = self.encoder()
        gbl_model = Model(inputs = gbl_model.input, outputs = [gbl_model.output,BatchNormalization()(gbl_model.output)])

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
        gbl_model = Model(inputs = gbl_model.input, outputs = mult)
        gbl_model.summary()
        decoder = self.decoder()
        #decoder_model = Model(inputs = mult, outputs = decoder(mult))

        final_output = decoder(mult)
        gbl_model = Model(gbl_model.input, final_output)
        #gbl_model = Model(inputs = gbl_model.input, outputs = decoder(gbl_model.output))
        gbl_model.summary()

        #final_model = Model(inputs = gbl_model.input, outputs = decoder_model.output)


        '''TCN = [self.ConvBlock(0, i, 0)[0] for i in range(2)]
        Sc_layers = [TCN[i].output[0] for i in range(2)]
        for verticals in range(2):
            for blocks in range(5):
                new_model = self.ConvBlock(blocks+1, verticals, blocks+1)[0]
                TCN[verticals] = Model(inputs = TCN[verticals].input, outputs = new_model(TCN[verticals].output[1]))
                Sc_layers[verticals] = Add()([Sc_layers[verticals], TCN[verticals].output[0]])

        for j in range(2):
            gbl_model = Model(inputs = gbl_model.input, outputs = TCN[i](gbl_model.outputs))'''



    def encoder(self):
        #input dimension 1xL, output dimension 1xN
        input_layer = Input(shape = (self.A, self.L, 1))
        layer1 = Conv2D(self.N, (1,self.L), input_shape = (self.A, self.L,1), activation = 'relu')(input_layer)
        layer2 = Flatten()(layer1)
        layer3 = Reshape(target_shape = (self.A, self.N, 1))(layer2)


        layer4 = Conv2D(self.N, (1,self.L), input_shape = (self.A, self.L,1), activation = 'sigmoid')(input_layer)
        layer5 = Flatten()(layer4)
        layer6 = Reshape(target_shape = (self.A, self.N, 1))(layer5)

        layer7 = Multiply()([layer3, layer6])

        model = Model(inputs = input_layer, outputs = layer7)
        return model

    def ConvBlock(self, x, vertical, block):
        #input is BxN
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
        model1 = Model(inputs = input_layer, outputs = Skip_Connection )
        return [model, model1]
    
    def decoder(self):
        input_layer = Input(shape = (self.A, self.N, 1))
        layer1 = Conv2D(self.L, (1, self.N), activation = 'relu', input_shape = (self.A, self.N, 1))(input_layer)
        layer2 = Flatten()(layer1)
        output = Reshape(target_shape = (self.A, self.L, 1))(layer2)

        model = Model(inputs = input_layer, outputs = output)
        return model

    def ChannelChanger(self, input_channels, output_channels):
        input_layer = Input(shape = (input_channels, self.N, 1))
        layer1 = Conv2D(output_channels, (input_channels, 1), activation = 'relu')(input_layer)
        layer2 = Flatten()(layer1)
        layer3 = Reshape(target_shape = (output_channels, self.N, 1))(layer2)

        model = Model(inputs = input_layer, outputs = layer3)
        return model

test_model = models(10, 100, 200, 10, 20, 30, 2, 3).final_model
test_model.summary()