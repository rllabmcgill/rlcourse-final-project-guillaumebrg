__author__ = 'Guillaume'

from keras.models import Model
from keras.layers import Input, Convolution2D, Activation, Flatten, Dense, \
    merge, RepeatVector, TimeDistributed, LSTM, Masking, Reshape, Lambda, Permute
from keras.optimizers import SGD, Adam
import keras.backend as K


def CNN_3x3(grid_shape, lr=0.1, dueling=True, adam=False):
    # Input node
    grid = Input(grid_shape)
    # Hidden layer
    x = Convolution2D(16, 3, 3)(grid)
    x = Activation("relu")(x)
    # Output layer
    if dueling:
        a = Convolution2D(4, 1, 1)(x)
        a = Flatten()(a)

        v = Convolution2D(1, 1, 1)(x)
        v = Flatten()(v)
        v = RepeatVector(4)(v)
        v = Flatten()(v)

        q = merge([a,v], mode='sum')
    else:
        q = Convolution2D(4, 1, 1)(x)
        q = Flatten()(q)
    # Keras model
    neuralnet = Model(grid, q)
    # Compile
    #neuralnet.compile(Adam(lr=lr), loss="mse")
    if adam:
        neuralnet.compile(Adam(lr=lr), loss="mse")
    else:
        neuralnet.compile(SGD(lr=lr), loss="mse")
    return neuralnet


def stateful_RCNN_3x3(input_shape, lr=0.1, adam=False):
    # Input node
    grids = Input(batch_shape=input_shape)
    # Hidden layer
    x = LSTM(16, return_sequences=True, stateful=True)(grids)
    # Output layer
    q = TimeDistributed(Dense(4))(x)
    # Keras model
    neuralnet = Model(grids, q)
    # Compile
    if adam:
        neuralnet.compile(Adam(lr=lr), loss="mse")
    else:
        neuralnet.compile(SGD(lr=lr), loss="mse")
    return neuralnet


def RCNN_3x3(input_shape, lr=0.1, adam=False):
    # Input node
    grids = Input(input_shape)
    x = Masking(mask_value=0.)(grids)
    # Hidden layer
    x = LSTM(16, return_sequences=True)(x)
    # Output layer
    q = TimeDistributed(Dense(4))(x)
    # Keras model
    neuralnet = Model(grids, q)
    # Compile
    if adam:
        neuralnet.compile(Adam(lr=lr), loss="mse")
    else:
        neuralnet.compile(SGD(lr=lr), loss="mse")
    return neuralnet


def CNN_7x7(grid_shape, lr=0.1, dueling=True, adam=False):
    # Input node
    grid = Input(grid_shape)
    # Hidden layer
    x = Convolution2D(16, 3, 3, subsample=(2,2))(grid)
    x = Activation("relu")(x)
    x = Convolution2D(16, 3, 3)(x)
    x = Activation("relu")(x)
    # Output layer
    if dueling:
        a = Convolution2D(4, 1, 1)(x)
        a = Flatten()(a)

        v = Convolution2D(1, 1, 1)(x)
        v = Flatten()(v)
        v = RepeatVector(4)(v)
        v = Flatten()(v)

        q = merge([a,v], mode='sum')
    else:
        q = Convolution2D(4, 1, 1)(x)
        q = Flatten()(q)
    # Keras model
    neuralnet = Model(grid, q)
    # Compile
    #neuralnet.compile(Adam(lr=lr), loss="mse")
    if adam:
        neuralnet.compile(Adam(lr=lr), loss="mse")
    else:
        neuralnet.compile(SGD(lr=lr), loss="mse")
    return neuralnet


def stateful_RCNN_7x7(input_shape, lr=0.1, adam=False):
    # Input node
    grid = Input(batch_shape=input_shape)
    # Hidden layer
    x = Convolution2D(16, 3, 3, subsample=(2,2))(grid)
    x = Activation("relu")(x)
    #x = Convolution2D(16, 3, 3)(x)
    #x = Activation("relu")(x)
    # Hidden layer
    reshaped = Reshape((1, 16*3*3))(x)
    #mask = Masking(mask_value=0.)(flatten)
    x = LSTM(16, return_sequences=True, stateful=True)(reshaped)
    # Output layer
    q = TimeDistributed(Dense(4))(x)
    # Keras model
    neuralnet = Model(grid, q)
    # Compile
    if adam:
        neuralnet.compile(Adam(lr=lr), loss="mse")
    else:
        neuralnet.compile(SGD(lr=lr), loss="mse")
    return neuralnet


def stateful_RMCNN_3x3(shape_t, shape_past, lr=0.1, adam=False):
    # Input node
    grid = Input(batch_shape=shape_t)
    previous_grids = Input(batch_shape=shape_past)
    encoder = Convolution2D(16, 3, 3, activation="relu")
    # Hidden layer
    e = encoder(grid)
    e_past = TimeDistributed(encoder)(previous_grids)
    # Hidden layer
    e_reshaped = Reshape((1, 16*1*1))(e)
    h = LSTM(16, return_sequences=True, stateful=True)(e_reshaped)

    e_past_reshaped = TimeDistributed(Reshape((16*1*1, )))(e_past)
    m_val = TimeDistributed(Dense(16, activation="relu"))(e_past_reshaped)
    m_key = TimeDistributed(Dense(16, activation="relu"))(e_past_reshaped)

    p = RepeatVector(shape_past[1])(Flatten()(h))
    p = merge([p, m_key], mode="mul")
    p = Lambda(lambda x: K.sum(x, axis=2), output_shape=lambda s: (s[0], s[1]))(p)
    p = Activation("softmax")(p)
    p = RepeatVector(16)(p)
    p = Permute((2,1))(p)

    o = merge([p, m_val], mode="mul")
    o = Lambda(lambda x: K.sum(x, axis=1, keepdims=True), output_shape=lambda s: (s[0], 1, s[2]))(o)

    c = merge([h, o], mode="concat", concat_axis=2)
    # Output layer
    q = TimeDistributed(Dense(4))(c)
    # Keras model
    neuralnet = Model([grid, previous_grids], q)
    # Compile
    if adam:
        neuralnet.compile(Adam(lr=lr), loss="mse")
    else:
        neuralnet.compile(SGD(lr=lr), loss="mse")
    return neuralnet

