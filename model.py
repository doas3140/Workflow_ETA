from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, TimeDistributed
from tensorflow.python.keras.layers import Dense, Flatten, MaxPooling2D, Convolution2D
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import TensorBoard

import numpy as np
import os

def create_cnn_model(p):
    cnn_input = Input((95,250,4))
    c = cnn_input
    for i in range(p['cnn_layers']):
        c = Convolution2D(
                            filters = p['cnn_init_channels'] + i*p['cnn_channel_increase_delta'], 
                            kernel_size = (p['cnn_kernel_h'],p['cnn_kernel_w']),
                            strides = (p['cnn_strides_h'],p['cnn_strides_w']),
                            activation=p['cnn_activation']
                         )(c)
        # c = MaxPooling2D( pool_size = (p['mp_kernel_h'],p['mp_kernel_w']) )(c)
    cnn_output = Flatten()(c)
    return Model(inputs=cnn_input, outputs=cnn_output)

def create_lstm_model(cnn_model,p):
    lstm_input = Input((None,95,250,4))
    l = TimeDistributed(cnn_model)(lstm_input)
    for i in range(p['lstm_layers']):
        RNN = p['rnn_unit_function']
        l = RNN(p['lstm_units'])(l)
    for i in range(p['dense_layers_after_lstm']):
        l = Dense(p['dense_layer_units_after_lstm'])(l)
    lstm_output = Dense(1)(l)
    return Model(inputs=lstm_input, outputs=lstm_output)

def create_model(p,const,print_summary=False):
    cnn_model = create_cnn_model(p)
    lstm_model = create_lstm_model(cnn_model,p)
    optimizer = Adam(lr=p['learning_rate'])
    lstm_model.compile(
                    optimizer = optimizer,
                    loss = const['loss'],
                    metrics = ['mae','mse']
                )
    if print_summary:
        cnn_model.summary()
        lstm_model.summary()
    return lstm_model

def create_generator(X,Y,batch_size=1):
    while True:
        idx = np.random.randint(0, len(X), batch_size)
        yield np.expand_dims(X[idx][0],axis=0),Y[idx]

def fit_model(model,epochs,p, X_train,Y_train, X_valid,Y_valid, batch_size=1,verbose=1):
    train_gen = create_generator(X_train, Y_train, batch_size=batch_size)
    valid_gen = create_generator(X_valid, Y_valid, batch_size=batch_size)
    history = model.fit_generator(
                                    generator = train_gen,
                                    epochs = epochs,
                                    steps_per_epoch = X_train.shape[0] // batch_size,
                                    validation_data = valid_gen,
                                    validation_steps = X_valid.shape[0] // batch_size,
                                    verbose = verbose
                                 )
    return history

def params2folder_name(p):
    s = ''
    for a,aa in p.items():
        if callable(aa): continue # skip functions
        s += str(aa)+'_'
    s = s[:-1]
    return s

def logdir_path(folder_path,p):
    s = params2folder_name(p)
    return os.path.join(folder_path,s)

def tensorboard(logdir_path, batch_size):
    return TensorBoard(
                        log_dir = logdir_path,
                        histogram_freq = 0,
                        batch_size = batch_size,
                        write_graph = True,
                        write_grads = False,
                        write_images = False
                      )

