'''
    Main Model Functions
'''

from data import DataGenerator
from keras.preprocessing.sequence import pad_sequences
from parameters import const_param as const

import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Input, TimeDistributed, Masking, Dropout
from keras.layers import Dense, Flatten, MaxPooling2D, Convolution2D
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

from sklearn.model_selection import KFold

from tqdm import tqdm
import numpy as np
import os
from datetime import timedelta
import time


def create_cnn_model(p,const):
    cnn_input = Input((95,250,4))
    c = cnn_input
    for i in range(p['cnn_layers']):
        c = Convolution2D(
                            filters = p['cnn_init_channels'] + i*p['cnn_channel_increase_delta'], 
                            kernel_size = (const['cnn_kernel_h'],const['cnn_kernel_w']),
                            strides = (1,1),
                            activation=const['cnn_activation']
                         )(c)
        c = MaxPooling2D( pool_size = (const['mp_kernel_h'],const['mp_kernel_w']) )(c)
    cnn_output = Flatten()(c)
    return Model(inputs=cnn_input, outputs=cnn_output)


def create_lstm_model(cnn_model,p,const):
    lstm_input = Input((None,95,250,4))
    l = TimeDistributed(cnn_model)(lstm_input)
    l = Masking(mask_value=0.)(l)
    for _ in range(p['lstm_layers']):
        RNN = p['rnn_unit_function']
        l = RNN(p['lstm_units'])(l)
    for _ in range(p['dense_layers_after_lstm']):
        l = Dense(p['dense_layer_units_after_lstm'])(l)
        l = Dropout(0.33)(l)
    lstm_output = Dense(1)(l)
    return Model(inputs=lstm_input, outputs=lstm_output)


def std_error(y_true,y_pred):
    global const
    return K.std( (y_true - y_pred)*const['Y_std'] )


def create_model(p,const,print_summary=False):
    ''' Creates Full Model from CNN and LSTM
    '''
    cnn_model = create_cnn_model(p,const)
    lstm_model = create_lstm_model(cnn_model,p,const)
    optimizer = Adam(lr=p['learning_rate'])
    lstm_model.compile(
                    optimizer = optimizer,
                    loss = const['loss'],
                    metrics = ['mse',std_error]
                )
    if print_summary:
        cnn_model.summary()
        lstm_model.summary()
    return lstm_model


class EvaluateData(keras.callbacks.Callback):
    def __init__(self,generator,log_word):
        self.generator = generator
        self.log_word = log_word

    def on_epoch_end(self,batch,logs):
        metric_values = self.model.evaluate_generator(self.generator)
        metric_names = self.model.metrics_names
        for metric_name,value in zip(metric_names,metric_values):
            logs[ self.log_word +'_'+ metric_name ] = value


class PredictData(keras.callbacks.Callback):
    def __init__(self,generator,denormalize_fun,log_word):
        self.generator = generator
        self.denormalize = denormalize_fun
        self.log_word = log_word

    def on_epoch_end(self,batch,logs):
        predictions = self.model.predict_generator(self.generator)
        predictions = self.denormalize(predictions)
        if self.log_word != '':
                logs[ self.log_word +'_pred'] = predictions
                logs[ self.log_word +'_prediction_mean' ] = predictions.mean()
                logs[ self.log_word +'_prediction_std' ] = predictions.std()
                logs[self.log_word +'_labels'] = self.generator.get_y_array()
        else:
                logs[ self.log_word +'pred'] = predictions
                logs[self.log_word +'labels'] = self.generator.get_y_array()
                logs[ self.log_word +'prediction_mean' ] = predictions.mean()
                logs[ self.log_word +'prediction_std' ] = predictions.std()


def fit_model(model, train_indexes, valid_indexes, test_indexes, const, p, verbose=1):
    ''' train_history has train and valid '''
    train_gen = DataGenerator(train_indexes, const['batch_size'], const['datadir'])
    valid_gen = DataGenerator(valid_indexes, const['batch_size'], const['datadir'])
    test_gen = DataGenerator(test_indexes, const['batch_size'], const['datadir'])
    callbacks = []
    # tensorboard_logdir_path = p2logdir_path(folder_path=const['tensorboard_dir'],p=p)
    # callbacks.append( tensorboard(tensorboard_logdir_path, batch_size=const['batch_size']) )
    callbacks.append( EvaluateData(test_gen,log_word='test') )
    callbacks.append( PredictData(test_gen,denormalize,log_word='test') )
    callbacks.append( PredictData(valid_gen,denormalize,log_word='val') )
    callbacks.append( PredictData(train_gen,denormalize,log_word='') )
    history = model.fit_generator(
                                        generator = train_gen,
                                        epochs = const['epochs'],
                                        steps_per_epoch = len(train_gen),
                                        validation_data = valid_gen,
                                        validation_steps = len(valid_gen),
                                        verbose = verbose,
                                        callbacks = callbacks
                                    )
    return history.history


def denormalize(target_array):
    global const
    return np.array(target_array) * const['Y_std'] + const['Y_mean']


def check_indexes(data,*check_indexes):
    for indexes in check_indexes:
        for index in indexes:
            if index not in data:
                return False
    return True


def params2folder_name(p):
    s = ''
    for a,aa in p.items():
        if callable(aa): continue # skip functions
        s += str(aa)+'_'
    s = s[:-1]
    return s

def p2logdir_path(folder_path,p):
    if not os.path.exists(folder_path): os.makedirs(folder_path)
    s = params2folder_name(p)
    return os.path.join(folder_path,s)


def fit_kfold_model(create_model_fun, data_indexes, test_indexes, const, p, verbose=1): # index_X - training data (both valid+training) indexes
    ''' All returned variables are lists of values for each epoch! '''
    valid_fitness_list = []; train_fitness_list = []; test_fitness_list = []; 
    valid_std_list = []; train_std_list = []; test_std_list = []; 
    test_prediction_mean_list = []; test_prediction_std_list = []; 
    valid_prediction_mean_list = []; valid_prediction_std_list = []; 
    keras_histories = []

    for i,(index_train, index_valid) in enumerate(KFold(n_splits=const['kfold_split'],shuffle=True).split(data_indexes)):
        print('Fold {}:'.format(i+1))
        
        train_indexes, valid_indexes = data_indexes[ index_train ], data_indexes[ index_valid ] # KFold returns indexes of data_indexes, this returns indexes of data
        model = create_model_fun(p,const)
        history = fit_model(model,train_indexes,valid_indexes,test_indexes,const,p,verbose)
        keras_histories.append(history)

        train_fitness_list.append( history[ const['fitness_result'] ] )
        valid_fitness_list.append( history[ 'val_'+const['fitness_result'] ] )
        test_fitness_list.append( history[ 'test_'+const['fitness_result'] ] )

        train_std_list.append( history[ 'std_error' ] )
        valid_std_list.append( history[ 'val_std_error' ] )
        test_std_list.append( history[ 'test_std_error' ] )

        test_prediction_mean_list.append( history[ 'test_prediction_mean' ] )
        test_prediction_std_list.append( history[ 'test_prediction_std' ] )

        valid_prediction_mean_list.append( history[ 'valid_prediction_mean' ] )
        valid_prediction_std_list.append( history[ 'valid_prediction_std' ] )
    
    train_fitness,valid_fitness,test_fitness = np.mean( train_fitness_list,axis=0 ), np.mean( valid_fitness_list,axis=0 ), np.mean( test_fitness_list,axis=0 )
    train_std,valid_std,test_std = np.mean( train_std_list,axis=0 ), np.mean( valid_std_list,axis=0 ), np.mean( test_std_list,axis=0 )    

    print( '\n \t train-mse: {:>10} \t valid-mse: {:>10} \t test-mse: {:>10}'.format( \
        train_fitness[-1],valid_fitness[-1],test_fitness[-1]) )
    print( ' \t train-std: {:>10} \t valid-std: {:>10} \t test-std: {:>10}'.format( \
        str(timedelta(seconds=train_std[-1])),str(timedelta(seconds=valid_std[-1])),str(timedelta(seconds=test_std[-1]))) )

    return keras_histories,{
            'fitness':{
                    'train':{
                        # 'list':np.array(train_fitness_list),
                        'mean':np.mean(train_fitness_list,axis=0),
                        'std':np.std(train_fitness_list,axis=0)
                        },
                    'valid':{
                        # 'list':np.array(valid_fitness_list),
                        'mean':np.mean(valid_fitness_list,axis=0),
                        'std':np.std(valid_fitness_list,axis=0)
                        },
                    'test':{
                        # 'list':np.array(test_fitness_list),
                        'mean':np.mean(test_fitness_list,axis=0),
                        'std':np.std(test_fitness_list,axis=0)
                        },
                },
            'std':{
                    'train':{
                        # 'list':np.array(train_std_list),
                        'mean':np.mean(train_std_list,axis=0)/3600,
                        'std':np.std(train_std_list,axis=0)/3600
                        },
                    'valid':{
                        # 'list':np.array(valid_std_list),
                        'mean':np.mean(valid_std_list,axis=0)/3600,
                        'std':np.std(valid_std_list,axis=0)/3600
                        },
                    'test':{
                        # 'list':np.array(test_std_list),
                        'mean':np.mean(test_std_list,axis=0)/3600,
                        'std':np.std(test_std_list,axis=0)/3600
                        },
                },
            'predictions':{
                    'valid':{
                        'std':np.mean(valid_prediction_std_list,axis=0)/3600,
                        'mean':np.mean(valid_prediction_mean_list,axis=0)/3600
                        },
                    'test':{
                        'std':np.mean(test_prediction_std_list,axis=0)/3600,
                        'mean':np.mean(test_prediction_mean_list,axis=0)/3600
                        }
                }
           }



#