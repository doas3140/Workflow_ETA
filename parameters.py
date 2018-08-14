from keras.layers import SimpleRNN, GRU, LSTM
from skopt.space import Real, Categorical, Integer

import os
import json


def get_metadata_dict(datadir):
    jsonpath = os.path.join(datadir,'meta.json')
    return json.loads( open(jsonpath).read() )

datadir = os.path.join('/mnt/sda1/Datasets/Workflow_ETA/data_prod/tmpDir/tempminidata')
results_dir = os.path.join(os.getcwd(),'temp') # where all the results is stored
try:
    METADATA = get_metadata_dict(datadir)
except:
    METADATA = {'Y_mean':0,'Y_std':1}

const_param = {
    # variables
    'Y_mean':METADATA['Y_mean'],
    'Y_std':METADATA['Y_std'],
    # directories
    'unfiltered_data_dir':os.path.join('/mnt/sda1/Datasets/Workflow_ETA/data_prod/tmpDir/maindatafolder'),
    'datadir':datadir,
    'results_dir':results_dir,
    'plot_dir':os.path.join(results_dir,'plots'),
    'best_model_dir':os.path.join(results_dir,'best_model'),
    'tensorboard_dir':os.path.join(results_dir,'tensorboard'),
    # model param
    'loss':'mse',
    'epochs':3,
    'kfold_split':2,
    'test_split':0.2, # % out of whole dataset
    'fitness_result':'mean_squared_error', # result from keras model.history.history dict. Result will always be from validation set.
    'skopt_n_calls':11,
    'batch_size':4,
    # cnn
    'cnn_activation':'relu',
    'cnn_kernel_h':3,
    'cnn_kernel_w':3,
    'mp_kernel_h':2,
    'mp_kernel_w':2,
}


hyperparam_dimensions = [
    Real(        low=1e-6, high=1e-2, prior='log-uniform', name='learning_rate'                ),
    
    Integer(     low=   1, high=  10,                      name='cnn_init_channels'            ),
    Integer(     low=   0, high=   6,                      name='cnn_channel_increase_delta'   ),
    Integer(     low=   2, high=   5,                      name='cnn_layers'                   ),
    
    Categorical( categories= [1],                          name='lstm_layers'                  ),
    Categorical( categories= [LSTM],                       name='rnn_unit_function'            ),
    Integer(     low=  30, high= 150,                      name='lstm_units'                   ),
    Categorical( categories= [0,1,2],                      name='dense_layers_after_lstm'      ),
    Integer(     low=  10, high=  20,                      name='dense_layer_units_after_lstm' )
]


prior_hyperparam = {
    ## overall
    'learning_rate':0.001,

    ## cnn and mp (max pooling)
    'cnn_init_channels':5,
    'cnn_channel_increase_delta':3,
    'cnn_layers':3,

    ## lstm and dense (after lstm)
    'lstm_layers':1,
    'rnn_unit_function':LSTM,
    'lstm_units':40,
    'dense_layers_after_lstm':0,
    'dense_layer_units_after_lstm':10,
}

prior_hyperparam_value_only = [a for a in prior_hyperparam.values()]
prior_hyperparam_keys_only = [a for a in prior_hyperparam.keys()]


plotinfo = {
        'fitness':{
            'y_label':'fitness',
            'x_label':'epochs',
            'title':'fitness values',
            'img_name':'fitness.png'
        },
        'std':{
            'y_label':'std',
            'x_label':'epochs',
            'title':'prediction std from correct values in hours',
            'img_name':'std.png'
        },
        'predictions':{
            'y_label':'predictions',
            'x_label':'epochs',
            'title':'prediction values in hours',
            'img_name':'predictions.png'
        },
        'colors':{
            'train':'yellow',
            'valid':'blue',
            'test':'red'
        }
    }