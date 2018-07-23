from tensorflow.python.keras.layers import SimpleRNN, GRU, LSTM
from skopt.space import Real, Categorical, Integer


''' PARAMETERS
# overall
learning_rate - 

# cnn and mp (max pooling)
cnn_init_channels - 
cnn_channel_increase_delta - 
cnn_kernel_h - 
cnn_kernel_w - 
cnn_strides_h - 
cnn_strides_w - 
cnn_activation - 
cnn_layers - (mp layer is always after cnn)
mp_kernel_h - 
mp_kernel_w - 

# lstm and dense (after lstm)
lstm_layers - 
rnn_unit_function - 
lstm_units - 
dense_layers_after_lstm - 
dense_layer_units_after_lstm - 
'''
# constant variables
const = {
    'loss':'mse',
    'epochs':1,
    'kfold_split':2,
    'test_split':0.2, # % out of whole dataset
    'fitness_result':'val_mean_absolute_error', # result from keras model.history.history dict
    'skopt_n_calls':11,
    'batch_size':1
}

dimensions = [
    Real(        low=1e-6, high=1e-2, prior='log-uniform', name='learning_rate' ),
    
    Integer(     low=   1, high=  10,                      name='cnn_init_channels' ),
    Integer(     low=   0, high=   6,                      name='cnn_channel_increase_delta' ),
    Integer(     low=   1, high=   4,                      name='cnn_kernel_h' ),
    Integer(     low=   1, high=   7,                      name='cnn_kernel_w' ),
    Categorical( categories= [2],                          name='cnn_strides_h' ),
    Categorical( categories= [2],                          name='cnn_strides_w' ),
    Categorical( categories= ['relu'],                     name='cnn_activation' ),
    Integer(     low=   2, high=   5,                      name='cnn_layers' ),
    #Integer(     low=   2, high=   4,                      name='mp_kernel_h' ),
    #Integer(     low=   2, high=   4,                      name='mp_kernel_w' ),
    
    # Integer(     low=   1, high=   2,                      name='lstm_layers' ),
    Categorical( categories= [1],                          name='lstm_layers'),

    Categorical( categories= [SimpleRNN,LSTM,GRU],         name='rnn_unit_function' ),
    Integer(     low=  30, high= 150,                      name='lstm_units' ),
    Integer(     low=   0, high=   1,                      name='dense_layers_after_lstm' ),
    Integer(     low=  10, high=  20,                      name='dense_layer_units_after_lstm' )
]

prior_params = {
    # # overall
    'learning_rate':0.001, # learning_rate - 

    # # cnn and mp (max pooling)
    'cnn_init_channels':5, # cnn_init_channels - 
    'cnn_channel_increase_delta':3, # cnn_channel_increase_delta - 
    'cnn_kernel_h':3, # cnn_kernel_h - 
    'cnn_kernel_w':3, # cnn_kernel_w - 
    'cnn_strides_h':2, # cnn_strides_h - 
    'cnn_strides_w':2, # cnn_strides_w - 
    'cnn_activation':'relu', # cnn_activation - 
    'cnn_layers':3, # cnn_layers - (mp layer is always after cnn)
    #'mp_kernel_h':2, # mp_kernel_h -999 
    #'mp_hernel_w':2, # mp_kernel_w - 

    # # lstm and dense (after lstm)
    'lstm_layers':1, # lstm_layers - 
    'rnn_unit_function':GRU, # rnn_unit_function - 
    'lstm_units':40, # lstm_units - 
    'dense_layers_after_lstm':0, # dense_layers_after_lstm - 
    'dense_layer_units_after_lstm':10, # dense_layer_units_after_lstm - 
}

prior_params_value_only = [a for a in prior_params.values()]