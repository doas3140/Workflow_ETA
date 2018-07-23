from model import create_model, fit_model
from parameters import dimensions # delete

from tensorflow.python.keras.layers import SimpleRNN, GRU, LSTM
from skopt.space import Real, Categorical, Integer

import itertools
import numpy as np

def selectdimensions(dimensions,selected_dim_names): # names of selected dimensions
    output = []
    for d in dimensions:
        if d.name in selected_dim_names:
            output.append(d)
    return output

def dimensions2paramcategories(dimensions):
    # for each dimension creates a list of possible values, ex: {'learning_rate':[1e-2,1e-3,...],...}
    parameters = {}
    for d in dimensions:
        if isinstance(d,Real):
            low,high = d.bounds
            categories = [ low, np.mean([low,high]), high ]
            parameters[d.name] = categories
        elif isinstance(d,Categorical):
            categories = d.categories
            parameters[d.name] = categories
        elif isinstance(d,Integer):
            low,high = d.bounds
            a = np.arange(low,high+1)
            if len(a) > 10:
                categories = [int(a.min()),int(a.mean()),int(a.max())]
            else:
                categories = a
            parameters[d.name] = categories
    return parameters

def parameters2combinations(parameters):
    # creates all possible combinations [comb1,comb2,...]
    categories_list = [a for a in parameters.values()]
    combinations = []
    for comb in itertools.product(*categories_list):
        combinations.append(comb)
    print('Total combinations:',len(combinations))
    return combinations

def appendparameters(prior_params,parameters):
    '''
    appends parameters to prior_parameters by replacing all vars that are in parameters dict
    '''
    for i,(name,value) in enumerate(parameters.items()):
        prior_params[name] = value
    return prior_params

def create_test_data(X_shape,Y_shape): # shape w/out batch
    x = np.expand_dims( np.ones(X_shape) ,axis=0)
    y = np.expand_dims( np.ones(Y_shape) ,axis=0)
    return x,y, x,y

def test_model_params(dimensions,prior_params,const,X_shape,Y_shape):
    '''
    dimensions - list w/ all possbile dimensions in skopt variables
    prior_params - list of default parameters
    '''
    parameters = dimensions2paramcategories(dimensions)
    combinations = parameters2combinations(parameters)
    index2name = [name for name in parameters.keys() ] # index of combinations list -> name of parameter
    print('Testing model parameters...')
    X_train,Y_train, X_valid,Y_valid = create_test_data(X_shape,Y_shape=Y_shape)
    for comb in combinations[:1]:
        p_comb = {index2name[i]:c for i,c in enumerate(comb)}
        p = appendparameters(prior_params,p_comb) # every param is left unchanged unless it is in comb
        model = create_model(p,const)
        fit_model(model,1,p, X_train,Y_train, X_valid,Y_valid,batch_size=const['batch_size'],verbose=0)
    print('Tests passed...')
        
        
        
    
    
# selected_dim_names = ['cnn_kernel_h','cnn_kernel_2','cnn_strides_h','cnn_strides_w','cnn_layers']
# selected_dimensions = selectdimensions(dimensions,selected_dim_names)
# test_model_params(0,selected_dimensions)

# combinations = []
# for combination in itertools.product(*only_cat):
#     combinations.append(combination)
# print(len(combinations))
# print(combinations[0])

# a,b = create_test_data(X[0].shape,(1))
# print(a.shape,b.shape)

# a = np.array([1,2,3,4,5,6])
# print(a[ [0,2,3] ])