from parameters import const, dimensions, prior_params, prior_params_value_only
from model import create_model, logdir_path, tensorboard, fit_model
from data import index2data, get_data
from tests import test_model_params, selectdimensions

from tensorflow.python.keras import backend as K
from sklearn.model_selection import KFold
from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.plots import plot_convergence

import os
import numpy as np


def main():

    global X,Y, X_test,Y_test
    X,Y, X_test,Y_test = get_data(npy_path=os.path.join(os.getcwd(),'..','data','data.npy'), test_split=const['test_split'])

    selected_dim_names = ['cnn_kernel_h','cnn_kernel_2','cnn_strides_h','cnn_strides_w','cnn_layers']
    selected_dimensions = selectdimensions(dimensions,selected_dim_names)
    test_model_params( selected_dimensions, prior_params, const, X_shape=X[0].shape, Y_shape=(1) )

    global smallest_result; smallest_result = 999
    @use_named_args(dimensions)
    def fitness(**p): # p = { 'p1':0.1,'p2':3,... }
        # global const
        model = create_model(p,const,print_summary=False)
        tensorboard_logdir_path = logdir_path(folder_path=os.getcwd(),p=p)
        tensorboard_callback = tensorboard(tensorboard_logdir_path, batch_size=const['batch_size'])
        global X,Y, X_train,Y_train, X_valid,Y_valid # X,Y = {train set + valid set}
        result_list = []
        for train_index, valid_index in KFold(n_splits=const['kfold_split'],shuffle=True).split(X):
            X_train,Y_train, X_valid,Y_valid = index2data(X,Y,train_index,valid_index)
            history = fit_model(model,const['epochs'],p, X_train,Y_train, X_valid,Y_valid, batch_size=const['batch_size']) # 'val_loss','val_mean_squared_error','val_mean_squared_error' + w/out 'val'
            result_list.append( history.history[ const['fitness_result'] ][-1] ) # -1 = last epoch
        global smallest_result # mean absolute error
        result = np.mean(result_list)
        if result < smallest_result:
            model.save(os.path.join(os.getcwd(),'best_model'))
            smallest_result = result
        del model; K.clear_session()
        return result

    search_result = gp_minimize(
                                    func = fitness,
                                    dimensions = dimensions,
                                    acq_func = 'EI', # Expected Improvement.
                                    n_calls = const['skopt_n_calls'],
                                    x0 = prior_params_value_only
                               )
    best_params = search_result.space.point_to_dict(search_result.x) # { 'p1':1,'p2':2,... }
    plot_convergence(search_result)
    sorted_results = sorted(zip(search_result.func_vals, search_result.x_iters))

if __name__ == '__main__':
    main()