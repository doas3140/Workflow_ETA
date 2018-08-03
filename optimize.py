
from model import fit_kfold_model, create_model
from parameters import const_param as const
from parameters import hyperparam_dimensions as dimensions
from parameters import prior_hyperparam as prior
from parameters import prior_hyperparam_value_only as prior_values
from parameters import prior_hyperparam_keys_only as prior_names
from data import DataGenerator, get_index2path_dict, split_indexes
from visualization import save_history_plots, save_skopt_plots
from parameters import plotinfo

from keras import backend as K
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from skopt import gp_minimize
from skopt.utils import use_named_args

import os
import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import timedelta
import time


def main():
    global data_indexes, test_indexes, smallest_result, num_call, skopt_history; 
    num_call, skopt_history, smallest_result = 0, [], 999; 
    data_indexes, test_indexes = split_indexes(const['datadir'],const['test_split'])
    search_result = gp_minimize(
                                    func = fitness,
                                    dimensions = dimensions,
                                    acq_func = 'EI', # Expected Improvement.
                                    n_calls = const['skopt_n_calls'],
                                    x0 = prior_values
                               )
    skopt_result_string = create_skopt_results_string(search_result,prior_names,savepath=os.path.join(const['results_dir'],'optimization_info.txt'))
    save_skopt_history_plots(skopt_history,plotinfo,datadir=os.path.join(const['plot_dir'],'skopt'))
    print(skopt_result_string)
    # save_skopt_plots(const['plot_dir'],search_result,prior_names)


@use_named_args(dimensions)
def fitness(**p): # p = { 'p1':0.1,'p2':3,... }
    global const, data_indexes, test_indexes, num_call, skopt_history
    print('\n \t ::: {} SKOPT CALL ::: \n'.format(num_call+1))
    model = create_model(p,const,print_summary=False)
    history = fit_kfold_model(model, data_indexes, test_indexes, const, p, verbose=1)
    save_history_plots(history,plotinfo,folderpath=os.path.join(const['plot_dir'],str(num_call)))
    result = history['fitness']['valid']['mean'][-1] # last epoch
    save_best_model(model,result,const)
    num_call += 1
    skopt_history.append(history)
    return result


def save_skopt_history_plots(histories,plotinfo,datadir):
    mean_history, mean_history_epochs = create_skopt_mean_history(histories)
    save_history_plots(mean_history_epochs,plotinfo,folderpath=os.path.join(datadir,'epochs'))
    plotinfo['fitness']['x_label'], plotinfo['std']['x_label'], plotinfo['predictions']['x_label'] = \
        'skopt calls', 'skopt calls', 'skopt calls'
    save_history_plots(mean_history,plotinfo,folderpath=os.path.join(datadir,'skopt_calls'))


def get_best_params(search_result,names):
    outputDict = {}
    for i,nr in enumerate(search_result.x):
        outputDict[ names[i] ] = str(nr)
    return outputDict


def save_best_model(model,result,const):
    global smallest_result
    if result < smallest_result:
        model.save(const['best_model_dir'])
        smallest_result = result
    

def create_skopt_mean_history(histories):
    # create dict
    mean_history = {}; mean_history_epochs = {}; # one where x axis is epochs, and one where x axis is # of skopt calls
    for history in histories:
        for metric_name,y_dict in history.items():
            mean_history[metric_name] = {}
            mean_history_epochs[metric_name] = {}
            for data_name in y_dict.keys():
                mean_history[metric_name][data_name] = {}
                mean_history[metric_name][data_name]['list'] = []
                mean_history_epochs[metric_name][data_name] = {}
                mean_history_epochs[metric_name][data_name]['list'] = []
    # append mean values
    for history in histories:
        for metric_name,y_dict in history.items():
            for data_name in y_dict.keys():
                mean_history[metric_name][data_name]['list'].append(y_dict[data_name]['mean'])
    # calculate mean and std
    for history in histories:
        for metric_name,y_dict in history.items():
            for data_name in y_dict.keys():
                mean_history[metric_name][data_name]['mean'] = np.mean(mean_history[metric_name][data_name]['list'],axis=1)
                mean_history[metric_name][data_name]['std'] = np.std(mean_history[metric_name][data_name]['list'],axis=1)
                mean_history_epochs[metric_name][data_name]['mean'] = np.mean(mean_history[metric_name][data_name]['list'],axis=0)
                mean_history_epochs[metric_name][data_name]['std'] = np.std(mean_history[metric_name][data_name]['list'],axis=0)
    return mean_history, mean_history_epochs


def create_skopt_results_string(search_result,prior_names,savepath=False):
    # print(search_result)
    s = ''
    s += '::: ALL PARAMETERS :::\n'
    sorted_results = sorted(zip(search_result.func_vals, search_result.x_iters))
    for name in prior_names:
        s += '{} '.format(name)
    s += '\n'
    for fitness_value,parameter_values in sorted_results:
        s += '{:.8}'.format(fitness_value)
        for x in parameter_values:
            x = str(x)
            if len(x) < 10:
                s += '{:>10} '.format(x)
            else:
                s += '{:>40}'.format(x)
        s += '\n'
    s += '::: BEST SCORE :::\n'
    s += str(search_result.fun) + '\n'
    s += '::: BEST PARAM :::\n'
    best_param = get_best_params(search_result,prior_names)
    s += json.dumps(best_param,indent=4)
    if savepath:
        with open(savepath,'w') as f:
            f.write(s)
    return s


if __name__ == '__main__':
    main()