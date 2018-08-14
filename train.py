
from model import fit_kfold_model, create_model
from parameters import const_param as const
from parameters import hyperparam_dimensions as dimensions
from parameters import prior_hyperparam as p
from parameters import prior_hyperparam_value_only as prior_values
from parameters import prior_hyperparam_keys_only as prior_names
from data import DataGenerator, split_indexes
from visualization import save_history_plots, save_skopt_plots
from parameters import plotinfo

import os
import numpy as np
import json
import time
import pickle


def main():
    data_indexes, test_indexes = split_indexes(const['datadir'],const['test_split'])
    model = create_model(p,const,print_summary=False)
    keras_histories,history = fit_kfold_model(model, data_indexes, test_indexes, const, p, verbose=1)
    save_histories(keras_histories,folderpath=os.path.join(const['results_dir'],'keras_histories'),name='keras_histories')
    save_histories(history,folderpath=os.path.join(const['results_dir'],'plot_histories'),name='plot_history')
    # save_history_plots(history,plotinfo,folderpath=os.path.join(const['plot_dir']))
    model.save(const['best_model_dir'])

def save_histories(histories,folderpath,name):
    if not os.path.exists(folderpath): os.makedirs(folderpath)
    pickle.dump(histories,open(os.path.join(folderpath,name+'.pickle'),'wb'))


    

if __name__ == '__main__':
    main()