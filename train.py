
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


def main():
    data_indexes, test_indexes = split_indexes(const['datadir'],const['test_split'])
    model = create_model(p,const,print_summary=False)
    history = fit_kfold_model(model, data_indexes, test_indexes, const, p, verbose=1)
    save_history_plots(history,plotinfo,folderpath=os.path.join(const['plot_dir']))
    model.save(const['best_model_dir'])
    

if __name__ == '__main__':
    main()