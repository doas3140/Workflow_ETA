'''
    Some Visualization Functions
'''
from skopt.plots import plot_convergence, plot_evaluations, plot_objective
from matplotlib import pyplot as plt

import numpy as np
import matplotlib
import os
import json


matplotlib.rc('figure', figsize = (14, 7)) # Plot size to 14" x 7"
matplotlib.rc('font', size = 14) # Font size to 14
matplotlib.rc('axes.spines', top = False, right = False) # Do not display top and right frame lines
matplotlib.rc('axes', grid = False) # Remove grid lines
matplotlib.rc('axes', facecolor = 'white') # Set backgound color to white


def plot_multiple_mean_std(x,y_dict,y_label,x_label,title,colors_dict,savepath=False):
    _, ax = plt.subplots()
    for name,data in y_dict.items():
        mean,std,color = data['mean'],data['std'],colors_dict[name]
        ax.plot(x, mean, lw = 3, color=color, alpha = 0.8, label = name)
        y_low = mean - std
        y_high = mean + std
        ax.fill_between(x, y_low, y_high, color=color, alpha = 0.1)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(loc = 'best')
    if savepath: plt.savefig(savepath)
    plt.close()


def save_history_plots(history,plotinfo,folderpath):
    """ history - {'metric_name':{'train':shape(n,epochs),'test':shape(n,epochs),...},...}
    \n  this will take np.mean((n,epochs),axis=0) and std and plot each metric graph
    \n  make sure n is atleast 1, so the shape wont be (epochs)
    \n  plotinfo - {'metric_name':{'y_label':'','x_label':'','title':'','imgname':''},...},...}
    """
    if not os.path.exists(folderpath): os.makedirs(folderpath)
    num_epochs = len(history['fitness']['train']['mean'])
    for metric_name,y_dict in history.items():
        plot_multiple_mean_std(
            x=np.arange(num_epochs), y_dict=y_dict,
            y_label=plotinfo[metric_name]['y_label'], x_label=plotinfo[metric_name]['x_label'], title=plotinfo[metric_name]['title'],
            colors_dict=plotinfo['colors'],
            savepath=os.path.join(folderpath,plotinfo[metric_name]['img_name'])
            )
            

def save_skopt_plots(dirpath,search_result,prior_names):
    if not os.path.exists(dirpath): os.makedirs(dirpath)
    # ---- Evalution
    plot_evaluations(search_result, bins=20)
    plt.savefig( os.path.join(dirpath,'evaluation_plot.png') )
    # ---- Convergence (previously looked better enquire what is going on)
    plot_convergence(search_result)
    plt.savefig( os.path.join(dirpath,'convergence_plot.png') )
    # ---- Partial Dependence plots are only approximations of the modelled fitness function 
    # - which in turn is only an approximation of the true fitness function in fitness
    plot_objective(result=search_result)
    plt.savefig( os.path.join(dirpath,'objective_plot.png') )
