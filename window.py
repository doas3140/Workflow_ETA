
import numpy as np
import os
import json

def read_json(timestep):
    jsonpath = timestep
    return {} #TODO

def get_all_wf_names(timesteps):
    names = []
    for t in timesteps:
        jsondict = read_json(t)
        for wf in jsondict.keys():
            if wf not in names:
                names.append( wf )
    return names

def get_all_task_names(desired_wf,timesteps):
    names = []
    for t in timesteps:
        jsondict = read_json(t)
        for wf in jsondict.keys():
            if wf == desired_wf:
                for task_index, task in jsondict[wf].items():
                    if task_index != 'npy_index':
                        if task not in names:
                            names.append( task )
    name2index = { n:i for i,n in enumerate(names) }
    return names,name2index
    

def create_time_task_matrix(desired_wf,timesteps):
    ''' returns matrix (tasks,timesteps,2), where 2 = (npy file name number,index of graph sequence) '''
    task_names,task2index = get_all_task_names(desired_wf,timesteps)
    timestep2index = { t:i for i,t in enumerate(timesteps) }
    matrix = np.zeros( (len(task_names), len(timesteps), 2) )
    for t in timesteps:
        jsondict = read_json(t) # maybe should read once xd
        for wf in jsondict.keys():
            if wf == desired_wf:
                for task_index, task in jsondict[wf].items():
                    if task_index != 'npy_index':
                        matrix[ task2index[task] ][ timestep2index[t] ] = (jsondict[wf]['npy_index'],task_index)
    return matrix,task_names,timesteps

def a(rootfolder): # 'data' folder
    folders = ['1','2','3','4']
    folderpaths = [ os.path.join(rootfolder,folder) for folder in folders ]
    workflow_names = get_all_wf_names(folders)
    for wf in workflow_names:
        matrix,index2task,index2timestep = create_time_task_matrix(wf,folders)

