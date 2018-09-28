'''
    Functions for Loading and Filtering Data
'''

from parameters import const_param as const

import json
import os
import numbers
from datetime import datetime
import numpy as np
from PIL import Image
import sys
from tqdm import tqdm


def parsejson(timestamp_path):
    ''' jsondict = {'wf':{task1:{...},task2:{...},completed:123,assigned:123,...],...} '''
    jsondict = {}; checklist = []; 
    json_paths = [f.path for f in os.scandir(timestamp_path) if f.name[-5:]=='.json']
    for jsonpath in json_paths:
        j = json.loads( open(jsonpath,'r').read() )
        for wf,aa in j.items():
            jsondict[wf] = aa
            if wf in checklist: print('ERROR: two workflows with the same name',jsonpath)
            else: checklist.append(wf)
    return jsondict


def get_all_workflow_data(rootpath,savedata_path):
    ''' wfdata = {'wf':{completed:123,assigned:123,(all who results are numbers)},...} '''
    wfdata = {}
    folderpaths = [f.path for f in os.scandir(rootpath) if f.is_dir() and f.name != os.path.basename(savedata_path)]
    for timestamp_path in folderpaths:
        jsondict = parsejson( timestamp_path )
        for wf,aa in jsondict.items():
            if not wf in wfdata: wfdata[wf] = {}
            for status,timestamp in aa.items():
                if isinstance(timestamp,numbers.Integral): # if is number (means it's not task)
                    if status in wfdata[wf]:
                        if not wfdata[wf][status] == timestamp:
                            print('Different value in different timestamps for status:',status)
                    else:
                        wfdata[wf][status] = timestamp
    return wfdata


def get_all_targets_mean_std(rootpath,wfdata,savedata_path,filter_graphs=False):
    print('\n Calculating mean and std of Y')
    targets = []
    folderpaths = [f.path for f in os.scandir(rootpath) if f.is_dir() and f.name != os.path.basename(savedata_path)]
    for timestamp_path in tqdm(folderpaths):
        jsondict = parsejson( timestamp_path )
        for wf in tqdm(jsondict.keys(),total=len(jsondict)):
            if 'completed' in wfdata[wf]:
                skip_graphs = not filter_graphs
                graph_npys,graph_timestamps,graph_metadata = get_graphs_timestamps_meta(timestamp_path,jsondict,wf,skip_graphs)
                if graph_timestamps == []:
                    continue
                completed_timestamp = wfdata[wf]['completed']
                target = calculate_target(completed_timestamp,graph_timestamps)
                if filter_rules(graph_npys,target):
                    targets.append( target )
    Y_mean, Y_std = np.mean(targets), np.std(targets)
    print('\n Finished calculating... Y_mean: {:.10}, Y_std: {:.10}'.format(Y_mean,Y_std))
    return Y_mean, Y_std


def check_if_empty( im_reg, empty_im_path=os.path.join(os.getcwd(),'empty.png'), crop_box=(70,35,320,130) ):
    '''
    im_reg - PIL Image w/ region (70,35,320,130) or can be np array
    empty_unique can load from image path or manually creating array of values
    '''
    global empty_unique
    if 'empty_unique' not in globals():
        if True: # manualy load unique pixel values (if True)
            empty_unique = [145,151,169,178,191,193,197,201,213,217,218,220,221,230,235,236,237,243,251,255]
        else:
            empty_im = Image.open( empty_im_path )
            empty_reg = empty_im.crop(crop_box)
            empty_unique = np.unique(np.array(empty_reg)) 
    unique = np.unique(np.array(im_reg))
    if np.array_equal(unique,empty_unique):
        return True
    else:
        return False


def preprocess_image(npypath):
    '''
    return str('ignore') to skip this image
    '''
    global stats
    crop_box = (70,35,320,130)
    try:
        npy = np.load(npypath)
    except FileNotFoundError:
        print('Npy file doesnt exist:',npypath)
        stats['file_not_found'] += 1
        return 'ignore'
    im = Image.fromarray(npy)
    im_reg = im.crop(crop_box)
    if check_if_empty(im_reg):
        stats['empty_images'] += 1
        # return 'ignore'
    return np.array(im_reg) / 255


def preprocess_target(target,mean,std):
    return (target - mean)/std


def all_graphs_are_empty(graph_npys):
    for npy in graph_npys:
        is_empty = check_if_empty(npy)
        if not is_empty:
            return False
    return True


def filter_rules(graph_npys,target):
    '''
    graph_npys - list of graphs (to filter particular graph check preprocess_image function)
    target - (completed_date - graph_date) in seconds
    if one input is None then its ignored
    if both are None then returns False
    '''
    global stats
    if graph_npys is not None:
        if all_graphs_are_empty(graph_npys):
            stats['workflows_where_all_graphs_are_empty'] += 1
            return False
    if target is not None:
        if target < 0:
            return False
    if graph_npys is None and target is None:
        return False
    return True


def createdata(timestamp_path,wfdata,Y_mean,Y_std):
    '''
    npydata - has task graph npy paths and timestamps when graphs were made
    wfdata - has all statuses of workflows
    '''
    global stats
    X = []; Y = []; METADATA = []; # METADATA - info about X (task,wf,etc)
    jsondict = parsejson( timestamp_path )
    for wf in tqdm(jsondict.keys(),total=len(jsondict)):
        stats['total_workflows'] += 1
        if 'completed' in wfdata[wf]: # if it is completed
            graph_npys,graph_timestamps,graph_metadata = get_graphs_timestamps_meta(timestamp_path,jsondict,wf,skip_graphs=False)
            if graph_timestamps == []:
                continue
            completed_timestamp = wfdata[wf]['completed']
            target = calculate_target(completed_timestamp,graph_timestamps)
            if filter_rules(graph_npys,target):
                X.append(graph_npys)
                target = preprocess_target(target,Y_mean,Y_std)
                Y.append(target)
                METADATA.append(graph_metadata)
            else:
                stats['workflows_after_completion'] += 1
        else:
            stats['not_completed_workflows'] += 1
    stats['saved_workflows'] = len(Y)
    return X, Y, METADATA


def calculate_target(completed_timestamp,graph_timestamps):
    completed_date = datetime.fromtimestamp( completed_timestamp )
    avg_graph_timestamp = np.mean(graph_timestamps)
    graph_date = datetime.fromtimestamp( avg_graph_timestamp )
    target = int( (completed_date - graph_date).total_seconds() )
    return target


def get_graphs_timestamps_meta(timestamp_path,jsondict,wf,skip_graphs=False):
    '''
    graph_npys = []
    graph_timestamps = [int,int,int] timestamps 
    graph_metadata = [{'workflow':wf,'task':task},...]
    '''
    global stats
    graph_npys = []; graph_timestamps = []; graph_metadata = []; 
    if not skip_graphs:
        stats['total_images'] = 0
    stats['graphs_are_all_empty_error'] = 0
    stats['there_are_no_timestamps_error'] = 0
    for task,bb in jsondict[wf].items():
        if not isinstance(bb,numbers.Integral): # if is not number (means it's task)
            for strdate,npypath in bb.items(): # can be w/out for
                if not skip_graphs:
                    stats['total_images'] += 1
                    npypath = os.path.normpath( os.path.join( timestamp_path, os.path.normpath(npypath) ) )
                    im = preprocess_image(npypath)
                    if isinstance(im,str):
                        if im == 'ignore': continue
                    graph_npys.append( im )
                graphdate_start = datetime.strptime(strdate[:26],'%Y-%m-%d %H:%M:%S.%f')
                graphdate_end = datetime.strptime(strdate[30:],'%Y-%m-%d %H:%M:%S.%f')
                graph_timestamps.append( graphdate_end.timestamp() )
                graph_metadata.append( {'workflow':wf, 'task':task} )
    if graph_timestamps == []:
        stats['there_are_no_timestamps_error'] += 1
        # raise Exception('ERROR: graphs are all empty, shouldnt be happening!')
    if graph_npys == [] and skip_graphs == False:
        stats['graphs_are_all_empty_error'] += 1
        # raise Exception('ERROR: there are no timestamps, shouldnt be happening!')
    if graph_npys != []: graph_npys = np.array(graph_npys)
    else: graph_npys = None
    return graph_npys, graph_timestamps, graph_metadata


def create_meta_json(METADATA,data_indexes,timestamp_path,folderpath):
    jsondict = {}
    for i,meta in enumerate(METADATA):
        index = int(data_indexes[i])
        for j,graph_meta in enumerate(meta):
            if j == 0: 
                jsondict[ graph_meta['workflow'] ] = {}
                jsondict[ graph_meta['workflow'] ]['npy_index'] = index
            jsondict[ graph_meta['workflow'] ][j] = graph_meta['task']
    timestamp_name = os.path.basename(timestamp_path)
    jsonpath = os.path.join(folderpath,'json',timestamp_name+'.json')
    if not os.path.exists(os.path.dirname(jsonpath)): os.makedirs(os.path.dirname(jsonpath))
    json.dump( jsondict, open(jsonpath,'w'), indent=4 )


def save_timestamp_data(X,Y,METADATA,timestamp_name,folderpath):
    global data_index
    if not os.path.exists(folderpath): os.makedirs(folderpath)
    timestamp_path = os.path.join(folderpath,timestamp_name)
    if not os.path.exists(timestamp_path): os.makedirs(timestamp_path)
    start_index = data_index
    for x,y,meta in zip(X,Y,METADATA):
        npypath = os.path.join(timestamp_path,str(data_index)+'.npy')
        np.save(npypath,(x,y))
        data_index += 1
    end_index = data_index
    create_meta_json(METADATA,np.arange(start_index,end_index),timestamp_path,folderpath)


def empty_stats():
    stats = {
        'total_images':0,
        'empty_images':0,
        'file_not_found':0,
        'total_workflows':0,
        'workflows_where_all_graphs_are_empty':0,
        'not_completed_workflows':0,
        'workflows_after_completion':0,
        'saved_workflows':0,
        'there_are_no_timestamps_error':0,
        'graphs_are_all_empty_error':0
    }
    return stats


def check_if_folder_has_json_file(folderpath):
    return len([f for f in os.listdir(folderpath) if f[-5:] == '.json']) > 0


import matplotlib.pyplot as plt

def main(rootpath,savedata_path):
    ''' go through all folders and create filtered data to datapath '''
    global stats; all_stats = {}; empty_stats(); 
    delete_csvs(folderpath=savedata_path)
    folderpaths = [f.path for f in os.scandir(rootpath) if f.is_dir() and f.name != os.path.basename(savedata_path)]
    folderpaths = list(filter(check_if_folder_has_json_file,folderpaths))
    wfdata = get_all_workflow_data(rootpath,savedata_path)
    Y_mean, Y_std = get_all_targets_mean_std(rootpath,wfdata,savedata_path)
    for timestamp_path in tqdm(folderpaths):
        stats = empty_stats()
        X,Y,METADATA = createdata(timestamp_path,wfdata,Y_mean,Y_std)
        save_timestamp_data( X, Y, METADATA, timestamp_name=os.path.basename(timestamp_path), folderpath=savedata_path )
        all_stats[os.path.basename(timestamp_path)] = stats
    save_constants(Y_mean,Y_std,jsonpath=os.path.join(savedata_path,'meta.json'))
    return all_stats


def save_constants(Y_mean,Y_std,jsonpath):
    jsondict = {
        'Y_mean':Y_mean,
        'Y_std':Y_std
    }
    json.dump( jsondict, open(jsonpath,'w'), indent=4 )


# def print_stats(all_stats):
#     # independent stats
#     for timestamp_name,timestamp_stats in all_stats.items():
#         print('\n\n')
#         print('::: timestamp:',timestamp_name,':::')
#         for status,number in timestamp_stats.items():
#             print('{} : {}'.format(status,number))
#     # overall stats
#     summed_stats = empty_stats()
#     for timestamp_name,timestamp_stats in all_stats.items():
#         for status,number in timestamp_stats.items():
#             summed_stats[status] += number
#     print('\n\n')
#     print('::: SUMMED STATS :::')
#     for status,number in summed_stats.items():
#         print('{} : {}'.format(status,number))

def get_stats_summary_string(all_stats,savepath=False):
    s = ''
    # independent stats
    for timestamp_name,timestamp_stats in all_stats.items():
        s += '\n\n\n'
        s += '::: timestamp: '+ timestamp_name +' :::\n'
        for status,number in timestamp_stats.items():
            s += '{} : {}\n'.format(status,number)
    # overall stats
    summed_stats = empty_stats()
    for timestamp_name,timestamp_stats in all_stats.items():
        for status,number in timestamp_stats.items():
            summed_stats[status] += number
    s += '\n\n\n'
    s += '::: SUMMED STATS :::\n'
    for status,number in summed_stats.items():
        s += '{} : {}\n'.format(status,number)
    if savepath:
        with open(savepath,'w') as f:
            f.write(s)
    return s

def delete_csvs(folderpath):
    csv_paths = [f.path for f in os.scandir(folderpath) if f.name[-4:]=='.csv']
    for csv_path in csv_paths:
        os.remove(csv_path)


if __name__ == '__main__':
    global stats; data_index = 0
    all_stats = main( const['unfiltered_data_dir'], const['datadir'] )
    s = get_stats_summary_string(all_stats,savepath=os.path.join(const['datadir'],'summary.txt'))
    print(s)