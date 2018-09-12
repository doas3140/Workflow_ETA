import keras
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

import numpy as np
import os
import keras


class DataGenerator(keras.utils.Sequence):
    def __init__(self, indexes, batch_size, datadir):
        self.batch_size = batch_size
        self.indexes = indexes
        self.batches = []
        self.index2path = get_index2path_dict(datadir,indexes)
        self.num = len(self) * batch_size # self.num % batch_size = 0
        self.on_epoch_start()

    def __len__(self): # how many batches
        return len(self.indexes) // self.batch_size

    def __getitem__(self, batch_index):
        X = []
        Y = np.empty((self.batch_size), dtype=float)
        for i,index in enumerate(self.batches[batch_index]):
            x, Y[i] = np.load(self.index2path[index])
            X.append(x)
        return pad_sequences(X), Y

    def on_epoch_start(self):
        np.random.shuffle(self.indexes)
        indexes = self.indexes[ :self.num ]
        self.batches = np.split( indexes, indices_or_sections=len(self) )
    
    def get_y_array(self):
        Y = []
        for i in range(len(self)):
            x,y = self[i]
            Y.extend(y)
        return np.array(Y)


def get_index2path_dict(datadir,indexes=[]):
        ''' if indexes == []: gets all indexes '''
        index2path = {}
        folders = [f for f in os.scandir(datadir) if f.is_dir() and f.name != 'json']
        for folder in folders:
            files = [f for f in os.scandir(folder.path) if f.name[-4:] == '.npy']
            for f in files:
                index = int(f.name[:-4])
                if indexes == []:
                    index2path[index] = f.path
                elif index in indexes:
                    index2path[index] = f.path
        return index2path


def split_indexes(datadir,test_split):
    index2path = get_index2path_dict(datadir)
    all_indexes = [int(os.path.basename(path)[:-4]) for path in index2path.values()]
    data_indexes,test_indexes,_,_ = train_test_split(all_indexes,all_indexes,test_size=test_split)
    return np.array(data_indexes), np.array(test_indexes)



# # for keras 1.2
#     class DataGenerator():
#         def __init__(self, indexes, batch_size, datadir):
#             self.batch_size = batch_size
#             self.indexes = indexes
#             self.batches = []
#             self.index2path = get_index2path_dict(datadir,indexes)
#             self.num_batches = len(self.indexes) // self.batch_size
#             self.num = self.num_batches * batch_size # self.num % batch_size = 0
#             self.on_epoch_start()

#         def __len__(self): # how many samples
#             return self.num

#         def __getitem__(self, batch_index):
#             X = []
#             Y = np.empty((self.batch_size), dtype=int)
#             for i,index in enumerate(self.batches[batch_index]):
#                 x, Y[i] = np.load(self.index2path[index])
#                 X.append(x)
#             return pad_sequences(X), Y

#         def on_epoch_start(self):
#             np.random.shuffle(self.indexes)
#             indexes = self.indexes[ :self.num ]
#             self.batches = np.split( indexes, indices_or_sections=self.num_batches )
        
#         def generator_function(self):
#             def generator():
#                 self.on_epoch_start()
#                 for batch_index in range(self.num_batches):
#                     X = []
#                     Y = np.empty((self.batch_size), dtype=int)
#                     for i,index in enumerate(self.batches[batch_index]):
#                         x, Y[i] = np.load(self.index2path[index])
#                         X.append(x)
#                     yield pad_sequences(X), Y
#             return generator()