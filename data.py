import numpy as np
from sklearn.model_selection import train_test_split

def preprocess_data(X,Y):
    # normalize Y
    Y_mean = Y.mean()
    Y_std = Y.std()
    Y = (Y-Y_mean)/Y_std
    # normalize X
    X /= 255
    return X,Y,Y_mean,Y_std

def split_data(X,Y,train_split,valid_split,test_split):
    ''' UNUSED '''
    if not train_split + valid_split + test_split == 1:
        raise Exception('Sum of all splits must be equal to 1')
    X,X_test,Y,Y_test = train_test_split(X,Y,test_size=test_split)
    left_split = valid_split/(1-test_split) # or valid_split/(train_split+valid_split)
    X_train,X_valid,Y_train,Y_valid = train_test_split(X,Y,test_size=left_split)
    return X_train,Y_train, X_valid,Y_valid, X_test,Y_test

def get_data(npy_path,test_split):
    X,Y = np.load(npy_path)
    X,Y,Y_mean,Y_std = preprocess_data(X,Y)
    X,X_test,Y,Y_test = train_test_split(X,Y,test_size=test_split)
    return X,Y, X_test,Y_test # X,Y - {train set + valid set}

def index2data(X,Y,train_index,valid_index):
    X_train, X_valid = X[train_index], X[valid_index]
    Y_train, Y_valid = Y[train_index], Y[valid_index]
    return X_train,Y_train, X_valid, Y_valid