import numpy as np

def load():
    x_train = np.load('dataset/train_fog96/npy/train_GT.npy')
    x_test = np.load('dataset/train_fog96/npy/test_GT.npy')
    x_train_fog = np.load('dataset/train_fog96/npy/train_fog.npy')
    x_test_fog = np.load('dataset/train_fog96/npy/test_fog.npy')
    return x_train, x_test, x_train_fog, x_test_fog

