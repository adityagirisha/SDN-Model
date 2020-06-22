import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.utils import shuffle
import pickle


def read_csv(target_name, normalize=False):

    colnames=['flow_packet_count','flow_byte_count','rx_packets','tx_packets','rx_bytes','tx_bytes','Label']
    df=pd.read_csv("final.csv", names=colnames)
    df = df.iloc[1:]
    df = shuffle(df)
    if list(df.columns.values).count(target_name) != 1: 
        print("No target Label Found!")
        return
    target2idx = {target: idx for idx, target in enumerate(sorted(list(set(df[target_name].values))))}
    y = np.vectorize(lambda x: target2idx[x])(df[target_name].values)
    df = df.drop([target_name], axis=1).values
    X=preprocessing.normalize(df)
    n_classes = 2
    if X.shape[0] != y.shape[0]:
        raise Exception("X.shape = {} and y.shape = {} are inconsistent!".format(X.shape, y.shape))
    
    return X, y, n_classes

def storeNN(i,payload):
    file_name="models/save_file"+str(i+1)+".pkl"
    with open(file_name, 'wb') as output:
        pickle.dump(payload, output, pickle.HIGHEST_PROTOCOL)
    print("uploaded")
    return 
 
def loadNN(i):
    file_name="models/save_file"+str(i+1)+".pkl"
    with open(file_name, 'rb') as input:
        payload = pickle.load(input)
        return payload

    
def crossval_folds(N, n_folds, seed=1):
    np.random.seed(seed)
    idx_all_permute = np.random.permutation(N)
    N_fold = int((N/n_folds))
    idx_folds = []
    for i in range(n_folds):
        start = i*N_fold
        end = min([(i+1)*N_fold, N])
        idx_folds.append(idx_all_permute[start:end])
    return idx_folds