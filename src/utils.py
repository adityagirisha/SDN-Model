import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.utils import shuffle
import pickle


def read_csv(target_name, normalize=False):
    colnames = [ 'Source IP',  'Source Port',  'Destination IP',  'Destination Port',  'Protocol'
             , 'Timestamp',  'Flow Duration',  'Total Fwd Packets',  'Total Backward Packets', 
             'Total Length of Fwd Packets', 'Total Length of Bwd Packets', 'Flow Packets/s', 'Fwd Packets/s',  
             'Bwd Packets/s',  'Average Packet Size',  'Fwd Header Length.1',  'Label']
#     data11=pd.read_csv("data/day1/syn11.csv", names=colnames,dtype={'Source IP':str,'Destination IP':str})
#     data12=pd.read_csv("data/day1/syn12.csv", names=colnames,dtype={'Source IP':str,'Destination IP':str})
#     data11 = data11.iloc[1:]
#     data12 = data12.iloc[1:]
#     frames=[data11,data12]
#     df = pd.concat(frames, axis=0, join='outer', ignore_index=False, keys=None,
#           levels=None, names=None, verify_integrity=False, copy=True)
#     df.to_csv(r'data/merged.csv', index = False)
    df=pd.read_csv("data/merged.csv", names=colnames,dtype={'Source IP':str,'Destination IP':str})
    df = df.iloc[1:]
    print(df) 
    if list(df.columns.values).count(target_name) != 1: 
        print("No target Label Found!")
        return
    df = df[[ 'Source IP',  'Source Port',  'Destination IP',  'Destination Port',  'Protocol'
             ,  'Flow Duration' ,'Average Packet Size', 'Label']]
#     df = shuffle(df)
    columnsToEncode = list(df.select_dtypes(include=['category', 'object']))  
    le = preprocessing.LabelEncoder()
    for feature in columnsToEncode:
        try:
            df[feature]=df[feature].astype(str)
            df[feature] = le.fit_transform(df[feature])
        except Exception as e:
            print ('error:'+ feature)
      
    target2idx = {target: idx for idx, target in enumerate(sorted(list(set(df[target_name].values))))}
    X = df.drop([target_name], axis=1).values
    
    y = np.vectorize(lambda x: target2idx[x])(df[target_name].values)
    n_classes = len(target2idx.keys())
    if X.shape[0] != y.shape[0]:
        raise Exception("X.shape = {} and y.shape = {} are inconsistent!".format(X.shape, y.shape))
    if normalize:
        X = (X - X.mean(axis=0)) / X.std(axis=0)
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