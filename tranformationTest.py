"""
File to test different "Post Hoc"-Layers
"""


from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import numpy as np
from sklr.pairwise import PairwisePartialLabelRanker
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklr.metrics import tau_x_score

random_state = 0
from utils import create_apiRanks
from _overlap_intervals import test

def round_transformation(array):
    erg = np.round(array)
    erg2 = create_apiRanks(erg)
    return erg2
"""
def create_apiRanks(array):
    for i in range(array.shape[0]):
        row = array[i, :]
        for ranks in range(0, len(row)):
            try:
                next_lowest = np.min(row[row > ranks - 1])
            except ValueError as ve:
                break  # we have built the rank according to the api
            row = np.where(row == next_lowest, ranks, np.where(row <= ranks - 1, row, row + 1))
        array[i, :] = row + 1
    return array"""

def round_normalize_transformation(array):
    ranks = np.argsort(array, axis=-1) + 1
    normalized_array = (array - np.min(array, axis=-1)[:,None]) / (np.max(array,axis=-1)[:,None] - np.min(array,axis=-1)[:,None])
    erg = np.round(normalized_array * ranks)
    erg2 = create_apiRanks(erg)
    return erg2

def round_normalize_epsilonClose_tranformation(array,epsilon=0.1):
    n_samples,n_classes = array.shape
    low_array = array - epsilon
    high_array = array + epsilon
    interval_array = np.zeros((n_samples,n_classes,2))
    interval_array[:,:,0] = low_array
    interval_array[:,:,1] = high_array
    consensus = np.zeros(shape=(n_samples, n_classes))  # N_samples, N_outputs
    for i in range(consensus.shape[0]):
        test(interval_array[i], n_classes, consensus[i])
    erg2 = create_apiRanks(consensus)
    return erg2

clas_estimator = RandomForestClassifier(random_state=random_state, n_jobs=-1)
clas_model = PairwisePartialLabelRanker(clas_estimator)

regr_model = RandomForestRegressor(random_state=random_state, n_jobs=-1)

regr_name = "RF-Naive"
clas_name = "RF-JC"

# Options for JC data extraction
as_frame = False
return_X_y = True
# data
"""
    'Authorship': 42834,
    "BLOCKS":42836,
    "BREAST":42838,
    'ECOLI': 42844,
    "GLASS":42847,
    'IRIS': 42851,
    "LETTER":42853,
    'LIBRAS': 42855,
    "PENDIGITS":42856,
    "SATIMAGE":42858,
    "SEGMENT":42859,
    "VEHICLE":42863,
    "VOWEL":42865,
    'WINE': 42872,
"""

X, Y = fetch_openml(data_id=42872, as_frame=as_frame, return_X_y=return_X_y, parser='auto')
X = np.ascontiguousarray(X)
ergs = []
Y = Y.astype(np.float64)
n_folds = 5
folder = KFold(n_splits=n_folds,shuffle=True,random_state=random_state)

for train_idx,test_idx in folder.split(X,Y):
    train_X,test_X,train_Y,test_Y = X[train_idx],X[test_idx],Y[train_idx],Y[test_idx]

    scaler = StandardScaler()

    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)




    # models
    regr_model.fit(train_X,train_Y)


    regr_model.fit(train_X,train_Y)
    regr_output_raw = regr_model.predict(test_X)
    regr_output_transformation1 = round_transformation(regr_output_raw)
    regr_output_transformation2 = round_normalize_transformation(regr_output_raw)
    regr_output_transformation3 = round_normalize_epsilonClose_tranformation(regr_output_raw, epsilon=0.1)

    clas_model.fit(train_X,train_Y)
    clas_output = clas_model.predict(test_X)

    print("Regressor Transformation 1:", tau_x_score(regr_output_transformation1, test_Y))
    print("Regressor Transformation 2: ", tau_x_score(regr_output_transformation2, test_Y))
    print("Regressor Transformation 3: ", tau_x_score(regr_output_transformation3, test_Y))
    print("Classifier: ", tau_x_score(clas_output, test_Y))
    print()

    ergs += [tau_x_score(regr_output_transformation1, test_Y)]
    ergs += [tau_x_score(regr_output_transformation2, test_Y)]
    ergs += [tau_x_score(regr_output_transformation3, test_Y)]
    ergs += [tau_x_score(clas_output, test_Y)]

ergs = np.array(ergs).reshape(5,-1)
print(np.mean(ergs,axis=0))
