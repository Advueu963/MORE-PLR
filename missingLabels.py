from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklr.pairwise import PairwisePartialLabelRanker
from sklr.metrics import tau_x_score

from MORE_models import PLR_MultiOutputRegressor, PLR_RegressorChain, PLR_RegressorChainInterval, \
    PLR_RandomForestRegressor

random_state = 0
from utils import create_apiRanks

def optimism_change_Y(X,Y):
    return X , create_apiRanks(Y) # Negative Values are ranked higher than positive values. Therefore we can just call this function

def pessimism_change_Y(X,Y):
    return X, create_apiRanks(
        np.where(Y < 0 , Y.shape[1]+1,Y) # change the -1 to the biggest values which hurt the api form. The ranks are then readjusted such that exactly these values resemble lowest rank (highest value)
    )

def optimism_data_approach(X, Y):
    # Assume that potentially Y contains -1 == Nan missing labels
    missing_ranks = np.any(Y <0, axis=1) # save the indices we change
    non_change_X,non_change_Y = X[~missing_ranks],Y[~missing_ranks]
    change_X,change_Y = optimism_change_Y(X[missing_ranks], Y[missing_ranks])

    # create the new arrays and make sure the row-order is the same as original array.
    X_opt = np.zeros(X.shape)
    Y_opt = np.zeros(Y.shape)

    X_opt[missing_ranks] = change_X
    X_opt[~missing_ranks] = non_change_X

    Y_opt[missing_ranks] = change_Y
    Y_opt[~missing_ranks] = non_change_Y

    return X_opt, Y_opt
def pessimism_data_approach(X, Y):
    # Assume that potentially Y contains -1 == Nan missing labels
    # Assume that potentially Y contains -1 == Nan missing labels
    missing_ranks = np.any(Y < 0, axis=1)  # save the indices we change
    non_change_X, non_change_Y = X[~missing_ranks], Y[~missing_ranks]
    change_X, change_Y = pessimism_change_Y(X[missing_ranks], Y[missing_ranks])

    # create the new arrays and make sure the row-order is the same as original array.
    X_opt = np.zeros(X.shape)
    Y_opt = np.zeros(Y.shape)

    X_opt[missing_ranks] = change_X
    X_opt[~missing_ranks] = non_change_X

    Y_opt[missing_ranks] = change_Y
    Y_opt[~missing_ranks] = non_change_Y

    return X_opt, Y_opt

def optimism_pessimism_data_approach(X, Y):
    # Assume that potentially Y contains -1 == Nan missing labels
    missing_ranks = np.any(Y < 0, axis=1)  # save the indices we change
    non_change_X, non_change_Y = X[~missing_ranks], Y[~missing_ranks]
    _,Y_opt = optimism_change_Y(X[missing_ranks], Y[missing_ranks])
    _,Y_pes = optimism_change_Y(X[missing_ranks], Y[missing_ranks])


    Y_opt_pes = np.zeros(
        shape=(
            Y.shape[0] + Y_opt.shape[0],
            Y.shape[1]
        )
    )
    # Transform missing_ranks so that:
    #1.this mask has doubled the True and leaved the False unchanged
    #2. Therefore we have both possibilities directly after each other and the unchanged are at the "same" position (not really but relatively same position) .
    repeats = missing_ranks + 1
    missing_ranks = np.repeat(  # this mask has doubled the True and leaved the False unchanged
        missing_ranks,
        repeats=repeats,
        axis = 0,
    )
    X_opt_pes = np.repeat(
        X,
        repeats=repeats,
        axis=0
    )

    Y_opt_pes[~missing_ranks] = non_change_Y
    modulo_uneven = np.array(range(missing_ranks.size)) % 2 == 1 # must be boolean array otherwise the broadcast wont work
    Y_opt_pes[(missing_ranks & modulo_uneven)] = Y_opt
    Y_opt_pes[(missing_ranks & ~modulo_uneven)] = Y_pes



    return X_opt_pes, Y_opt_pes


def create_missing_labels(Y, percentage, random_state=0):
    # source: https://stackoverflow.com/questions/31389481/numpy-replace-random-elements-in-an-array
    random_generator = np.random.default_rng(random_state) # reproducibility
    shape = Y.shape       # Store original shape
    temp = Y.flatten()  # Flatten to 1D
    inds = random_generator.choice(temp.size, size=int(np.ceil(temp.size*percentage)), replace=False)  # Get random indices
    temp[inds] = -1
    temp = temp.reshape(shape) # brings back the original Y. Does it holds: Y == Y.flatten().reshape(Y.shape)
    return temp

def create_missing_labels2(Y, percentage, random_state=0):
    # source: https://stackoverflow.com/questions/31389481/numpy-replace-random-elements-in-an-array
    random_generator = np.random.default_rng(random_state) # reproducibility
    n_rows,n_columns = Y.shape
    for i in range(n_rows):
        for j in range(n_columns):
            if random_generator.binomial(1,percentage):
                Y[i,j] = -1
    return Y

clas_estimator = SVC(random_state=random_state,probability=True)
clas_model = PairwisePartialLabelRanker(clas_estimator)

mort = PLR_RandomForestRegressor(random_state=random_state, missing_label_strategy="drop_individuals")

regr_estimator = DecisionTreeRegressor(random_state=random_state)
singleTarget_Model = PLR_MultiOutputRegressor(regr_estimator, missing_label_strategy="drop_individuals")


regr_estimator = RandomForestRegressor(random_state=random_state)

chain_model = PLR_RegressorChainInterval(regr_estimator,
                                 order=None,
                                 random_state=random_state,
                                 missing_label_strategy="drop_individuals")
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

X, Y = fetch_openml(data_id=45735, as_frame=as_frame, return_X_y=return_X_y, parser='auto')
X = np.ascontiguousarray(X)
ergs = []
Y = Y.astype(np.float64)

folder = KFold(n_splits=5,shuffle=True,random_state=random_state)

for train_idx,test_idx in folder.split(X,Y):
    X_train,X_test,Y_train,Y_test = X[train_idx],X[test_idx],Y[train_idx],Y[test_idx]
    missing_Y_train = create_missing_labels(Y_train, percentage=0.5, random_state=random_state)

    clas_model.fit(X_train, missing_Y_train)

    clas_Y_pred = clas_model.predict(X_test)

    print("JC: ", tau_x_score(Y_test, clas_Y_pred))







"""
print("MISSING:", missing_Y_train[:4], missing_Y_train.shape)
X_opt,Y_opt = optimism_data_approach(X_train, missing_Y_train)
print("OPTIMISM:",Y_opt[:4], Y_opt.shape)
X_pes, Y_pes = pessimism_data_approach(X_train, missing_Y_train)
print("PESSIMISM:",Y_pes[:4], Y_pes.shape)
X_opt_pes,Y_opt_pes = optimism_pessimism_data_approach(X_train, missing_Y_train)
print("BOTH:",Y_opt_pes[:4], Y_opt_pes.shape)
"""
