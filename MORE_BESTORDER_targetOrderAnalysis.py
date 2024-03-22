import itertools

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.model_selection import KFold
from sklr.pairwise import PairwisePartialLabelRanker
from sklearn.multioutput import RegressorChain
from sklr.metrics import tau_x_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.neighbors import KNeighborsRegressor
import seaborn as sns
import matplotlib.pyplot as plt

kernel = DotProduct() + WhiteKernel()

import utils
data_id = 42851
# Authorship: 42834
# IRIS: 42851
# WINE: 42872
# ECOLI: 42844
# Stock : 42862
# LIBRAS: 42855

name_to_data = {
     'IRIS': 42834,
     'WINE': 42872,
    'Stock' : 42862,
    'Authorship': 42834,
    'ECOLI': 42844,
    'LIBRAS': 42855
}

as_frame = False
return_X_y = True
X, Y = fetch_openml(data_id=data_id, as_frame=as_frame, return_X_y=return_X_y)
Y = Y.astype(float)
random_state = 0
samples,n_classes = Y.shape


estimator = RandomForestClassifier(random_state=random_state)
old_model = PairwisePartialLabelRanker(estimator)

estimator_regr = KNeighborsRegressor(n_neighbors=5)
n_splits = 5


def findBestOrder(X, Y):
    Z = np.append(X, Y, axis=1)
    correlationMatrix = pd.DataFrame(Z, columns=[f"X{x}" for x in range(X.shape[1])] + [f"Y{x}" for x in
                                                                                        range(Y.shape[1])])\
                        .corr(method="kendall")
    sns.heatmap(correlationMatrix, annot=True)
    plt.show()

    _, n_features = X.shape
    _, n_classes = Y.shape
    erg = []

    relevant_indices = [(x, x + n_features) for x in
                        range(n_classes)]  # the indices of all the Y columns in correlationMatrix
    feature_indices = [x for x in range(n_features)]
    for i in range(n_classes):
        # restrict the correlation matrix to the relevant values
        relevant_area = np.power(correlationMatrix.values[feature_indices, :][:, [y for _, y in relevant_indices]],2)
        # Sum up the correlation values to determine target with best correlation
        correlation_points = np.sum(relevant_area, axis=0)
        # get target with highest correlation
        target_with_max_corr = np.argmax(correlation_points)
        # append original target index to resulting order
        erg.append(relevant_indices[target_with_max_corr][0])
        # expand the feature space with the new found best target index
        # Simulates the idea of Regressor Chain
        feature_indices.append(relevant_indices[target_with_max_corr][1])
        # remove the new target index form the relevant indices
        relevant_indices.remove(relevant_indices[target_with_max_corr])
    print(erg)
    return erg

def evaluation():
    for order in itertools.permutations(range(n_classes)):
        test_score_regr = 0
        test_score_clas = 0
        new_model = RegressorChain(estimator_regr, order=order)
        folder = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        for train_idx,test_idx in folder.split(X,Y):
            X_train,X_test,Y_train,Y_test = X[train_idx],X[test_idx],Y[train_idx],Y[test_idx]

            old_model.fit(X_train,Y_train)
            new_model.fit(X_train,Y_train)


            y_pred_old = old_model.predict(X_test)
            y_pred_new = new_model.predict(X_test)

            y_pred_new = utils.transform_target_toAPIForm(pd.DataFrame(np.round(y_pred_new)))

            test_score_clas += tau_x_score(Y_test,y_pred_old)
            test_score_regr += tau_x_score(Y_test,y_pred_new)

        test_score_clas /= n_splits
        test_score_regr /= n_splits
        print("TEST SCORE OLD: ", test_score_clas)
        print("TEST SCORE NEW: ", test_score_regr)
        print("ORDER: ", list(order))

def evaluation2():
    test_score_regr = 0
    test_score_clas = 0
    folder = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_idx, test_idx in folder.split(X, Y):
        X_train, X_test, Y_train, Y_test = X[train_idx], X[test_idx], Y[train_idx], Y[test_idx]

        order = findBestOrder(X_train,Y_train)
        new_model = RegressorChain(estimator_regr, order=order)
        old_model.fit(X_train, Y_train)
        new_model.fit(X_train, Y_train)

        y_pred_old = old_model.predict(X_test)
        y_pred_new = new_model.predict(X_test)

        y_pred_new = utils.transform_target_toAPIForm(pd.DataFrame(np.round(y_pred_new)))

        test_score_clas += tau_x_score(Y_test, y_pred_old)
        test_score_regr += tau_x_score(Y_test, y_pred_new)

    test_score_clas /= n_splits
    test_score_regr /= n_splits
    print("TEST SCORE OLD: ", test_score_clas)
    print("TEST SCORE NEW: ", test_score_regr)
    print("ORDER: ", list(order))

#evaluation()
evaluation2()
findBestOrder(X,Y)

