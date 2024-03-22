from time import perf_counter
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedKFold
from sklr.metrics import tau_x_score
from utils import mean_bucketSize, create_missing_labels
from sklearn.preprocessing import StandardScaler


def model_scores(model, train_X, train_Y, test_X, test_Y):
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)

    a = perf_counter()
    model.fit(train_X, train_Y)

  #  time = []
 #   for _ in range(20):
    Y_pred_model = model.predict(test_X)
    b = perf_counter()
#        time.append(b-a)

    #try:
    #    tau_x_score(Y_pred_model, test_Y)
    #except ValueError as ve:
    #    print("PROBLEM")

    print(f"TIME OF {model}: {b-a}")
    return (
    tau_x_score(Y_pred_model, test_Y),
    b-a,
    mean_bucketSize(Y_pred_model),
    )

def model_scores_Pipeline(model, train_X, train_Y, test_X, test_Y):
    # assume the model is a pipeline
    a = perf_counter()
    model.fit(train_X, train_Y)

  #  time = []
 #   for _ in range(20):
    Y_pred_model = model.predict(test_X)
    b = perf_counter()
#        time.append(b-a)

    try:
        tau_x_score(Y_pred_model, test_Y)
    except ValueError as ve:
        print("PROBLEM")

    print(f"TIME OF {model}: {b-a}")
    return (
    tau_x_score(Y_pred_model, test_Y),
    b-a,
    mean_bucketSize(Y_pred_model),
    )

def model_evaluation(models, X, Y, random_state,
                     model_score_function):
    # The Output of clas_model is assumed to be in the correct api format
    n_folds = 10
    n_repeats = 5
    folder = RepeatedKFold(n_splits=n_folds,n_repeats=n_repeats,random_state=random_state)

    res = np.zeros(shape=(len(models),n_folds * n_repeats, 3))
    split = 0
    for train_idx,test_idx in folder.split(X,Y):
        X_train,X_test,Y_train,Y_test = X[train_idx],X[test_idx],Y[train_idx],Y[test_idx]
        for i in range(len(models)):
            res[i,split] = model_score_function(model=models[i],
                                          train_X=X_train,
                                          train_Y=Y_train,
                                          test_X=X_test,
                                          test_Y=Y_test)

        split += 1
    # res has its first dimension distinguishing between the different models
    # In its second dimension it distinguished between the diffrent fols
    # The third dimension holds: accuracy, time, mean_buckets_per_rank in this order
    return res

def model_evaluation_missingLabels(models, X, Y, random_state,
                     model_score_function, percentage_missing_labels):
    # The Output of clas_model is assumed to be in the correct api format
    n_folds = 10
    n_repeats = 5
    folder = RepeatedKFold(n_splits=n_folds,n_repeats=n_repeats,random_state=random_state)

    res = np.zeros(shape=(len(models),n_folds * n_repeats, 3))
    split = 0
    for train_idx,test_idx in folder.split(X,Y):
        X_train,X_test,Y_train,Y_test = X[train_idx],X[test_idx],Y[train_idx],Y[test_idx]
        Y_train = create_missing_labels(Y_train,percentage=percentage_missing_labels, random_state=random_state)
        for i in range(len(models)):
            res[i,split] = model_score_function(model=models[i],
                                          train_X=X_train,
                                          train_Y=Y_train,
                                          test_X=X_test,
                                          test_Y=Y_test)

        split += 1
    # res has its first dimension distinguishing between the different models
    # In its second dimension it distinguished between the diffrent fols
    # The third dimension holds: accuracy, time, mean_buckets_per_rank in this order
    return res