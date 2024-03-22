from time import perf_counter
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import RepeatedKFold
from sklr.pairwise import PairwisePartialLabelRanker
from sklearn.svm import SVR, SVC
from utils import build_plottable_evaluationDataFrame, plot_evaluation_data
from utils_chain import model_evaluation, model_scores
from MORE_models import PLR_RegressorChain, PLR_MultiOutputRegressor, PLR_RegressorChainInterval
import os
import sys

from sklr.metrics import tau_x_score
from utils import mean_bucketSize, create_missing_labels
name_to_data_lr = {
        "LR-LETTER": 45727,
}

def add_to_data(data, data_name,score,time,bucket_per_rank,algo, percentage):
        if(type(bucket_per_rank) in [np.ndarray,list]): # just to make sure it is a list or np.Array
            assert len(score) == len(bucket_per_rank) == len(time)
            for i in range(len(score)):
                # add the scores
                data['data'].append(data_name)
                data['tau_x_score'].append(score[i])
                data['prediction_time'].append(np.sqrt(time[i]))
                data['buckets_per_rank'].append(bucket_per_rank[i])
                data['algo'].append(algo)
                data['percentage'].append(percentage)
        else:
            data['data'].append(data_name)
            data['tau_x_score'].append(score)
            data['prediction_time'].append(np.sqrt(time))
            data['buckets_per_rank'].append(bucket_per_rank)
            data['algo'].append(algo)
            data['percentage'].append(percentage)

if __name__ == '__main__':
    
    data = {'data': [], 'tau_x_score': [], 'prediction_time': [], 'buckets_per_rank': [],
            'algo': [], 'percentage':[]}
    
    
    random_state = 0
    # Options for JC data extraction
    as_frame = False
    return_X_y = True
    
    # The Output of clas_model is assumed to be in the correct api format
    n_folds = 10
    n_repeats = 5
    folder = RepeatedKFold(n_splits=n_folds,n_repeats=n_repeats,random_state=random_state)

    n_jobs = int(os.environ['SLURM_CPUS_PER_TASK'])  # important that this equals the number of cpus on the linux cluster batch job
    number = 20
    percentage = float(sys.argv[1])
    
    
    
    clas_estimator = SVC(probability=True, random_state=random_state, cache_size=4000)
    clas_model = PairwisePartialLabelRanker(clas_estimator, n_jobs=n_jobs)

    regr_estimator = SVR(cache_size=4000)
    regr_model_chain = PLR_RegressorChain(estimator=regr_estimator,
                                          order=None,  # the order will later on be updated to an appropriate order
                                          random_state=random_state)  # not really necessary but double is better

    regr_estimator = SVR(cache_size=4000)
    regr_model_singleTarget = PLR_MultiOutputRegressor(estimator=regr_estimator,
                                                       n_jobs=n_jobs)

    regr_name_chain = "CHAIN-SVR"
    regr_name_singleTarget = "SingleTarget-SVR"
    clas_name = "SVC-JC"
    
    model = clas_model
    model_name = clas_name
    
    # LR Letter
    X, Y = fetch_openml(data_id=45727,
                        as_frame=as_frame,
                        return_X_y=return_X_y,
                        parser='auto')
    X = np.ascontiguousarray(X) # need for performance improvement at SVM
    Y = Y.astype(np.float64)
    
    all_splits = [
        (train_idx, test_idx)
        for train_idx, test_idx in folder.split(X,Y)
        
        
    ]
    train_idx,test_idx = all_splits[number]
    
    scaler = StandardScaler()
    
    
    train_X,test_X,train_Y,test_Y = X[train_idx],X[test_idx],Y[train_idx],Y[test_idx]
    Y_train = create_missing_labels(train_Y,percentage=percentage, random_state=random_state)

    
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
    add_to_data(data,
                    "LR-LETTER",
                    tau_x_score(Y_pred_model, test_Y),
                    b-a,
                    mean_bucketSize(Y_pred_model),
                    model_name,
                    percentage
                    )
    pd.DataFrame(data).to_csv(
        f"../data/MultiRegression/LR-SVM/LR-Letter-Missing/LR_{model_name}_{number}_{percentage}.csv"
    )

