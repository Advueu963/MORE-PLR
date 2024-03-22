from time import perf_counter
import numpy as np
import pandas as pd
import scipy.stats
import sklearn.gaussian_process
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import KFold
from sklearn.multioutput import RegressorChain
from sklr.metrics import tau_x_score
import utils
import seaborn as sns
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from mutlOutRegr._overlap_intervals import test
from mapie.regression import MapieRegressor
from mutlOutRegr.RegressorChainInterval import PLR_RegressorChainInterval

def findBestOrder(X, Y):
    Z = np.append(X, Y, axis=1)
    correlationMatrix = pd.DataFrame(Z, columns=[f"X{x}" for x in range(X.shape[1])] + [f"Y{x}" for x in
                                                                                        range(Y.shape[1])])\
                        .corr(method="kendall")
    #sns.heatmap(correlationMatrix, annot=True)
    #plt.show()

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


def get_Buckets_Sizes_Counts(Y):
    unique_rankings,counts = np.unique(Y.astype(int),axis=0,return_index=True)
    bucket_sizes_of_unique = np.array(list(map(lambda x: len(set(x)),np.unique(unique_rankings,axis=0))))
    return unique_rankings,bucket_sizes_of_unique,counts

def mean_bucketSize(Y):
    return np.mean(get_Buckets_Sizes_Counts(Y)[1])

def model_predict_round(model,test_X):
    regr_output = model.predict(test_X)
    #print(regr_output)
    return utils.transform_arrayToAPI(np.round(regr_output))

def model_predict_intervalGaussian(model: sklearn.gaussian_process.GaussianProcessRegressor, test_X):
    regr_output,regr_std = model.predict(test_X,return_std=True)
    alpha = 0.05
    quantile = scipy.stats.norm.ppf(1- alpha/2)
    print("QUANTILE : ", quantile)
    top = regr_output + quantile * regr_std
    bottom = regr_output - quantile * regr_std
    intervals = np.stack((bottom,top),axis=2)
    consensus = np.empty(shape=(intervals.shape[0],intervals.shape[1])) # N_samples, N_outputs
    #for i in range(consensus.shape[0]):
    #    test(intervals[i],intervals.shape[1],consensus[i])
    return utils.transform_arrayToAPI(consensus)


def model_predict_intervalChain(model: PLR_RegressorChainInterval, test_X):
    pred_intervals = model.predict(test_X)

    consensus = np.empty(shape=(pred_intervals.shape[0],pred_intervals.shape[1])) # N_samples, N_outputs
    Parallel(n_jobs=-1)(
        delayed(test)(pred_intervals[i],pred_intervals.shape[1],consensus[i])
        for i in range(consensus.shape[0])
    )
    return utils.transform_arrayToAPI(consensus)

def model_predict_intervalBagging(model: sklearn.gaussian_process.GaussianProcessRegressor, test_X):
    individual_bagging_predicts = np.array([m.predict(test_X) for m in model.estimators_])
    mean_prediction = np.mean(individual_bagging_predicts, axis=0)
    std_prediction = np.std(individual_bagging_predicts, axis=0)
    factor = 1.96
    top = mean_prediction + factor * std_prediction / np.sqrt(5)
    bottom = mean_prediction - factor * std_prediction / np.sqrt(5)
    intervals = np.stack((bottom,top),axis=2)
    consensus = np.empty(shape=(intervals.shape[0],intervals.shape[1])) # N_samples, N_outputs
    for i in range(consensus.shape[0]):
        test(intervals[i],intervals.shape[1],consensus[i])
    return utils.transform_arrayToAPI(consensus)

def model_predict_conformal(conformel_model, test_X):
    alpha = 0.05
    quantile = scipy.stats.norm.ppf(1 - alpha / 2)
    y_pred,y_predInterval = conformel_model.predict(test_X,alpha=alpha)
    top = y_pred  + quantile * y_predInterval[:,0]
    bottom = y_pred - quantile * y_predInterval[:,1]
    intervals = np.stack((bottom, top), axis=2)


def model_scores_Chain(regr_model, clas_model, train_X, train_Y, test_X, test_Y):
    model1_fitted = regr_model.fit(train_X, train_Y)
    model2_fitted = clas_model.fit(train_X, train_Y)

    a = perf_counter()
    Y_pred_regr = model_predict_round(model1_fitted, test_X)
    b = perf_counter()
    Y_pred_clas = model2_fitted.predict(test_X)
    c = perf_counter()

    return (
    tau_x_score(Y_pred_regr, test_Y),
    tau_x_score(Y_pred_clas, test_Y),
    b - a,
    c - b,
    mean_bucketSize(Y_pred_regr),
    mean_bucketSize(Y_pred_clas),
    mean_bucketSize(test_Y)
            )

def model_scores_ChainInterval(regr_model, clas_model, train_X, train_Y, test_X, test_Y):
    model1_fitted = regr_model.fit(train_X, train_Y)
    model2_fitted = clas_model.fit(train_X, train_Y)

    a = perf_counter()
    Y_pred_regr = model_predict_intervalChain(model1_fitted, test_X)
    b = perf_counter()
    Y_pred_clas = model2_fitted.predict(test_X)
    c = perf_counter()

    return (
    tau_x_score(Y_pred_regr, test_Y),
    tau_x_score(Y_pred_clas, test_Y),
    b - a,
    c - b,
    mean_bucketSize(Y_pred_regr),
    mean_bucketSize(Y_pred_clas),
    mean_bucketSize(test_Y)
            )

def model_scores_ChainConformel(regr_model, clas_model, train_X, train_Y, test_X, test_Y):
    model1_fitted = regr_model.fit(train_X, train_Y)
    model2_fitted = clas_model.fit(train_X, train_Y)

    a = perf_counter()
    Y_pred_regr = model_predict_conformal(model1_fitted, test_X)
    b = perf_counter()
    Y_pred_clas = model2_fitted.predict(test_X)
    c = perf_counter()

    return (
    tau_x_score(Y_pred_regr, test_Y),
    tau_x_score(Y_pred_clas, test_Y),
    b - a,
    c - b,
    mean_bucketSize(Y_pred_regr),
    mean_bucketSize(Y_pred_clas),
    mean_bucketSize(test_Y)
            )


def model_scores_GaussianInterval(regr_model, clas_model, train_X, train_Y,test_X,test_Y):
    model1_fitted = regr_model.fit(train_X, train_Y)
    model2_fitted = clas_model.fit(train_X, train_Y)

    a = perf_counter()
    Y_pred_regr = model_predict_intervalGaussian(model1_fitted, test_X)
    b = perf_counter()
    Y_pred_clas = model2_fitted.predict(test_X)
    c = perf_counter()

    return (
    tau_x_score(Y_pred_regr, test_Y),
    tau_x_score(Y_pred_clas, test_Y),
    b - a,
    c - b,
    mean_bucketSize(Y_pred_regr),
    mean_bucketSize(Y_pred_clas),
    mean_bucketSize(test_Y)
            )

def model_scores_GaussianBagging(regr_model, clas_model, train_X, train_Y,test_X,test_Y):
    model1_fitted = regr_model.fit(train_X, train_Y)
    model2_fitted = clas_model.fit(train_X, train_Y)

    a = perf_counter()
    Y_pred_regr = model_predict_intervalBagging(regr_model,test_X)
    b = perf_counter()
    Y_pred_clas = model2_fitted.predict(test_X)
    c = perf_counter()

    return (
    tau_x_score(Y_pred_regr, test_Y),
    tau_x_score(Y_pred_clas, test_Y),
    b - a,
    c - b,
    mean_bucketSize(Y_pred_regr),
    mean_bucketSize(Y_pred_clas),
    mean_bucketSize(test_Y)
            )

def model_evaluation_ChainInterval(regr_estimator, clas_model, X, Y, random_state):
    # The Output of clas_model is assumed to be in the correct api format
    n_folds = 5
    folder = KFold(n_splits=n_folds,shuffle=True,random_state=random_state)

    res = []
    for train_idx,test_idx in folder.split(X,Y):
        res.append(model_scores_ChainInterval(regr_model=PLR_RegressorChainInterval(regr_estimator, order=findBestOrder(X[train_idx], Y[train_idx])),
                                              clas_model=clas_model,
                                              train_X=X[train_idx],
                                              train_Y=Y[train_idx],
                                              test_X=X[test_idx],
                                              test_Y=Y[test_idx])
                   )
    res = np.array(res).reshape(n_folds, -1)
    test_score_regr, test_score_clas, time_regr, time_clas, bucket_per_rank_regr, bucket_per_rank_clas, true_buckets_per_rank = res.sum(
        axis=0) / n_folds
    return (test_score_regr, time_regr, bucket_per_rank_regr), (test_score_clas, time_clas, bucket_per_rank_clas), (
    1, 0, true_buckets_per_rank)

def model_evaluation_Chain(regr_estimator, clas_model, X, Y, random_state):
    # The Output of clas_model is assumed to be in the correct api format
    n_folds = 5
    folder = KFold(n_splits=n_folds,shuffle=True,random_state=random_state)

    res = []
    for train_idx,test_idx in folder.split(X,Y):
        res.append(model_scores_Chain(regr_model=RegressorChain(regr_estimator, order=findBestOrder(X[train_idx], Y[train_idx])),
                                      clas_model=clas_model,
                                      train_X=X[train_idx],
                                      train_Y=Y[train_idx],
                                      test_X=X[test_idx],
                                      test_Y=Y[test_idx])
                   )



    res = np.array(res).reshape(n_folds,-1)
    test_score_regr,test_score_clas,time_regr,time_clas,bucket_per_rank_regr,bucket_per_rank_clas,true_buckets_per_rank = res.sum(axis=0) / n_folds
    return (test_score_regr,time_regr,bucket_per_rank_regr),(test_score_clas,time_clas,bucket_per_rank_clas) , (1,0,true_buckets_per_rank)

def model_evaluation_ChainConformel(regr_estimator, clas_model, X, Y, random_state):
    # The Output of clas_model is assumed to be in the correct api format
    n_folds = 5
    folder = KFold(n_splits=n_folds,shuffle=True,random_state=random_state)

    res = []
    for train_idx,test_idx in folder.split(X,Y):
        res.append(model_scores_ChainConformel(regr_model=MapieRegressor(RegressorChain(regr_estimator, order=findBestOrder(X[train_idx], Y[train_idx]))),
                                      clas_model=clas_model,
                                      train_X=X[train_idx],
                                      train_Y=Y[train_idx],
                                      test_X=X[test_idx],
                                      test_Y=Y[test_idx])
                   )

    res = np.array(res).reshape(n_folds,-1)
    test_score_regr,test_score_clas,time_regr,time_clas,bucket_per_rank_regr,bucket_per_rank_clas,true_buckets_per_rank = res.sum(axis=0) / n_folds
    return (test_score_regr,time_regr,bucket_per_rank_regr),(test_score_clas,time_clas,bucket_per_rank_clas) , (1,0,true_buckets_per_rank)


def model_evaluation_Gaussian(regr_model, clas_model, X, Y, random_state):
    # The Output of clas_model is assumed to be in the correct api format
    n_folds = 5
    folder = KFold(n_splits=n_folds,shuffle=True,random_state=random_state)
    
    res = []
    for train_idx,test_idx in folder.split(X,Y):
        res.append(model_scores_Chain(regr_model, clas_model, X[train_idx], Y[train_idx],
                                      X[test_idx], Y[test_idx]))
        print("Split Done")
    res = np.array(res).reshape(n_folds, -1)


    test_score_regr, test_score_clas, time_regr, time_clas, bucket_per_rank_regr, bucket_per_rank_clas, true_buckets_per_rank = np.mean(res,axis=0)

    return (test_score_regr,time_regr,bucket_per_rank_regr),(test_score_clas,time_clas,bucket_per_rank_clas) , (1,0,true_buckets_per_rank)


def model_evaluation_GaussianCompleteOutput(regr_model, clas_model, X, Y, random_state):
    # Used to later draw confidence intervals of the estimate scores
    # The Output of clas_model is assumed to be in the correct api format
    n_folds = 5
    folder = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    res = []
    for train_idx, test_idx in folder.split(X, Y):
        res.append(model_scores_Chain(regr_model, clas_model, X[train_idx], Y[train_idx],
                                      X[test_idx], Y[test_idx]))
        print("Split Done")
    res = np.array(res).reshape(n_folds, -1)

    test_score_regr, test_score_clas, time_regr, time_clas, bucket_per_rank_regr, bucket_per_rank_clas, true_buckets_per_rank = res.T

    return (test_score_regr, time_regr, bucket_per_rank_regr), (test_score_clas, time_clas, bucket_per_rank_clas), (
    1, 0, true_buckets_per_rank)

def model_evaluation_GaussianBagging(regr_model, clas_model, X, Y, random_state):
    # The Output of clas_model is assumed to be in the correct api format
    n_folds = 5
    folder = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    res = []
    for train_idx, test_idx in folder.split(X, Y):
        res.append(model_scores_GaussianBagging(regr_model, clas_model, X[train_idx], Y[train_idx],
                                X[test_idx], Y[test_idx]))
        print("Split Done")
    res = np.array(res).reshape(n_folds, -1)

    test_score_regr, test_score_clas, time_regr, time_clas, bucket_per_rank_regr, bucket_per_rank_clas, true_buckets_per_rank = res.T

    return (test_score_regr, time_regr, bucket_per_rank_regr), (test_score_clas, time_clas, bucket_per_rank_clas), (
    1, 0, true_buckets_per_rank)


def model_evaluation_GaussianInterval(regr_model, clas_model, X, Y, random_state):
    # The Output of clas_model is assumed to be in the correct api format
    n_folds = 5
    folder = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    res = []
    for train_idx, test_idx in folder.split(X, Y):
        res.append(model_scores_GaussianInterval(regr_model, clas_model, X[train_idx], Y[train_idx],
                                X[test_idx], Y[test_idx]))
        print("Split Done")
    res = np.array(res).reshape(n_folds, -1)

    test_score_regr, test_score_clas, time_regr, time_clas, bucket_per_rank_regr, bucket_per_rank_clas, true_buckets_per_rank = res.sum(
        axis=0) / n_folds

    return (test_score_regr, time_regr, bucket_per_rank_regr), (test_score_clas, time_clas, bucket_per_rank_clas), (
    1, 0, true_buckets_per_rank)


def build_plottable_evaluationDataFrame(name_to_data, regr_estimator, clas_model, random_state,
                                        model_evaluation,
                                        regr_model_name="", clas_model_name=""):
    names,data_ids = list(name_to_data.keys()),list(name_to_data.values())
    # data frame later used for prediction
    data = {'data': [], 'tau_x_score': [], 'prediction_time': [], 'buckets_per_rank': [],
            'algo': []}

    def add_to_data(data_name,score,time,bucket_per_rank,algo):
        if(type(bucket_per_rank) in [np.ndarray,list]): # just to make sure it is a list or np.Array
            # Check if we are algo==DATA or a normal Model
            if(type(score) in [np.ndarray, list]):
                assert len(score) == len(bucket_per_rank) == len(time)
                for i in range(len(score)):
                    # add the scores
                    data['data'].append(data_name)
                    data['tau_x_score'].append(score[i])
                    data['prediction_time'].append(np.sqrt(time[i]))
                    data['buckets_per_rank'].append(bucket_per_rank[i])
                    data['algo'].append(algo)
            else:
                #Special Case were we have algo == DATA
                for i in range(len(bucket_per_rank)):
                    # add the scores
                    data['data'].append(data_name)
                    data['tau_x_score'].append(score)
                    data['prediction_time'].append(np.sqrt(time))
                    data['buckets_per_rank'].append(bucket_per_rank[i])
                    data['algo'].append(algo)
        else:
            data['data'].append(data_name)
            data['tau_x_score'].append(score)
            data['prediction_time'].append(np.sqrt(time))
            data['buckets_per_rank'].append(bucket_per_rank)
            data['algo'].append(algo)

    # Options for JC data extraction
    as_frame = False
    return_X_y = True

    for i in range(len(data_ids)):
        X, Y = fetch_openml(data_id=data_ids[i], as_frame=as_frame, return_X_y=return_X_y,parser='auto')
        X = StandardScaler().fit_transform(X)

        Y = Y.astype(np.float64)
        samples, n_classes = Y.shape


        regr_results,clas_results,best_results = model_evaluation(regr_estimator,clas_model,X,Y,random_state)

        # Add regression Outputs
        add_to_data(names[i],*regr_results,algo=regr_model_name)

        # Add Classification Outputs
        add_to_data(names[i], *clas_results, algo=clas_model_name)

        # Add "Best"/Data Results
        #add_to_data(names[i], *best_results, algo="Data")


        print(f"FINISHED DATA_ID: {data_ids[i]}")
    return pd.DataFrame(data)

def plot_errorbar(array_x,normalizer=5):
    mean_x = np.mean(array_x)
    std_x = np.std(array_x)
    return(mean_x - (std_x/np.sqrt(normalizer)*1.96) ,
           mean_x + (std_x/np.sqrt(normalizer)*1.96))


def plot_evaluation_data(dataframe: pd.DataFrame,regr_name,clas_name, xticks, fileName=None):
    df_long = pd.melt(dataframe,id_vars=['data','algo'],value_name='Value',var_name='Metric')


    sns.set_style('darkgrid')
    g = sns.FacetGrid(df_long, row='Metric',hue='algo',sharey=False,height=3,aspect=5)
    g.map(sns.pointplot,'data','Value',order=df_long['data'].unique(),
          errorbar=plot_errorbar)
    g.set_xticklabels(labels=xticks,rotation=45)
    g.add_legend()
    g.tight_layout()
    plt.show()
    if fileName is not None:
        g.fig.savefig(f"../data/MultiRegression/{fileName}.png")
    else:
        g.fig.savefig(
            f'../data/MultiRegression/benchMark_{regr_name}-vs-{clas_name}.png'
        )
