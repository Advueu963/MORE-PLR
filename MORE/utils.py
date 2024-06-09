"""
    General functions used for visualization or plotting.
"""

import math
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
import seaborn as sns
import matplotlib.pyplot as plt
from time import perf_counter
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedKFold
from sklr.metrics import tau_x_score
from sklearn.preprocessing import StandardScaler


def create_apiRanks(array):
    # File to build dense ranking vectors. Example: (1,2,2,4) -> (1,2,2,3).
    min_val = np.min(array)
    if min_val < 0:
        array = (
            array - min_val
        )  # scale all values to be at least >= 0. Remark: this is just quick and dirty

    for i in range(array.shape[0]):
        row = array[i, :]
        for ranks in range(0, len(row)):
            try:
                next_lowest = np.min(row[row > ranks - 1])
            except ValueError as ve:
                break  # we have built the rank according to the api
            row = np.where(
                row == next_lowest, ranks, np.where(row <= ranks - 1, row, row + 1)
            )
        array[i, :] = row + 1
    return array


def transform_arrayToAPI(array):
    # Get the ranks of the models. Yet they can be not in the proper form of all values 1 <= value <= n_items
    non_api_ranks = np.round(array)
    # Create correct api ranks
    a = perf_counter()
    api_ranks = create_apiRanks(non_api_ranks)
    b = perf_counter()

    return api_ranks


def get_Buckets_Sizes_Counts(Y):
    unique_rankings, counts = np.unique(Y.astype(int), axis=0, return_counts=True)
    mean_bucket_size = np.sum(
        list(map(lambda x: len(set(x)), unique_rankings)) * counts
    ) / len(Y)
    return unique_rankings, mean_bucket_size, counts


def mean_bucketSize(Y):
    return get_Buckets_Sizes_Counts(Y)[1]


def amount_rankings(Y):
    return len(get_Buckets_Sizes_Counts(Y)[0])


def model_predict_round(model, test_X):
    regr_output = model.predict(test_X)
    # print(regr_output)
    return transform_arrayToAPI(np.round(regr_output))


"""
Functions for visualization
"""


def build_plottable_evaluationDataFrame(
    name_to_data,
    models,
    random_state,
    model_evaluation_function,
    model_score_function,
    model_names=[""],
):
    names, data_ids = list(name_to_data.keys()), list(name_to_data.values())
    # data frame later used for prediction
    data = {
        "data": [],
        "tau_x_score": [],
        "prediction_time": [],
        "buckets_per_rank": [],
        "algo": [],
    }

    def add_to_data(data_name, score, time, bucket_per_rank, algo):
        if type(bucket_per_rank) in [
            np.ndarray,
            list,
        ]:  # just to make sure it is a list or np.Array
            # Check if we are algo==DATA or a normal Model
            if type(score) in [np.ndarray, list]:
                assert len(score) == len(bucket_per_rank) == len(time)
                for i in range(len(score)):
                    # add the scores
                    data["data"].append(data_name)
                    data["tau_x_score"].append(score[i])
                    data["prediction_time"].append(np.sqrt(time[i]))
                    data["buckets_per_rank"].append(bucket_per_rank[i])
                    data["algo"].append(algo)
            else:
                # Special Case were we have algo == DATA
                for i in range(len(bucket_per_rank)):
                    # add the scores
                    data["data"].append(data_name)
                    data["tau_x_score"].append(score)
                    data["prediction_time"].append(np.sqrt(time))
                    data["buckets_per_rank"].append(bucket_per_rank[i])
                    data["algo"].append(algo)
        else:
            data["data"].append(data_name)
            data["tau_x_score"].append(score)
            data["prediction_time"].append(np.sqrt(time))
            data["buckets_per_rank"].append(bucket_per_rank)
            data["algo"].append(algo)

    # Options for data extraction
    as_frame = False
    return_X_y = True

    for i in range(len(data_ids)):
        X, Y = fetch_openml(
            data_id=data_ids[i], as_frame=as_frame, return_X_y=return_X_y, parser="auto"
        )
        X = np.ascontiguousarray(X)  # need for performance improvement at SVM
        Y = Y.astype(np.float64)

        # results shape (models, n_folds, 3), the 3 properties are: accuracy, speed, mean_buckets
        results = model_evaluation_function(
            models=models,
            X=X,
            Y=Y,
            random_state=random_state,
            model_score_function=model_score_function,
        )

        # Add regression Outputs
        for index_models in range(results.shape[0]):
            model_accuracies, model_times, model_bucket_sizes = results[index_models].T
            add_to_data(
                names[i],
                score=model_accuracies,
                time=model_times,
                bucket_per_rank=model_bucket_sizes,
                algo=model_names[index_models],
            )
        # Add "Best"/Data Results
        # add_to_data(names[i], *best_results, algo="Data")
        del results

        print(f"FINISHED DATA_ID: {data_ids[i]}")
    return pd.DataFrame(data)


def build_plottable_evaluationDataFrame_csvData(
    name_to_data,
    models,
    random_state,
    model_evaluation_function,
    model_score_function,
    model_names=[""],
):
    names, csv_file_name = list(name_to_data.keys()), list(name_to_data.values())
    # data frame later used for prediction
    data = {
        "data": [],
        "tau_x_score": [],
        "prediction_time": [],
        "buckets_per_rank": [],
        "algo": [],
    }

    def add_to_data(data_name, score, time, bucket_per_rank, algo):
        if type(bucket_per_rank) in [
            np.ndarray,
            list,
        ]:  # just to make sure it is a list or np.Array
            # Check if we are algo==DATA or a normal Model
            if type(score) in [np.ndarray, list]:
                assert len(score) == len(bucket_per_rank) == len(time)
                for i in range(len(score)):
                    # add the scores
                    data["data"].append(data_name)
                    data["tau_x_score"].append(score[i])
                    data["prediction_time"].append(np.sqrt(time[i]))
                    data["buckets_per_rank"].append(bucket_per_rank[i])
                    data["algo"].append(algo)
            else:
                # Special Case were we have algo == DATA
                for i in range(len(bucket_per_rank)):
                    # add the scores
                    data["data"].append(data_name)
                    data["tau_x_score"].append(score)
                    data["prediction_time"].append(np.sqrt(time))
                    data["buckets_per_rank"].append(bucket_per_rank[i])
                    data["algo"].append(algo)
        else:
            data["data"].append(data_name)
            data["tau_x_score"].append(score)
            data["prediction_time"].append(np.sqrt(time))
            data["buckets_per_rank"].append(bucket_per_rank)
            data["algo"].append(algo)

    for i in range(len(csv_file_name)):
        # Read Data
        dataFrame = pd.read_csv(f"data/{csv_file_name[i]}.csv")

        # Split in Features and Targets
        X, Y = dataFrame.iloc[:, :-6], dataFrame.iloc[:, -6:]

        Y = Y.astype(np.float64)

        # results shape (models, n_folds, 3), the 3 properties are: accuracy, speed, mean_buckets
        results = model_evaluation_function(
            models=models,
            X=X,
            Y=Y,
            random_state=random_state,
            model_score_function=model_score_function,
        )

        # Add Outputs ot dataFrame
        for index_models in range(results.shape[0]):
            model_accuracies, model_times, model_bucket_sizes = results[index_models].T
            add_to_data(
                names[i],
                score=model_accuracies,
                time=model_times,
                bucket_per_rank=model_bucket_sizes,
                algo=model_names[index_models],
            )
        # Add "Best"/Data Results
        # add_to_data(names[i], *best_results, algo="Data")
        del results

        print(f"FINISHED DATA_FILE: {csv_file_name[i]}")
    return pd.DataFrame(data)


def build_plottable_evaluationDataFrame_missingLabels(
    name_to_data,
    models,
    random_state,
    percentage,
    model_evaluation_function,
    model_score_function,
    model_names=[""],
):
    names, data_ids = list(name_to_data.keys()), list(name_to_data.values())
    # data frame later used for prediction
    data = {
        "data": [],
        "tau_x_score": [],
        "prediction_time": [],
        "buckets_per_rank": [],
        "algo": [],
        "percentage": [],
    }

    def add_to_data(data_name, score, time, bucket_per_rank, algo, percentage):
        if type(bucket_per_rank) in [
            np.ndarray,
            list,
        ]:  # just to make sure it is a list or np.Array
            assert len(score) == len(bucket_per_rank) == len(time)
            for i in range(len(score)):
                # add the scores
                data["data"].append(data_name)
                data["tau_x_score"].append(score[i])
                data["prediction_time"].append(np.sqrt(time[i]))
                data["buckets_per_rank"].append(bucket_per_rank[i])
                data["algo"].append(algo)
                data["percentage"].append(percentage)
        else:
            data["data"].append(data_name)
            data["tau_x_score"].append(score)
            data["prediction_time"].append(np.sqrt(time))
            data["buckets_per_rank"].append(bucket_per_rank)
            data["algo"].append(algo)
            data["percentage"].append(percentage)

    # Options for data extraction
    as_frame = False
    return_X_y = True

    for i in range(len(data_ids)):
        X, Y = fetch_openml(
            data_id=data_ids[i], as_frame=as_frame, return_X_y=return_X_y, parser="auto"
        )
        X = np.ascontiguousarray(X)  # need for performance improvement at SVM
        Y = Y.astype(np.float64)

        # results shape (models, n_folds, 3), the 3 properties are: accuracy, speed, mean_buckets
        results = model_evaluation_function(
            models=models,
            X=X,
            Y=Y,
            random_state=random_state,
            percentage_missing_labels=percentage,
            model_score_function=model_score_function,
        )

        # Add regression Outputs
        for index_models in range(results.shape[0]):
            model_accuracies, model_times, model_bucket_sizes = results[index_models].T
            add_to_data(
                names[i],
                score=model_accuracies,
                time=model_times,
                bucket_per_rank=model_bucket_sizes,
                algo=model_names[index_models],
                percentage=percentage,
            )
        # Add "Best"/Data Results
        # add_to_data(names[i], *best_results, algo="Data")
        del results  # flush memory

        print(f"FINISHED DATA_ID: {data_ids[i]}")
    return pd.DataFrame(data)


def plot_errorbar(array_x, normalizer=5):
    mean_x = np.mean(array_x)
    std_x = np.std(array_x)
    return (
        mean_x - (std_x / np.sqrt(normalizer) * 1.96),
        mean_x + (std_x / np.sqrt(normalizer) * 1.96),
    )


def plot_evaluation_data(dataframe: pd.DataFrame, fileName, xticks):
    df_long = pd.melt(
        dataframe, id_vars=["data", "algo"], value_name="Value", var_name="Metric"
    )

    sns.set_style("darkgrid")
    g = sns.FacetGrid(
        df_long, row="Metric", hue="algo", sharey=False, height=3, aspect=5
    )
    g.map(
        sns.pointplot,
        "data",
        "Value",
        order=df_long["data"].unique(),
        dodge=True,
        errorbar=plot_errorbar,
    )
    g.set_xticklabels(labels=xticks, rotation=45)
    g.add_legend()
    g.tight_layout()
    plt.show()
    g.figure.savefig(f"data/MultiRegression/{fileName}.png")


"""
Missing labels Utils
"""


def optimism_change_Y(X, Y):
    return X, create_apiRanks(
        Y
    )  # Negative Values are ranked higher than positive values. Therefore we can just call this function


def pessimism_change_Y(X, Y):
    return X, create_apiRanks(
        np.where(
            Y < 0, Y.shape[1] + 1, Y
        )  # change the -1 to the biggest values which hurt the api form. The ranks are then readjusted such that exactly these values resemble lowest rank (highest value)
    )


def optimism_data_approach(X, Y):
    # Assume that potentially Y contains -1 == Nan missing labels
    missing_ranks = np.any(Y < 0, axis=1)  # save the indices we change
    non_change_X, non_change_Y = X[~missing_ranks], Y[~missing_ranks]
    change_X, change_Y = optimism_change_Y(X[missing_ranks], Y[missing_ranks])

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
    _, Y_opt = optimism_change_Y(X[missing_ranks], Y[missing_ranks])
    _, Y_pes = optimism_change_Y(X[missing_ranks], Y[missing_ranks])

    Y_opt_pes = np.zeros(shape=(Y.shape[0] + Y_opt.shape[0], Y.shape[1]))
    # Transform missing_ranks so that:
    # 1.this mask has doubled the True and leaved the False unchanged
    # 2. Therefore we have both possibilities directly after each other and the unchanged are at the "same" position (not really but relatively same position) .
    repeats = missing_ranks + 1
    missing_ranks = (
        np.repeat(  # this mask has doubled the True and leaved the False unchanged
            missing_ranks,
            repeats=repeats,
            axis=0,
        )
    )
    X_opt_pes = np.repeat(X, repeats=repeats, axis=0)

    Y_opt_pes[~missing_ranks] = non_change_Y
    modulo_uneven = (
        np.array(range(missing_ranks.size)) % 2 == 1
    )  # must be boolean array otherwise the broadcast wont work
    Y_opt_pes[(missing_ranks & modulo_uneven)] = Y_opt
    Y_opt_pes[(missing_ranks & ~modulo_uneven)] = Y_pes

    return X_opt_pes, Y_opt_pes


def create_missing_labels2(Y, percentage, random_state=0):
    random_generator = np.random.default_rng(random_state)  # reproducibility
    n_rows, n_columns = Y.shape
    for i in range(n_rows):
        for j in range(n_columns):
            if random_generator.binomial(1, percentage):
                Y[i, j] = -1
    return Y


def create_missing_labels(Y, percentage, random_state=0):
    # source: https://stackoverflow.com/questions/31389481/numpy-replace-random-elements-in-an-array
    random_generator = np.random.default_rng(random_state)  # reproducibility
    shape = Y.shape  # Store original shape
    temp = Y.flatten()  # Flatten to 1D
    inds = random_generator.choice(
        temp.size, size=int(np.floor(temp.size * percentage)), replace=False
    )  # Get random indices
    temp[inds] = -1
    temp = temp.reshape(
        shape
    )  # brings back the original Y. Does it holds: Y == Y.flatten().reshape(Y.shape)

    return temp


"""
    This section consists auxiliary functions to evaluate the performance of the tested models in the typical sklearn api.

"""


def model_scores(model, train_X, train_Y, test_X, test_Y):
    """
    Outputs the model tau_x_score, when fitted on the training data and predicting the test data.
    Additionally it measures the overall time needed for both training an prediction and outputs the mean bucket size for the bucket orders.
    Note: The features are normalized.
    Args:
        model (sklearn.model): The model to be used.
        train_X (np.ndarray): The training data point features
        train_Y (np.ndarray): The training data point targets
        test_X (np.ndarray): The test data point features
        test_Y (np.ndarray): The test data point targets

    Returns:
        triple: (tau_x_score, training and prediction time, mean_bucket_size)
    """
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

    # try:
    #    tau_x_score(Y_pred_model, test_Y)
    # except ValueError as ve:
    #    print("PROBLEM")

    print(f"TIME OF {model}: {b-a}")
    return (
        tau_x_score(Y_pred_model, test_Y),
        b - a,
        mean_bucketSize(Y_pred_model),
    )


def model_scores_Pipeline(model, train_X, train_Y, test_X, test_Y):
    """
    Similiar to *mode_scores* but instead does not normalize features.


    Args:
        model (sklearn.pipeline): A pipeline containing transformations and at the end a model to be used.
        train_X (np.ndarray): The training data point features
        train_Y (np.ndarray): The training data point targets
        test_X (np.ndarray): The test data point features
        test_Y (np.ndarray): The test data point targets

    Returns:
        triple: (tau_x_score, training and prediction time, mean_bucket_size)
    """
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
        b - a,
        mean_bucketSize(Y_pred_model),
    )


def model_evaluation(models, X, Y, random_state, model_score_function):
    """
    Gathers *model_score_function* for each model in models on X and Y.
    For this it performs 5 * 10-fol CV on X and Y executing *model_score_function* on each split.


    Args:
        models ([sklearn.model]): List of models that should be evaluated
        X (np.ndarray): The data points
        Y (np.ndarray): The targets
        random_state (int): random state for the 5 * 10-fold CV
        model_score_function (function): Either *model_scores* or model_scores_Pipeline*. 
            More generally should be a function receiving model, train_data, test_data and returning the (tau_x_score, train and prediction time, mean bucket rank) in this order.
            

    Returns:
        np.ndarray: A three dimensional array where the first dimension are the different models, the second the folds and the third containing (tau_x_score, train and prediction time, mean bucket rank)
    """
    # The Output of clas_model is assumed to be in the correct api format
    n_folds = 10
    n_repeats = 5
    folder = RepeatedKFold(
        n_splits=n_folds, n_repeats=n_repeats, random_state=random_state
    )

    res = np.zeros(shape=(len(models), n_folds * n_repeats, 3))
    split = 0
    for train_idx, test_idx in folder.split(X, Y):
        X_train, X_test, Y_train, Y_test = (
            X[train_idx],
            X[test_idx],
            Y[train_idx],
            Y[test_idx],
        )
        for i in range(len(models)):
            res[i, split] = model_score_function(
                model=models[i],
                train_X=X_train,
                train_Y=Y_train,
                test_X=X_test,
                test_Y=Y_test,
            )

        split += 1
    # res has its first dimension distinguishing between the different models
    # In its second dimension it distinguished between the diffrent fols
    # The third dimension holds: accuracy, time, mean_buckets_per_rank in this order
    return res


def model_evaluation_missingLabels(
    models, X, Y, random_state, model_score_function, percentage_missing_labels
):
    """
    Gathers *model_score_function* for each model in models on X and Y.
    For this it performs 5 * 10-fol CV on X and Y executing *model_score_function* on each split.
    The training targets Y_train in each fold have *percentage_missing_labels* to evaluate the models with missing labels.


    Args:
        models ([sklearn.models]): _description_
        X (np.ndarray): The data points
        Y (np.ndarray): The targets
        random_state (int): random state for the 5 * 10-fold CV
        model_score_function (function): Either *model_scores* or model_scores_Pipeline*.
            More generally should be a function receiving model, train_data, test_data and returning the (tau_x_score, train and prediction time, mean bucket rank) in this order.
        percentage_missing_labels (float): proportion of missingl targets in the training data.

    Returns:
        np.ndarray: A three dimensional array where the first dimension are the different models, the second the folds and the third containing (tau_x_score, train and prediction time, mean bucket rank)
    """
    # The Output of clas_model is assumed to be in the correct api format
    n_folds = 10
    n_repeats = 5
    folder = RepeatedKFold(
        n_splits=n_folds, n_repeats=n_repeats, random_state=random_state
    )

    res = np.zeros(shape=(len(models), n_folds * n_repeats, 3))
    split = 0
    for train_idx, test_idx in folder.split(X, Y):
        X_train, X_test, Y_train, Y_test = (
            X[train_idx],
            X[test_idx],
            Y[train_idx],
            Y[test_idx],
        )
        Y_train = create_missing_labels(
            Y_train, percentage=percentage_missing_labels, random_state=random_state
        )
        for i in range(len(models)):
            res[i, split] = model_score_function(
                model=models[i],
                train_X=X_train,
                train_Y=Y_train,
                test_X=X_test,
                test_Y=Y_test,
            )

        split += 1
    # res has its first dimension distinguishing between the different models
    # In its second dimension it distinguished between the diffrent fols
    # The third dimension holds: accuracy, time, mean_buckets_per_rank in this order
    return res
