import pandas as pd
import numpy as np
import scipy.stats as ss

def print_predictionScoreRanking(dataFrame, n_datasets, datasets, n_algorithms, algorithms):
    # 'tau_x_score', 'prediction_time', 'buckets_per_rank'
    property_of_interest = "tau_x_score"
    ranks = {a:[] for a in algorithms}

    for i in range(n_datasets):  # the rows of the table
        currentDataSet = datasets[i]
        for j in range(n_algorithms):  # the columns for the table
            currentAlgorithm = algorithms[j]

            data_mask = dataFrame["data"] == currentDataSet
            algo_mask = dataFrame["algo"] == currentAlgorithm

            interest_data = dataFrame.loc[data_mask & algo_mask, property_of_interest]

            mean, std = np.mean(interest_data), np.std(interest_data)

            # now check if the current algorithm performs best on the data
            other_algo_data = dataFrame.loc[data_mask & -algo_mask, [property_of_interest, "algo"]]
            n_bigger = np.sum([1 for i in range(n_algorithms) if i != j and mean >= np.mean(
                other_algo_data.loc[other_algo_data["algo"] == algorithms[i] # get the algorithm
                                    , property_of_interest] # get the score
            )])
            ranks[currentAlgorithm].append(n_bigger)
    return pd.DataFrame(ranks).mean()

def print_predictoinScoreRankings2(dataFrame, n_datasets, datasets, algorithms):
    # 'tau_x_score', 'prediction_time', 'buckets_per_rank'
    property_of_interest = "tau_x_score"
    ranks = {a:[] for a in algorithms}

    for i in range(n_datasets):  # the rows of the table
        currentDataSet = datasets[i]

        data_mask = dataFrame["data"] == currentDataSet

        interest_data = dataFrame.loc[data_mask, [property_of_interest, "algo"]].groupby(by="algo").mean()
        non_tie_ranks = ss.rankdata(interest_data.values.flatten(), method="min")
        for i in range(len(interest_data.index)):
            algo = interest_data.index[i]
            ranks[algo].append(len(non_tie_ranks) - non_tie_ranks[i] + 1)

    return pd.DataFrame(ranks).mean()

def print_timeRanks(dataFrame, n_datasets, datasets, algorithms):
    # 'tau_x_score', 'prediction_time', 'buckets_per_rank'
    property_of_interest = "prediction_time"
    ranks = {a:[] for a in algorithms}

    for i in range(n_datasets):  # the rows of the table
        currentDataSet = datasets[i]

        data_mask = dataFrame["data"] == currentDataSet

        interest_data = dataFrame.loc[data_mask, [property_of_interest, "algo"]].groupby(by="algo").mean()
        algo_ranks = ss.rankdata(interest_data.values.flatten(), method="min")
        for i in range(len(interest_data.index)):
            algo = interest_data.index[i]
            ranks[algo].append(algo_ranks[i])

    return pd.DataFrame(ranks).mean()


if __name__ == "__main__":
    dataFrame = pd.read_csv("../data/MultiRegression/LR_and_PLR.csv",index_col=0)

    n_datasets, datasets = len(dataFrame["data"].unique()), dataFrame["data"].unique()
    n_algorithms, algorithms = len(dataFrame["algo"].unique()), dataFrame["algo"].unique()

    erg = print_predictoinScoreRankings2(dataFrame, n_datasets, datasets, algorithms)

    erg2 = print_timeRanks(dataFrame, n_datasets, datasets, algorithms)
    erg3 = (erg + erg2) / 2
    print("Output resembles the mean amount of times the tau_x_score of the algorithm was bigger (>=) than the other algorithms")
    print("RANKS (TAU_X) & " + " & ".join([str(round(x,3)) for x in erg.values]) + "\\\\")
    print("RANKS (Prediction Time) & " + " & ".join([str(round(x, 3)) for x in erg2.values]) + "\\\\")
    print("RANKS ((TAU_X + Prediction_TIME) / 2) & "+ " & ".join([str(round(x, 3)) for x in (erg3).values]) + "\\\\")
    print(erg)
    print(erg2)
    print(erg3)