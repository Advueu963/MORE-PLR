# This script dedicades to transform a given csv file to a latex table content
import numpy as np
import pandas as pd

def print_latexTau_x_score(dataFrame,n_rows,datasets,n_columns,algorithms):
    # 'tau_x_score', 'prediction_time', 'buckets_per_rank'
    property_of_interest = "tau_x_score"
    resulting_table = [" & ".join(["Dataset", *algorithms]) + "\\\\"]

    for i in range(n_rows):  # the rows of the table
        currentDataSet = datasets[i]
        row_content = [currentDataSet]
        for j in range(n_columns):  # the columns for the table
            currentAlgorithm = algorithms[j]

            data_mask = dataFrame["data"] == currentDataSet
            algo_mask = dataFrame["algo"] == currentAlgorithm

            interest_data = dataFrame.loc[data_mask & algo_mask, property_of_interest]

            mean, std = np.mean(interest_data), np.std(interest_data)

            # now check if the current algorithm performs best on the data
            other_algo_data = dataFrame.loc[data_mask & -algo_mask, [property_of_interest, "algo"]]
            if mean >= np.max(other_algo_data.groupby(by="algo").mean()):
                current_string = "\\bf{%.3f $\\pm$ %.3f}" % (mean, std)
            else:
                current_string = "%5.3f $\\pm$ %5.3f" % (mean, std)
            row_content.append(current_string)

        resulting_table.append(" & ".join(row_content) + " \\\\")
    resulting_table = "\n".join(resulting_table)
    print(resulting_table)


def print_latexPredictionScore(dataFrame, n_rows, datasets, n_columns, algorithms):
    # 'tau_x_score', 'prediction_time', 'buckets_per_rank'
    property_of_interest = "prediction_time"
    resulting_table = [" & ".join(["Dataset", *algorithms]) + "\\\\"]

    for i in range(n_rows):  # the rows of the table
        currentDataSet = datasets[i]
        row_content = [currentDataSet]
        for j in range(n_columns):  # the columns for the table
            currentAlgorithm = algorithms[j]

            data_mask = dataFrame["data"] == currentDataSet
            algo_mask = dataFrame["algo"] == currentAlgorithm

            interest_data = dataFrame.loc[data_mask & algo_mask, property_of_interest]

            mean, std = np.mean(interest_data), np.std(interest_data)

            # now check if the current algorithm performs best on the data
            other_algo_data = dataFrame.loc[data_mask & -algo_mask, [property_of_interest, "algo"]]
            if mean <= np.min(other_algo_data.groupby(by="algo").mean()):
                current_string = "\\bf %.3f $\\pm$ %.3f" % (mean, std)
            else:
                current_string = "$%5.3f \\pm %5.3f$" % (mean, std)
            row_content.append(current_string)

        resulting_table.append(" & ".join(row_content) + " \\\\")
    resulting_table = "\n".join(resulting_table)
    print(resulting_table)


if __name__ == '__main__':
    dataFrame  = pd.read_csv("data/MultiRegression/PLRs.csv", index_col=0)
    n_rows,datasets = len(dataFrame["data"].unique()), dataFrame["data"].unique()
    n_columns,algorithms = len(dataFrame["algo"].unique()), dataFrame["algo"].unique()

    # 'tau_x_score', 'prediction_time', 'buckets_per_rank'
    print("ACCURACY")
    print_latexTau_x_score(dataFrame,n_rows,datasets,n_columns, algorithms)
    print("-------")
    print("TIME")
    print_latexPredictionScore(dataFrame,n_rows,datasets,n_columns, algorithms)
    print("----------")