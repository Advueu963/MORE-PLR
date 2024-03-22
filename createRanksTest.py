import numpy as np


def create_apiRanks2( array):
    for i in range(array.shape[0]):
        row = array[i, :]
        for ranks in range(0, len(row)):
            try:
                next_lowest = np.min(row[np.abs(row) > ranks - 1])
            except ValueError as ve:
                break  # we have built the rank according to the api
            row = np.where(row == next_lowest, ranks, np.where(row <= ranks - 1, row, row + 1))
        array[i, :] = row + 1
    return array

erg = np.array([
    [-1,-2,-3,-4],
    [0,1,2,41],
    [-6,-6,2,1]
],dtype=np.float64)

non_api_ranks = np.round(erg)
min_val = np.min(non_api_ranks)
if min_val < 0:
    non_api_ranks = non_api_ranks - min_val # scale all values to be at least >= 0
print(non_api_ranks)
print(create_apiRanks2(non_api_ranks))