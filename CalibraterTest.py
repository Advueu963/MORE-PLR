from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklr.pairwise import PairwisePartialLabelRanker

from utils import build_plottable_evaluationDataFrame,plot_evaluation_data
from utils_chain import model_evaluation, model_scores
from MORE_models import PLR_RandomForestRegressor, PLR_LinearRegressorCalibrater
random_state = 0
data_id = 42855
# Authorship: 42834
# IRIS: 42851
# WINE: 42872
# ECOLI: 42844
# Stock : 42862
# LIBRAS: 42855
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
name_to_data = {
    #'Authorship': 42834,
    #"BLOCKS":42836,
    #"BREAST":42838,
    #'ECOLI': 42844,
    #"GLASS":42847,

    #'IRIS': 42851,
    "LETTER":42853,
    #'LIBRAS': 42855,
    #"PENDIGITS":42856,
    #"SATIMAGE":42858,
    #"SEGMENT":42859,
    #"VEHICLE":42863,
    #"VOWEL":42865,
    #'WINE': 42872,
}
if __name__ == '__main__':
    clas_estimator = RandomForestClassifier(random_state=random_state)
    clas_model = PairwisePartialLabelRanker(clas_estimator)

    regr_estimator = RandomForestRegressor(random_state=random_state)
    regr_model = PLR_LinearRegressorCalibrater(estimator=regr_estimator,
                                random_state=random_state) # not really necessary but double is better

    regr_name = "LinearRegression-Calibrater"
    clas_name = "RF-JC"
    df = build_plottable_evaluationDataFrame(name_to_data, models=[regr_model, clas_model],
                                             random_state=random_state,
                                             model_evaluation_function=model_evaluation,
                                             model_names=[regr_name, clas_name],
                                             model_score_function=model_scores)

    df.to_csv("../data/MultiRegression/linearRegressorCalibration.csv")
    file_name = f"benchMark_{regr_name}-vs-{clas_name}.png"
    plot_evaluation_data(dataframe=df,
                         fileName=file_name,
                         xticks=list(name_to_data.keys()))