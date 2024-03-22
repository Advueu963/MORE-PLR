from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklr.pairwise import PairwisePartialLabelRanker

from utils import build_plottable_evaluationDataFrame,plot_evaluation_data
from utils_chain import model_evaluation, model_scores
from MORE_models import PLR_MultiOutputRegressor

random_state = 0
name_to_data_plr = {
        "PLR-AUTHORSHIP":42835,
        "PLR-BLOCKS":42836,
        "PLR-BREAST":42838,
        'PLR-ECOLI': 42844,
        "PLR-GLASS":42848,
        "PLR-IRIS":42871,
        #"PLR-LETTER":42853,
        'PLR-LIBRAS': 42855,
        "PLR-PENDIGITS":42857,
        "PLR-SATIMAGE":42858,
        "PLR-SEGMENT":42860,
        "PLR-VEHICLE":42864,
        "PLR-VOWEL":42866,
        'PLR-WINE': 42872,
        "PLR-YEAST":42870,
        # REAL DATA SETS
        "PLR-REAL-ALGAE":45755,
        "PLR-REAL-MOVIES":45738
}
if __name__ == '__main__':
    n_jobs=-1
    clas_estimator = RandomForestClassifier(random_state=random_state)
    clas_model = PairwisePartialLabelRanker(clas_estimator)

    regr_estimator = RandomForestRegressor(random_state=random_state)
    regr_model = PLR_MultiOutputRegressor(estimator=regr_estimator,
                                            n_jobs=n_jobs)

    regr_name = "SingleTarget-RF"
    clas_name = "JC-RF"
    df = build_plottable_evaluationDataFrame(name_to_data_plr,
                                             models=[regr_model, clas_model],
                                             random_state=random_state,
                                             model_evaluation_function=model_evaluation,
                                             model_names=[regr_name, clas_name],
                                             model_score_function=model_scores)
    print(df)
    df.to_csv("../data/MultiRegression/PLR-rfSingleTarget.csv")
    file_name = f"benchMark_PLR_{regr_name}-vs-{clas_name}"
    plot_evaluation_data(dataframe=df,
                         fileName=file_name,
                         xticks=list(name_to_data_plr.keys()))