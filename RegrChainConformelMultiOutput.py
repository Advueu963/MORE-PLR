from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklr.pairwise import PairwisePartialLabelRanker
from sklearn.neighbors import KNeighborsRegressor
from MORE_models import PLR_RegressorChainConformel
from utils import build_plottable_evaluationDataFrame, plot_evaluation_data
from utils_chain import model_evaluation, model_scores
random_state = 0
from dataLinks import name_to_data

if __name__ == '__main__':
    estimator = RandomForestClassifier(random_state=random_state)
    clas_model = PairwisePartialLabelRanker(estimator)

    estimator = RandomForestRegressor(random_state=0)
    regr_model = PLR_RegressorChainConformel(estimator=estimator,
                                            order=None,
                                            random_state=0,
                                             alpha=0.5)

    regr_name = "CHAIN_RandomForest_CalOrder_Conformel"
    clas_name = "JC_RF"
    df = build_plottable_evaluationDataFrame(name_to_data,
                                             models=[regr_model, clas_model],
                                             random_state=random_state,
                                             model_evaluation_function=model_evaluation,
                                             model_score_function=model_scores,
                                             model_names=[regr_name,clas_name]
                                             )

    df.to_csv("../data/MultiRegression/chainConformel.csv")
    file_name = f"benchMark_{regr_name}-vs-{clas_name}.png"
    plot_evaluation_data(df,file_name, list(name_to_data.keys()))