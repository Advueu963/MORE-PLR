from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklr.pairwise import PairwisePartialLabelRanker
from utils import build_plottable_evaluationDataFrame, plot_evaluation_data
from MORE_models import PLR_RegressorChainInterval
import utils_chain
if __name__ == '__main__':

    random_state = 0
    from dataLinks import name_to_data

    estimator = RandomForestClassifier(random_state=random_state)
    clas_model = PairwisePartialLabelRanker(estimator)

    regr_estimator = RandomForestRegressor(random_state=random_state)
    regr_model = PLR_RegressorChainInterval(estimator=regr_estimator,
                                            order=None,
                                            random_state=random_state)
    regr_name = "CHAIN_INTERVAL_RandomForest_CalOrder"
    clas_name = "JC_RF"
    df = build_plottable_evaluationDataFrame(name_to_data=name_to_data,
                                             models=[regr_model,clas_model],
                                             random_state=random_state,
                                             model_evaluation_function=utils_chain.model_evaluation,
                                             model_names=[regr_name,clas_name],
                                             model_score_function=utils_chain.model_scores)

    df.to_csv("../data/MultiRegression/chainInterval.csv")
    file_name = f"benchMark_{regr_name}-vs-{clas_name}.png"
    plot_evaluation_data(df,file_name, list(name_to_data.keys()))