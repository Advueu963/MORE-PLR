from sklearn.tree import DecisionTreeClassifier

from MORE_models import PLR_MultiOutputClassifier_Chain, PLR_MultiOutputClassifier, \
    PLR_MultiOutputClassifier_Chain_Interval, PLR_RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklr.pairwise import PairwisePartialLabelRanker
from utils import transform_arrayToAPI


import utils_chain
from utils import build_plottable_evaluationDataFrame, plot_evaluation_data
from dataLinks import name_to_data


if __name__ == '__main__':
    n_jobs = -1
    random_state = 0

    estimator = DecisionTreeClassifier(
                                    random_state=random_state)
    clas_model = PairwisePartialLabelRanker(estimator)

    estimator = DecisionTreeClassifier(
                                    random_state=random_state)
    naive_multiOutputClassifier = PLR_MultiOutputClassifier(estimator=estimator)

    estimator = DecisionTreeClassifier(
                                    random_state=random_state)
    chain_multiOutputClassifier = PLR_MultiOutputClassifier_Chain(estimator=estimator, random_state=0)

    estimator = RandomForestClassifier(n_estimators=100,
                                    random_state=random_state,
                                       n_jobs=n_jobs)
    chainInterval_multiOutputclassifier = PLR_MultiOutputClassifier_Chain_Interval(estimator=estimator, random_state=0)

    native_randomForestClassifier = PLR_RandomForestClassifier(n_estimators=100,
                                                               random_state=0,
                                                               n_jobs=n_jobs)

    test_model_name = "Naive-MultiClas-DT"
    test2_model_name = "Chain-MultiClas-DT"
    test3_model_name = "ChainInterval-MultiClas-RF/DT"
    test4_model_name = "Native-RandomForestClassifier"
    clas_name = "JC-DT"
    df = build_plottable_evaluationDataFrame(name_to_data=name_to_data,
                                             models=[naive_multiOutputClassifier, chain_multiOutputClassifier, chainInterval_multiOutputclassifier, native_randomForestClassifier, clas_model],
                                             random_state=random_state,
                                             model_evaluation_function=utils_chain.model_evaluation,
                                             model_names=[test_model_name,test2_model_name,test3_model_name,test4_model_name,clas_name],
                                             model_score_function=utils_chain.model_scores)

    df.to_csv("../data/MultiRegression/clasModels.csv")
    file_name = f"benchMark_clasModelsNative-Vs-JC.png"
    plot_evaluation_data(df,file_name, list(name_to_data.keys()))
