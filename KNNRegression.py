from sklearn.ensemble import RandomForestClassifier, StackingRegressor, AdaBoostRegressor, AdaBoostClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklr.pairwise import PairwisePartialLabelRanker
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from utils import build_plottable_evaluationDataFrame,plot_evaluation_data
from utils_chain import model_evaluation, model_scores
from MORE_models import PLR_RegressorChain, PLR_RegressorChainInterval, PLR_MultiOutputClassifier_Chain_Interval

random_state = 0
from dataLinks import name_to_data

if __name__ == '__main__':
    n_jobs=-1
    clas_estimator = DecisionTreeClassifier(random_state=random_state)
    clas_model = PairwisePartialLabelRanker(clas_estimator)

    regr_estimator = KNeighborsRegressor(n_neighbors=5,n_jobs=n_jobs)
    knn_model = PLR_RegressorChain(estimator=regr_estimator,
                                   order=None,  # the order will later on be updated to an appropriate order
                                   random_state=random_state) # not really necessary but double is better


    regr_estimator = LinearRegression()
    linear_model = PLR_RegressorChain(estimator=regr_estimator,
                                          order=None,  # the order will later on be updated to an appropriate order
                                          random_state=random_state)  # not really necessary but double is better


    knn_model_name = "Chain-KNN"
    linear_model_name = "Linear"
    adaBoost_model_name = "AdaBoost_KNN"
    adaBoost_model_clas_name = "AdaBoost_KNN_Classification"
    clas_name = "JC-DT"
    df = build_plottable_evaluationDataFrame(name_to_data, models=[knn_model,
                                                                   linear_model,
                                                                   clas_model],
                                             random_state=random_state,
                                             model_evaluation_function=model_evaluation,
                                             model_names=[knn_model_name,
                                                          linear_model_name,
                                                          clas_name],
                                             model_score_function=model_scores)
    print(df)
    df.to_csv("../data/MultiRegression/knnChain.csv")
    file_name = f"benchMark_{knn_model_name}-vs-{clas_name}.png"
    plot_evaluation_data(dataframe=df,
                         fileName=file_name,
                         xticks=list(name_to_data.keys()))