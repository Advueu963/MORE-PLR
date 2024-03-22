from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklr.pairwise import PairwisePartialLabelRanker

from utils import build_plottable_evaluationDataFrame,plot_evaluation_data
from utils_chain import model_evaluation, model_scores
from MORE_models import PLR_RegressorChain
random_state = 0
from dataLinks import name_to_data

if __name__ == '__main__':
    clas_estimator = DecisionTreeClassifier(random_state=random_state)
    clas_model = PairwisePartialLabelRanker(clas_estimator)

    regr_estimator = DecisionTreeRegressor(random_state=random_state)
    regr_model = PLR_RegressorChain(estimator=regr_estimator,
                                order=None,# the order will later on be updated to an appropriate order
                                random_state=random_state) # not really necessary but double is better

    regr_name = "Chain-DT"
    clas_name = "JC-DT"
    df = build_plottable_evaluationDataFrame(name_to_data, models=[regr_model, clas_model],
                                             random_state=random_state,
                                             model_evaluation_function=model_evaluation,
                                             model_names=[regr_name, clas_name],
                                             model_score_function=model_scores)
    print(df)
    df.to_csv("../data/MultiRegression/rfChain.csv")
    file_name = f"benchMark_{regr_name}-vs-{clas_name}.png"
    plot_evaluation_data(dataframe=df,
                         fileName=file_name,
                         xticks=list(name_to_data.keys()))