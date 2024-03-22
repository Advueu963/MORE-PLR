from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklr.pairwise import PairwisePartialLabelRanker
from utils import build_plottable_evaluationDataFrame, plot_evaluation_data
from MORE_models import PLR_RegressorChainInterval, PLR_RegressorChain, PLR_RegressorChainConformel, \
    PLR_RandomForestRegressor, PLR_LinearRegressorCalibrater, PLR_MLPCalibrater, PLR_MultiOutputRegressor
import utils_chain
from dataLinks import name_to_data_plr
import os


if __name__ == '__main__':
    n_jobs = int(os.environ['SLURM_CPUS_PER_TASK'])  # important that this equals the number of cpus on the linux cluster batch job
    random_state = 0
    
    estimator2 = DecisionTreeClassifier(random_state=random_state)
    clas_model_decisionTree = PairwisePartialLabelRanker(estimator2,
                                                         n_jobs=n_jobs)

    regr_estimator = RandomForestRegressor(n_estimators=100,
                                            n_jobs=n_jobs,
                                            random_state=random_state)
    regr_model_interval = PLR_RegressorChainInterval(estimator=regr_estimator,
                                            order=None,
                                            random_state=random_state)

    regr_estimator = DecisionTreeRegressor(
                                            random_state=random_state)
    regr_model_rounding = PLR_RegressorChain(estimator=regr_estimator,
                                            order=None,
                                            random_state=random_state)

    regr_estimator = DecisionTreeRegressor(
                                            random_state=random_state)
    regr_model_singleTarget = PLR_MultiOutputRegressor(
        estimator=regr_estimator,
        n_jobs=n_jobs
    )


    regr_estimator = DecisionTreeRegressor(
                                            random_state=random_state)
    calibration_linearMethod_model = PLR_LinearRegressorCalibrater(estimator=regr_estimator,
                                                       random_state=0)

    regr_name_singleTarget = "SingleTarget-DT"
    regr_name_interval = "Chain-RF(DT)-Interval"
    regr_name_rounding = "Chain-DT-Rounding"
    calibration_linearMethod_name = "Chain-LinearRegression-Calibration"
    calibration_MLPMethod_name = "Chain-MLP-Calibration"
    clas_name_decisionTree = "JC-DT"
    
    model = regr_model_singleTarget
    model_name = regr_name_singleTarget
    
    df = build_plottable_evaluationDataFrame(name_to_data=name_to_data_plr,
                                             models=[model],
                                             random_state=random_state,
                                             model_evaluation_function=utils_chain.model_evaluation,
                                             model_names=[model_name],
                                             model_score_function=utils_chain.model_scores)

    df.to_csv(f"../data/MultiRegression/PLR-DT/PLR_{model_name}.csv")
    #file_name = "benchMark_PLR_CHAINS-vs-JC_DT"
    #plot_evaluation_data(df,file_name, list(name_to_data_plr.keys()))