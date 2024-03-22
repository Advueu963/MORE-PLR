from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklr.pairwise import PairwisePartialLabelRanker
from utils import build_plottable_evaluationDataFrame, plot_evaluation_data
from MORE_models import PLR_RegressorChainInterval, PLR_RegressorChain, PLR_RegressorChainConformel, \
    PLR_RandomForestRegressor, PLR_LinearRegressorCalibrater, PLR_MultiOutputRegressor
import utils_chain
from dataLinks import name_to_data_lr
import os



if __name__ == '__main__':
    n_jobs = int(os.environ['SLURM_CPUS_PER_TASK'])  # important that this equals the number of cpus on the linux cluster batch job
    random_state = 0

    estimator = RandomForestClassifier(n_estimators=100,
                                            random_state=random_state)
    clas_model_randomForest = PairwisePartialLabelRanker(estimator,
                                                         n_jobs=n_jobs)

    regr_estimator = RandomForestRegressor(n_estimators=100,
                                            n_jobs=n_jobs,
                                            random_state=random_state)
    regr_model_mort = PLR_RandomForestRegressor(n_estimators=100,
                                            random_state=random_state,
                                            n_jobs=n_jobs)

    regr_estimator = RandomForestRegressor(n_estimators=100,
                                            n_jobs=n_jobs,
                                            random_state=random_state)
    regr_model_interval = PLR_RegressorChainInterval(estimator=regr_estimator,
                                            order=None,
                                            random_state=random_state)

    regr_estimator = RandomForestRegressor(n_estimators=100,
                                            n_jobs=n_jobs,
                                            random_state=random_state)
    regr_model_rounding = PLR_RegressorChain(estimator=regr_estimator,
                                            order=None,
                                            random_state=random_state)

    regr_estimator = RandomForestRegressor(n_estimators=100,
                                            n_jobs=n_jobs,
                                            random_state=random_state)
    regr_model_singleTarget = PLR_MultiOutputRegressor(
        estimator=estimator,
        n_jobs=n_jobs
    )

    regr_estimator = RandomForestRegressor(n_estimators=100,
                                            n_jobs=n_jobs,
                                            random_state=random_state)
    calibration_method_model = PLR_LinearRegressorCalibrater(estimator=regr_estimator,
                                                       random_state=0)


    regr_name_singleTarget = "SingleTarget-RF"
    regr_name_interval = "Chain-RF-Interval"
    regr_name_rounding = "Chain-RF-Rounding"
    regr_name_mort = "Native-RF"
    calibration_method_name = "Chain-LinearRegression-Calibration-RF"
    clas_name_randomForest = "JC-RF"
    
    model = regr_model_singleTarget
    model_name = regr_name_singleTarget
    
    df = build_plottable_evaluationDataFrame(name_to_data=name_to_data_lr,
                                             models=[model],
                                             random_state=random_state,
                                             model_evaluation_function=utils_chain.model_evaluation,
                                             model_names=[model_name],
                                             model_score_function=utils_chain.model_scores)

    df.to_csv(f"../data/MultiRegression/LR-RF/LR_{model_name}.csv")
    #file_name = "benchMark_LR_CHAINS-vs-JC_RF"
    #plot_evaluation_data(df,file_name, list(name_to_data_lr.keys()))