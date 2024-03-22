from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
import pandas as pd
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, RationalQuadratic, ExpSineSquared,ConstantKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, SVR

from MORE_models import PLR_RegressorChainInterval, PLR_RegressorChain, PLR_RegressorChainConformel, \
    PLR_RandomForestRegressor, PLR_LinearRegressorCalibrater, PLR_MLPCalibrater, PLR_MultiOutputRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklr.pairwise import PairwisePartialLabelRanker
from utils_chain import model_evaluation, model_scores_Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from MORE_models import PLR_RegressorChainInterval, PLR_RegressorChain, PLR_RegressorChainConformel, \
    PLR_RandomForestRegressor, PLR_LinearRegressorCalibrater, PLR_GaussianProcessRegressor
from utils import plot_evaluation_data
import os
if __name__ == "__main__":
    random_state = 0
    n_jobs = int(os.environ['SLURM_CPUS_PER_TASK'])  # important that this equals the number of cpus on the linux cluster batch job

    data_files = ["PLR-REAL-Political"]

    random_state = 0

    ##### State of the art ######
    estimator = DecisionTreeClassifier(random_state=random_state)
    clas_model_decisionTree = PairwisePartialLabelRanker(estimator,
                                                         n_jobs=n_jobs)

    estimator = RandomForestClassifier(n_estimators=100,
                                       n_jobs=n_jobs,
                                       random_state=random_state)
    clas_model_randomForest = PairwisePartialLabelRanker(estimator,
                                                         n_jobs=n_jobs)

    clas_estimator = SVC(probability=True, random_state=random_state, cache_size=2000)
    clas_model_svm = PairwisePartialLabelRanker(clas_estimator,
                                                n_jobs=n_jobs)

    #### Single Target #####
    regr_estimator = DecisionTreeRegressor(
        random_state=random_state)
    regr_model_singleTarget_DT = PLR_MultiOutputRegressor(
        estimator=regr_estimator,
        n_jobs=n_jobs    )
    regr_estimator = RandomForestRegressor(
        random_state=random_state)
    regr_model_singleTarget_RF = PLR_MultiOutputRegressor(
        estimator=regr_estimator,
        n_jobs=n_jobs    )
    regr_estimator = SVR(cache_size=20000)
    regr_model_singleTarget_SVM = PLR_MultiOutputRegressor(
        estimator=regr_estimator,
        n_jobs=n_jobs    )

    ###### DT ######
    regr_estimator = DecisionTreeRegressor(
        random_state=random_state)
    regr_model_rounding_dt = PLR_RegressorChain(estimator=regr_estimator,
                                             order=None,
                                             random_state=random_state)

    regr_estimator = DecisionTreeRegressor(
        random_state=random_state)
    regr_model_calibration_dt = PLR_LinearRegressorCalibrater(estimator=regr_estimator,
                                                                   random_state=0)


    ##### RFR #####
    regr_model_mort = PLR_RandomForestRegressor(random_state=random_state, n_jobs=n_jobs)
    regr_estimator = RandomForestRegressor(n_estimators=100,
                                            n_jobs=n_jobs,
                                            random_state=random_state)
    regr_model_interval = PLR_RegressorChainInterval(estimator=regr_estimator,
                                            order=None,
                                            random_state=random_state)
    regr_estimator = RandomForestRegressor(n_estimators=100,
                                            n_jobs=n_jobs,
                                            random_state=random_state)
    regr_model_rounding_rf = PLR_RegressorChain(estimator=regr_estimator,
                                                order=None,
                                                random_state=random_state)
    regr_estimator = RandomForestRegressor(n_estimators=100,
                                            n_jobs=n_jobs,
                                            random_state=random_state)
    regr_method_calibration_rf = PLR_LinearRegressorCalibrater(estimator=regr_estimator,
                                                               random_state=0)

    ###### SVR #######
    regr_estimator = SVR(cache_size=2000)
    regr_model_rounding_svr = PLR_RegressorChain(estimator=regr_estimator,
                                          order=None,  # the order will later on be updated to an appropriate order
                                          random_state=random_state)  # not really necessary but double is better


    regr_name_interval_rf = "Chain-RF-Interval"
    regr_name_rounding_rf = "Chain-RF-Rounding"
    regr_name_rounding_dt ="Chain-DT-Rounding"
    regr_name_rounding_svr = "CHAIN-SVR"
    regr_name_singleTarget_dt = "SingleTarget-DT"
    regr_name_singleTarget_rf = "SingleTarget-RF"
    regr_name_singleTarget_svm = "SingleTarget-SVR"
    regr_name_mort = "Native-RF"
    clas_name_decisionTree = "JC-DT"
    clas_name_randomForest = "JC-RF"
    clas_name_svm = "SVC-JC"


    plotData = {'data': [], 'tau_x_score': [], 'prediction_time': [], 'buckets_per_rank': [],
            'algo': []}
    model_names=[
        regr_name_interval_rf,
        regr_name_rounding_rf,
        regr_name_rounding_dt,
        regr_name_rounding_svr,
        regr_name_singleTarget_dt,
        regr_name_singleTarget_rf,
        regr_name_singleTarget_svm,
        regr_name_mort,
        clas_name_decisionTree,
        clas_name_randomForest,
        clas_name_svm
    ]

    def add_to_data(data_name, score, time, bucket_per_rank, algo):
        if (type(bucket_per_rank) in [np.ndarray, list]):  # just to make sure it is a list or np.Array
            # Check if we are algo==DATA or a normal Model
            if (type(score) in [np.ndarray, list]):
                assert len(score) == len(bucket_per_rank) == len(time)
                for i in range(len(score)):
                    # add the scores
                    plotData['data'].append(data_name)
                    plotData['tau_x_score'].append(score[i])
                    plotData['prediction_time'].append(np.sqrt(time[i]))
                    plotData['buckets_per_rank'].append(bucket_per_rank[i])
                    plotData['algo'].append(algo)
            else:
                # Special Case were we have algo == DATA
                for i in range(len(bucket_per_rank)):
                    # add the scores
                    plotData['data'].append(data_name)
                    plotData['tau_x_score'].append(score)
                    plotData['prediction_time'].append(np.sqrt(time))
                    plotData['buckets_per_rank'].append(bucket_per_rank[i])
                    plotData['algo'].append(algo)
        else:
            plotData['data'].append(data_name)
            plotData['tau_x_score'].append(score)
            plotData['prediction_time'].append(np.sqrt(time))
            plotData['buckets_per_rank'].append(bucket_per_rank)
            plotData['algo'].append(algo)

    for file in data_files:
        #Read Data
        dataFrame = pd.read_csv(f"../data/{file}.csv")

        # Split in Features and Targets
        X, Y = dataFrame.iloc[:, :-6], dataFrame.iloc[:, -6:]

        # Extract the numpy arrays
        X_data = X.values
        Y_data = Y.values

        # Build Numerical Preprocessor
        numerical_cols = [i for i, colname in enumerate(X.columns) if X[colname].dtype in ["int64", "float64"]]
        numerical_transformer = Pipeline(steps=[
            ("imputer_nan", SimpleImputer(strategy="most_frequent")),
            ("impute_977", SimpleImputer(missing_values=977, strategy="median")),
            ("scaler", StandardScaler())
        ])

        # Build Categorical Preprocessor
        categorical_cols = [i for i, colname in enumerate(X.columns) if X[colname].dtype == "object"]
        categorical_transformer = Pipeline(steps=[
            ("imputer_nan", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])
        preprocessor = ColumnTransformer(transformers=[
            ('numerical', numerical_transformer, numerical_cols),
            ('categorical', categorical_transformer, categorical_cols)
        ])

        ############ Random Forest Base Estimator##############

        #MORT
        mort_Pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            (regr_name_mort, regr_model_mort)
        ])
        #Chain-RF-Rounding
        roundingRF_Pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            (regr_name_rounding_rf, regr_model_rounding_rf)
        ])
        #Chain-RF-Interval
        intervalRF_Pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            (regr_name_interval_rf, regr_model_interval)
        ])
        # Singletarget-RF
        singleTargetRF_Pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            (regr_name_singleTarget_rf, regr_model_singleTarget_RF)
        ])
        #JC-RF
        jc_Pipeline_rf = Pipeline(steps=[
            ("preprocessor", preprocessor),
            (clas_name_randomForest, clas_model_randomForest)
        ])

        ######### Decision Tree Base Learner ###########
        #Rounding-DT
        roundingDT_Pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            (regr_name_rounding_dt, regr_model_rounding_dt)
        ])
        #SingleTarget-DT
        singleTargetDT_Pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            (regr_name_singleTarget_dt, regr_model_singleTarget_DT)
        ])
        #JC-DT
        jc_Pipeline_dt = Pipeline(steps=[
            ("preprocessor", preprocessor),
            (clas_name_decisionTree, clas_model_decisionTree)
        ])

        ########## SVM Base Learner ##################
        #Rounding-SVR
        roundingSVR_Pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            (regr_name_rounding_svr, regr_model_rounding_svr)
        ])

        #SingleTarget-SVR
        singleTargetSVR_Pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            (regr_name_singleTarget_svm, regr_model_singleTarget_SVM)
        ])

        #JC-SVM
        jc_Pipeline_svm = Pipeline(steps=[
            ("preprocessor", preprocessor),
            (clas_name_svm, clas_model_svm)
        ])

        #JC-SVM
        jc_Pipeline_svm = Pipeline(steps=[
            ("preprocessor", preprocessor),
            (clas_name_svm, clas_model_svm)
        ])


        # res.shape == (n_models, n_splits, 3) with accuracy speed and mean-buckets
        results = model_evaluation(
            models=[
                intervalRF_Pipeline,
                roundingRF_Pipeline,
                roundingDT_Pipeline,
                roundingSVR_Pipeline,
                singleTargetDT_Pipeline,
                singleTargetRF_Pipeline,
                singleTargetSVR_Pipeline,
                mort_Pipeline,
                jc_Pipeline_dt,
                jc_Pipeline_rf,
                jc_Pipeline_svm
            ],
            X=X_data,
            Y=Y_data,
            random_state=random_state,
            model_score_function=model_scores_Pipeline
        )

        # Add Outputs ot dataFrame
        for index_models in range(results.shape[0]):
            model_accuracies, model_times, model_bucket_sizes = results[index_models].T
            add_to_data(data_name=file, score=model_accuracies, time=model_times, bucket_per_rank=model_bucket_sizes,
                        algo=model_names[index_models])

    plotData = pd.DataFrame(plotData)
    plotData.to_csv("../data/MultiRegression/PLR_politicalEvaluation.csv")
    #file_name = f"benchMarkPolitical_MORE-vs-JC"
    #plot_evaluation_data(plotData,file_name,data_files)

