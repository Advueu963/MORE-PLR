import time
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, RationalQuadratic, ExpSineSquared,ConstantKernel
from sklearn.tree import DecisionTreeClassifier

from MORE_models import PLR_GaussianProcessRegressor
from sklearn.ensemble import RandomForestClassifier
from sklr.pairwise import PairwisePartialLabelRanker
from utils import build_plottable_evaluationDataFrame, plot_evaluation_data
from utils_chain import model_evaluation, model_scores

#kernel = 5*RationalQuadratic(length_scale=0.9) * RationalQuadratic(length_scale=0.1) + 10*RationalQuadratic(length_scale=2) + 0.2**2*WhiteKernel() * ExpSineSquared()
#kernel = RBF(length_scale=2)*RBF(length_scale=3)*RBF(length_scale=4) + 0.1*WhiteKernel(noise_level=0.9) + DotProduct(sigma_0=0.5)*ConstantKernel(2)
k1 = 20*RBF(length_scale=0.9)*RBF(length_scale=3)*RBF(length_scale=0.1)
k2 = 0.1 * RBF(length_scale=0.01) + 0.2*WhiteKernel(noise_level=4)
k3 = 0.9 * RationalQuadratic(length_scale=0.001, alpha=10) * ExpSineSquared(length_scale=2,periodicity=0.8)
k4 =10* RationalQuadratic(length_scale=0.1, alpha=0.1) * RationalQuadratic(length_scale=4, alpha=4)
kernel= k1 + k2 + k3 + k4

random_state = 0
name_to_data_plr = {
        "PLR-AUTHORSHIP":42835,
        "PLR-BLOCKS":42836,
        "PLR-BREAST":42838,
        'PLR-ECOLI': 42844,
        "PLR-GLASS":42848,
        "PLR-IRIS":42871,
        "PLR-LETTER":42853,
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
    estimator = RandomForestClassifier(random_state=random_state)
    rf_model_clas = PairwisePartialLabelRanker(estimator)

    estimator = DecisionTreeClassifier(random_state=random_state)
    dt_model_clas = PairwisePartialLabelRanker(estimator)

    regr_model = PLR_GaussianProcessRegressor(kernel=kernel,
                                              random_state=random_state,
                                              n_restarts_optimizer=0)

    regr_name = "GaussianProcess"
    rf_model_clas_name = "RF-JC"
    dt_model_clas_name = "DT-JC"

    df = build_plottable_evaluationDataFrame(name_to_data_plr,
                                             models=[regr_model],
                                             random_state=random_state,
                                             model_evaluation_function=model_evaluation,
                                             model_score_function=model_scores,
                                             model_names=[regr_name])

    df.to_csv("../data/MultiRegression/PLR-GPR/PLR-gaussianProcess.csv")
    #file_name = f"benchMark_PLR_GPR-vs-JC"
    #plot_evaluation_data(df, file_name, list(name_to_data_plr.keys()))
