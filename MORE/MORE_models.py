# external
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import (
    RegressorChain,
    MultiOutputRegressor,
    _fit_estimator,
)
from sklearn.utils import Bunch, _safe_indexing
from sklearn.utils._metadata_requests import _routing_enabled, process_routing
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import (
    check_is_fitted,
    has_fit_parameter,
    _is_arraylike,
    _num_samples,
    _make_indexable,
)

import numpy as np
import scipy.sparse as sp
import pandas as pd
from time import perf_counter
import threading
from joblib import effective_n_jobs

# code intern
from MORE.utils import (
    transform_arrayToAPI,
    optimism_data_approach,
    pessimism_data_approach,
    optimism_pessimism_data_approach,
)
from _overlap_intervals import get_overlaps

"""
    This file contains all evaluated Models of the paper.

"""


def _check_method_params(X, params, indices=None):
    # source: https://github.com/scikit-learn/scikit-learn/blob/f07e0138bfee41cd2c0a5d0251dc3fe03e6e1084/sklearn/utils/validation.py#L2114
    """Check and validate the parameters passed to a specific
    method like `fit`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data array.

    params : dict
        Dictionary containing the parameters passed to the method.

    indices : array-like of shape (n_samples,), default=None
        Indices to be selected if the parameter has the same size as `X`.

    Returns
    -------
    method_params_validated : dict
        Validated parameters. We ensure that the values support indexing.
    """

    method_params_validated = {}
    for param_key, param_value in params.items():
        if not _is_arraylike(param_value) or _num_samples(param_value) != _num_samples(
            X
        ):
            # Non-indexable pass-through (for now for backward-compatibility).
            # https://github.com/scikit-learn/scikit-learn/issues/15805
            method_params_validated[param_key] = param_value
        else:
            # Any other method_params should support indexing
            # (e.g. for cross-validation).
            method_params_validated[param_key] = _make_indexable(param_value)
            method_params_validated[param_key] = _safe_indexing(
                method_params_validated[param_key], indices
            )

    return method_params_validated


def _accumulate_prediction(predict, X, out, lock):
    """
    This is a utility function for joblib's Parallel.

    It can't go locally in ForestClassifier or ForestRegressor, because joblib
    complains that it cannot pickle it when placed there.
    """
    prediction = predict(X, check_input=False)
    with lock:
        if len(out) == 1:
            out[0] += prediction
        else:
            for i in range(len(out)):
                out[i] += prediction[i]


def find_chain_order(X, Y):
    Z = np.append(X, Y, axis=1)
    correlationMatrix = pd.DataFrame(
        Z,
        columns=[f"X{x}" for x in range(X.shape[1])]
        + [f"Y{x}" for x in range(Y.shape[1])],
    ).corr(method="kendall")
    # sns.heatmap(correlationMatrix, annot=True)
    # plt.show()

    _, n_features = X.shape
    _, n_classes = Y.shape
    erg = []

    relevant_indices = [
        (x, x + n_features) for x in range(n_classes)
    ]  # the indices of all the Y columns in correlationMatrix
    feature_indices = [x for x in range(n_features)]
    for i in range(n_classes):
        # restrict the correlation matrix to the relevant values
        relevant_area = np.power(
            correlationMatrix.values[feature_indices, :][
                :, [y for _, y in relevant_indices]
            ],
            2,
        )
        # Sum up the correlation values to determine target with best correlation
        correlation_points = np.sum(relevant_area, axis=0)
        # get target with highest correlation
        target_with_max_corr = np.argmax(correlation_points)
        # append original target index to resulting order
        erg.append(relevant_indices[target_with_max_corr][0])
        # expand the feature space with the new found best target index
        # Simulates the idea of Regressor Chain
        feature_indices.append(relevant_indices[target_with_max_corr][1])
        # remove the new target index form the relevant indices
        relevant_indices.remove(relevant_indices[target_with_max_corr])
    # print("BEST ORDER: ", erg)
    return erg


def _partition_estimators(n_estimators, n_jobs):
    """Private function used to partition estimators between jobs."""
    # Compute the number of jobs
    n_jobs = min(effective_n_jobs(n_jobs), n_estimators)

    # Partition estimators between jobs
    n_estimators_per_job = np.full(n_jobs, n_estimators // n_jobs, dtype=int)
    n_estimators_per_job[: n_estimators % n_jobs] += 1
    starts = np.cumsum(n_estimators_per_job)

    return n_jobs, n_estimators_per_job.tolist(), [0] + starts.tolist()


class PLR_RandomForestRegressor(RandomForestRegressor):
    def __init__(
        self, n_estimators=100, random_state=0, n_jobs=-1, missing_label_strategy=None
    ):
        self.missing_label_strategy = missing_label_strategy
        super().__init__(
            n_estimators=n_estimators, random_state=random_state, n_jobs=n_jobs
        )

    def fit(self, X, y, sample_weight=None):
        if self.missing_label_strategy == "optimism":
            X, y = optimism_data_approach(X, y)

        elif self.missing_label_strategy == "pessimism":
            X, y = pessimism_data_approach(X, y)

        elif self.missing_label_strategy == "balanced":
            X, y = optimism_pessimism_data_approach(X, y)

        elif self.missing_label_strategy == "drop_individuals":
            non_missing_ranks = np.any(y >= 0, axis=1)
            X, y = X[non_missing_ranks], y[non_missing_ranks]

        super().fit(X, y, sample_weight)

    def predict(self, X):
        """
        Predict regression target for X.

        The predicted regression target of an input sample is computed as the
        mean predicted regression targets of the trees in the forest.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The predicted values.
        """
        check_is_fitted(self)
        # Check data
        X = self._validate_X_predict(X)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # avoid storing the output of every estimator by summing them here
        if self.n_outputs_ > 1:
            y_hat = np.zeros((X.shape[0], self.n_outputs_), dtype=np.float64)
        else:
            y_hat = np.zeros((X.shape[0]), dtype=np.float64)

        # Parallel loop
        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.verbose, require="sharedmem")(
            delayed(_accumulate_prediction)(e.predict, X, [y_hat], lock)
            for e in self.estimators_
        )

        y_hat /= len(self.estimators_)

        return transform_arrayToAPI(y_hat)


class PLR_MultiOutputRegressor(MultiOutputRegressor):
    def __str__(self):
        return "PLR_MultiOutputRegressor"

    def __init__(self, estimator, n_jobs=-1, missing_label_strategy=None):
        """

        Args:
            estimator: base estimator to be used for each target
            n_jobs: controls parallelism according to sklearn.MultiOutputRegressor
            missing_label_strategy: determinies how to handle possible missing labels in the training set.
                None == Dont do anything --> Assumes no missing labels in training set
                "drop_individuals" == For each base estimator on a target remove the individual with missing target labels
                "optimism" == Assume the missing target labels are the best possible
                "pessimism" == Assume the missing target labels are the worst possible
                "balanced" == Assume that both optimal and pessimistic one are equal likely. Thus add both to the data
        """
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.missing_label_strategy = missing_label_strategy
        super().__init__(estimator=estimator, n_jobs=n_jobs)

    def fit(self, X, y, sample_weight=None, **fit_params):
        # source: https://github.com/scikit-learn/scikit-learn/blob/f07e0138bfee41cd2c0a5d0251dc3fe03e6e1084/sklearn/multioutput.py#L267
        if not hasattr(self.estimator, "fit"):
            raise ValueError("The base estimator should implement a fit method")

        y = self._validate_data(X="no_validation", y=y, multi_output=True)

        if y.ndim == 1:
            raise ValueError(
                "y must have at least two dimensions for "
                "multi-output regression but has only one."
            )
        if _routing_enabled():
            if sample_weight is not None:
                fit_params["sample_weight"] = sample_weight
            routed_params = process_routing(
                self,
                "fit",
                **fit_params,
            )
        else:
            if sample_weight is not None and not has_fit_parameter(
                self.estimator, "sample_weight"
            ):
                raise ValueError(
                    "Underlying estimator does not support sample weights."
                )

            fit_params_validated = _check_method_params(X, params=fit_params)
            routed_params = Bunch(estimator=Bunch(fit=fit_params_validated))
            if sample_weight is not None:
                routed_params.estimator.fit["sample_weight"] = sample_weight

        if self.missing_label_strategy == "drop_individuals":
            self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                delayed(_fit_estimator)(
                    self.estimator,
                    X[y[:, i] >= 0],
                    y[:, i][y[:, i] >= 0],
                    **routed_params.estimator.fit,
                )
                for i in range(y.shape[1])
            )
        elif self.missing_label_strategy == "optimism":
            X_opt, y_opt = optimism_data_approach(X, y)
            self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                delayed(_fit_estimator)(
                    self.estimator, X_opt, y_opt[:, i], **routed_params.estimator.fit
                )
                for i in range(y_opt.shape[1])
            )
        elif self.missing_label_strategy == "pessimism":
            X_pes, y_pes = pessimism_data_approach(X, y)
            self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                delayed(_fit_estimator)(
                    self.estimator, X_pes, y_pes[:, i], **routed_params.estimator.fit
                )
                for i in range(y_pes.shape[1])
            )
        elif self.missing_label_strategy == "balanced":
            X_opt_pes, y_opt_pes = optimism_pessimism_data_approach(X, y)
            self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                delayed(_fit_estimator)(
                    self.estimator,
                    X_opt_pes,
                    y_opt_pes[:, i],
                    **routed_params.estimator.fit,
                )
                for i in range(y_opt_pes.shape[1])
            )
        else:
            self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                delayed(_fit_estimator)(
                    self.estimator, X, y[:, i], **routed_params.estimator.fit
                )
                for i in range(y.shape[1])
            )

        if hasattr(self.estimators_[0], "n_features_in_"):
            self.n_features_in_ = self.estimators_[0].n_features_in_
        if hasattr(self.estimators_[0], "feature_names_in_"):
            self.feature_names_in_ = self.estimators_[0].feature_names_in_

        return self

    def predict(self, X):
        """Predict multi-output variable using model for each target variable.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y : {array-like, sparse matrix} of shape (n_samples, n_outputs)
            Multi-output targets predicted across multiple predictors.
            Note: Separate models are generated for each predictor.
        """

        check_is_fitted(self)
        if not hasattr(self.estimators_[0], "predict"):
            raise ValueError("The base estimator should implement a predict method")

        y = Parallel(n_jobs=self.n_jobs)(
            delayed(e.predict)(X) for e in self.estimators_
        )

        return transform_arrayToAPI(np.asarray(y).T)


class PLR_RegressorChain(RegressorChain):
    def __str__(self):
        return "PLR_RegressorChain"

    def __init__(
        self, estimator, random_state, order=None, missing_label_strategy=None
    ):
        """
        Args:
        estimator: base estimator to be used for each target
        random_state: number of the random generator to have reproducible results
        order: The order in which the target is arranged.
            None == Will use the `find_chain_order
        ` function to determine the order
            [int] == list of target indices determining the order.
        missing_label_strategy: determinies how to handle possible missing labels in the training set.
            None == Dont do anything --> Assumes no missing labels in training set
            "drop_individuals" == For each base estimator on a target remove the individual with missing target labels
            "optimism" == Assume the missing target labels are the best possible
            "pessimism" == Assume the missing target labels are the worst possible
            "balanced" == Assume that both optimal and pessimistic one are equal likely. Thus add both to the data
        """
        self.estimator = estimator
        self.missing_label_strategy = missing_label_strategy
        super(RegressorChain, self).__init__(
            base_estimator=estimator, order=order, cv=None, random_state=random_state
        )

    def fit(self, X, Y, **fit_params):
        # update the order of chain to data best chain
        if self.missing_label_strategy == "optimism":
            X, Y = optimism_data_approach(X, Y)

        elif self.missing_label_strategy == "pessimism":
            X, Y = pessimism_data_approach(X, Y)

        elif self.missing_label_strategy == "balanced":
            X, Y = optimism_pessimism_data_approach(X, Y)
        elif self.missing_label_strategy == "drop_individuals":
            non_missing_ranks = np.any(Y >= 0, axis=1)
            X, Y = X[non_missing_ranks], Y[non_missing_ranks]

        self.order = find_chain_order(X, Y)

        super(RegressorChain, self).fit(X, Y)

    def predict(self, X):
        raw_output = super(RegressorChain, self).predict(X)

        return transform_arrayToAPI(raw_output)


class PLR_RegressorChainInterval(PLR_RegressorChain):

    def __str__(self):
        return "PLR_RegressorChainInterval"

    def __init__(
        self, estimator, random_state, order=None, missing_label_strategy=None
    ):
        """
        Args:
        estimator: base estimator to be used for each target. !!!! Must be an ensemble of any kind !!!!
        random_state: number of the random generator to have reproducible results
        order: The order in which the target is arranged.
            None == Will use the `find_chain_order
        ` function to determine the order \n
            [int] == list of target indices determining the order.
        missing_label_strategy: determinies how to handle possible missing labels in the training set.
            None == Dont do anything --> Assumes no missing labels in training set \n
            "drop_individuals" == For each base estimator on a target remove the individual with missing target labels \n
            "optimism" == Assume the missing target labels are the best possible \n
            "pessimism" == Assume the missing target labels are the worst possible \n
            "balanced" == Assume that both optimal and pessimistic one are equal likely. Thus add both to the data
        """
        super().__init__(
            estimator,
            order,
            random_state,
            missing_label_strategy=missing_label_strategy,
        )

    def predict_raw_interval(self, X):
        """
        original: sklearn. RegressorChain implementation
        edited: Santo Thies
        Predict on the data matrix X using interval overlapping technique.

                        Parameters
                        ----------
                        X : {array-like, sparse matrix} of shape (n_samples, n_features)
                            The input data.

                        Returns
                        -------
                        Y_pred : array-like of shape (n_samples, n_classes)
                            The predicted values.
        """
        a = perf_counter()
        check_is_fitted(self)
        X = self._validate_data(X, accept_sparse=True, reset=False)
        Y_pred_interval = np.zeros(shape=(X.shape[0], len(self.estimators_), 2))
        Y_pred_chain = np.zeros((X.shape[0], len(self.estimators_)))
        for chain_idx, estimator in enumerate(self.estimators_):
            previous_predictions = Y_pred_chain[:, :chain_idx]
            if sp.issparse(X):
                if chain_idx == 0:
                    X_aug = X
                else:
                    X_aug = sp.hstack((X, previous_predictions))
            else:
                X_aug = np.hstack((X, previous_predictions))

            raw_outputs = np.zeros(
                (X_aug.shape[0], len(estimator.estimators_)), dtype=np.float64
            )

            for i, e in enumerate(estimator.estimators_):
                raw_outputs[:, i] = e.predict(X_aug)

            Y_pred_chain[:, chain_idx] = np.mean(raw_outputs, axis=-1)

            # Build empirical intervals
            deviation_term = np.std(raw_outputs, axis=-1) / np.sqrt(
                raw_outputs.shape[1]
            )
            Y_pred_interval[:, chain_idx, 0] = (
                Y_pred_chain[:, chain_idx] - deviation_term
            )
            Y_pred_interval[:, chain_idx, 1] = (
                Y_pred_chain[:, chain_idx] + deviation_term
            )

        # until now the ordering of the output is sorted according to the self.order_
        # we now rearrange this ordering to correspond to original output order Y_0, Y_1, ..., Y_k
        inv_order = np.empty_like(self.order_)
        inv_order[self.order_] = np.arange(len(self.order_))
        Y_pred_int = Y_pred_interval[:, inv_order]
        return Y_pred_int

    def predict(self, X):
        """Predict on the data matrix X using Intervaloverlapping technique.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        Y_pred : array-like of shape (n_samples, n_classes)
            The predicted values.
        """
        Y_pred_int = self.predict_raw_interval(X)
        n_samples, n_classes, _ = Y_pred_int.shape
        consensus = np.zeros(shape=(n_samples, n_classes))  # N_samples, N_outputs
        for i in range(consensus.shape[0]):
            get_overlaps(Y_pred_int[i], n_classes, consensus[i])

        erg = transform_arrayToAPI(consensus)
        # print("Ranking Build Time: ", c-b)
        # print("Transformation Time: ", d-c)

        return erg
