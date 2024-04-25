import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from utils import get_zero_nonzero_datasets

class PymerModelWrapper:

    """
    A wrapper class to make pymer models behave like sklearn models.
    """

    def __init__(self, pymer_model, formula, family, control_str = ''):
        self.model_type = pymer_model
        self.formula = formula
        self.family = family
        self.control_str = control_str

    def fit(self, X, y):
        data = self.combine_X_y(X, y)

        self.model = self.model_type(data = data, formula = self.formula, family = self.family)
        self.model.fit(REML = True, control = self.control_str, summarize = False)

    def predict(self, X):
        preds = self.model.predict(X, use_rfx = True, skip_data_checks = True, verify_predictions = False)
        preds = np.asarray(preds)

        if self.family == 'binomial':
            preds = (preds >= 0.5).astype(int) # just using the canonical threshold...

        return preds

    def predict_proba(self, X):
        assert self.family == 'binomial', f'Cannot make probability predictions with a {self.family} model.'

        preds = self.model.predict(X, use_rfx = True, skip_data_checks = True, verify_predictions = False)
        preds = np.asarray(preds)

        #  formatting the predictions like in sklearn
        preds_prob = np.ones((preds.shape[0], 2))
        preds_prob[ : , 1] = preds.copy()
        preds_prob[ : , 0] = 1 - preds

        return preds_prob

    def combine_X_y(self, X_data, y_data):
        dataset = X_data.copy(deep = True)
        response_name = 'local_extirpation' if self.family == 'binomial' else 'RR'
        dataset[response_name] = y_data # adding the response to the dataframe as expected by pymer

        return dataset

class HurdleModelEstimator(RegressorMixin, BaseEstimator):

    """
    A wrapper class to bind together the zero and nonzero model components
    of a two-stage hurdle model.
    """

    def __init__(self, zero_model, nonzero_model, prob_thresh = 0.5, data_args = {}):
        self.zero_model = zero_model
        self.nonzero_model = nonzero_model
        self.prob_thresh = prob_thresh
        self.data_args = data_args

    def fit(self, pp_data):
        X_zero, y_zero, X_nonzero, y_nonzero = get_zero_nonzero_datasets(pp_data, pred = False, **self.data_args)

        self.nonzero_model.fit(X_nonzero, y_nonzero)
        self.zero_model.fit(X_zero, y_zero)

        return self

    def predict(self, pp_data):
        X_zero, X_nonzero = get_zero_nonzero_datasets(pp_data, pred = True)

        y_pred_zero = self.zero_model.predict_proba(X_zero)[ : , 1] >= self.prob_thresh # hard classification, 1 = local extirpation
        y_pred_nonzero = self.nonzero_model.predict(X_nonzero)

        y_pred = (~y_pred_zero).astype(int) * y_pred_nonzero # if y_pred_zero >= prob_thresh, this is a local extirpation so our prediction should be zero

        return y_pred
