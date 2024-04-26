import warnings

import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold
from sklearn.metrics import recall_score, balanced_accuracy_score

from utils import get_zero_nonzero_datasets, ratios_to_DI_cats

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

def k_fold_cross_val(model, data, num_folds = 5, class_metrics = [], reg_metrics = [], verbose = True):

    """
    A custom k-fold cross validation function that is compatible with a two-stage
    hurdle model. This will handle transformation of predictions into defaunation
    categories.

    Paramaters
    ----------
    model : sklearn model-like

    data : pandas.DataFrame

    num_folds : integer

    class_metrics : list

    reg_metrics : list

    verbose : boolean


    Returns
    -------
    metric_dict : dictionary

    """

    kfold = KFold(n_splits = num_folds, random_state = 1693, shuffle = True)

    classes = {0 : 'low', 1 : 'medium', 2 : 'high'}
    metric_dict = {}

    for m in class_metrics:
        metric_dict[m] = {classes[c] : [] for c in classes}
    for m in reg_metrics:
        metric_dict[m] = []

    for i, (train_idx, test_idx) in enumerate(kfold.split(data)):
        if verbose:
            print(f'Fold {i}:')
        train_data, test_data = data.loc[train_idx].copy(deep = True), data.loc[test_idx].copy(deep = True)

        with warnings.catch_warnings(action = 'ignore'):
            if verbose:
                print('  training model')
            model.fit(train_data)
            y_pred = model.predict(test_data)

        # Turn predictions into DI categories
        if verbose:
            print('  getting test metrics')
        test_ratios = data['ratio'].loc[test_idx].copy(deep = True)
        pred_ratios = y_pred.copy()
        pred_ratios[pred_ratios != 0] = np.exp(pred_ratios[pred_ratios != 0]) # don't need to back-transform predicted extirpations - not RRs!

        pred_DI_cats = ratios_to_DI_cats(pred_ratios)
        true_DI_cats = ratios_to_DI_cats(test_ratios)

        # Get TEST metrics for this train/test split
        pseudo_r2 = np.corrcoef(pred_ratios, test_ratios)[0, 1] ** 2
        if 'pseudo_r2' in reg_metrics:
            metric_dict['pseudo_r2'].append(pseudo_r2)

        for c in classes:
            #  binarizing the true/pred labels
            true = (true_DI_cats == c).astype(int)
            pred = (pred_DI_cats == c).astype(int)

            #  calculating metrics
            sens = recall_score(true, pred, pos_label = 1)
            spec = recall_score(true, pred, pos_label = 0)
            balanced_acc = balanced_accuracy_score(true, pred)

            if 'sensitivity' in class_metrics:
                metric_dict['sensitivity'][classes[c]].append(sens)
            if 'specificity' in class_metrics:
                metric_dict['specificity'][classes[c]].append(spec)
            if 'balanced_accuracy' in class_metrics:
                metric_dict['balanced_accuracy'][classes[c]].append(balanced_acc)

    return metric_dict
