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

    def __init__(self, pymer_model, formula, family, control_str = '', use_rfx = True):
        self.model_type = pymer_model
        self.formula = formula
        self.family = family
        self.control_str = control_str
        self.use_rfx = use_rfx

    def fit(self, X, y):
        data = self.combine_X_y(X, y)

        self.model = self.model_type(data = data, formula = self.formula, family = self.family)
        self.model.fit(REML = True, control = self.control_str, summarize = False)

    def predict(self, X):
        preds = self.model.predict(X, use_rfx = self.use_rfx, skip_data_checks = True, verify_predictions = False)
        preds = np.asarray(preds)

        if self.family == 'binomial':
            preds = (preds >= 0.5).astype(int) # just using the canonical threshold...

        return preds

    def predict_proba(self, X):
        assert self.family == 'binomial', f'Cannot make probability predictions with a {self.family} model.'

        preds = self.model.predict(X, use_rfx = self.use_rfx, skip_data_checks = True, verify_predictions = False)
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

    def __init__(self, zero_model, nonzero_model, prob_thresh = 0.5, extirp_pos = True,
                 verbose = False, data_args = None):
        self.data_args = {} if data_args is None else data_args
        
        self.zero_model = zero_model
        self.nonzero_model = nonzero_model

        self.prob_thresh = prob_thresh
        self.extirp_pos = extirp_pos
        self.verbose = verbose

    def fit(self, pp_data, fit_args = None):
        if fit_args is None:
            fit_args = {'zero' : {}, 'nonzero' : {}}

        X_zero, y_zero, X_nonzero, y_nonzero = get_zero_nonzero_datasets(pp_data,
                                                                         pred = False,
                                                                         extirp_pos = self.extirp_pos,
                                                                         **self.data_args)

        if self.verbose:
            print('  fitting the nonzero model...')
        self.nonzero_model.fit(X_nonzero, y_nonzero, **fit_args['nonzero'])

        if self.verbose:
            print('  fitting the zero model...')
        self.zero_model.fit(X_zero, y_zero, **fit_args['zero'])

    def predict(self, pp_data, return_constit_preds = False):
        X_zero, X_nonzero = get_zero_nonzero_datasets(pp_data, extirp_pos = self.extirp_pos, pred = True, **self.data_args)

        y_pred_zero = self.zero_model.predict_proba(X_zero)[ : , 1] >= self.prob_thresh # hard classification
        y_pred_nonzero = self.nonzero_model.predict(X_nonzero)

        y_pred = (~y_pred_zero).astype(int) * y_pred_nonzero if self.extirp_pos else y_pred_zero.astype(int) * y_pred_nonzero

        if return_constit_preds:
            return y_pred, y_pred_zero.astype(int), y_pred_nonzero
        return y_pred

class TwoStageNovelModel(RegressorMixin, BaseEstimator):

    def __init__(self, classifier, regressor_decrease, regressor_increase, classes = None):
        self.classifier = classifier
        self.regressor_decline = regressor_decrease
        self.regressor_increase = regressor_increase

        self.classes = {'extirpated' : 0, 'decrease' : 1, 'no_effect' : 2,
                        'increase' : 3}

    def fit(self):
        pass

    def predict(self):
        pass

    def predict_proba(self):
        pass

def k_fold_cross_val(model, data, num_folds = 5, class_metrics = None, reg_metrics = None,
                     verbose = True, dataset = 'mammals'):

    """
    A standard random-split k-fold cross validation function that is compatible with
    a two-stage hurdle model. This will handle transformation of predictions into
    defaunation categories.

    Paramaters
    ----------
    model : sklearn model-like
        a model that behaves like an sklearn model (i.e., has fit and predict
        functions)
    data : pandas.DataFrame
        a dataframe holding the preprocessed predictors and the response variable
    num_folds : integer
        the number of folds to use for k-fold cross-validation
    class_metrics : list
        the classification metrics to record (will consider each class the positive
        class in turn)
    reg_metrics : list
        the regression metrics to record
    verbose : boolean
        should we print incremental updates during function execution?
    dataset : string
        the dataset being used, either 'mammals' (Benitez-Lopez et al., 2019) or
        'birds' (Ferreiro-Arias et al., 2024)

    Returns
    -------
    metric_dict : dictionary
        a dictionary that holds all obtained metrics for each cross-validation split
    """

    # Setting mutable defaults
    assert (class_metrics is not None) or (reg_metrics is not None), 'Please provide at least one classification or regression metric.'

    if class_metrics is None:
        class_metrics = []
    if reg_metrics is None:
        reg_metrics = []

    # Establishing k-fold parameters and structures to save results
    kfold = KFold(n_splits = num_folds, random_state = 1693, shuffle = True)

    classes = {0 : 'low', 1 : 'medium', 2 : 'high'}
    metric_dict = {}

    for m in class_metrics:
        metric_dict[m] = {classes[c] : [] for c in classes}
    for m in reg_metrics:
        metric_dict[m] = []

    # Running the k-fold cross-validation
    for i, (train_idx, test_idx) in enumerate(kfold.split(data)):
        if verbose:
            print(f'Fold {i}:')
        train_data, test_data = data.iloc[train_idx].copy(deep = True), data.iloc[test_idx].copy(deep = True)

        with warnings.catch_warnings(action = 'ignore'):
            if verbose:
                print('  training model')
            model.fit(train_data)
            y_pred = model.predict(test_data)

        # Turn predictions into DI categories
        if verbose:
            print('  getting test metrics')
        resp_col = 'ratio' if dataset == 'mammals' else 'RR'
        test_ratios = data[resp_col].iloc[test_idx].copy(deep = True)
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
