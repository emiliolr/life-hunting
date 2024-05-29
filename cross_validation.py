import os
import warnings
from datetime import datetime

import pandas as pd
import numpy as np

from sklearn.base import clone
from sklearn.model_selection import GroupKFold, KFold

from verde import BlockKFold

from utils import direct_train_test, preprocess_data, get_zero_nonzero_datasets, \
                  test_thresholds, ratios_to_DI_cats
from model_utils import HurdleModelEstimator

def run_cross_val(model, data, block_type = None, num_folds = 5, group_col = None, spatial_spacing = 5,
                  fit_args = None, pp_args = None, class_metrics = None, reg_metrics = None, verbose = True,
                  random_state = 1693, sklearn_submodels = False, back_transform = True, direct = None,
                  tune_hurdle_thresh = False):

    """
    A function to run k-fold cross-validation over a given dataset and with a given model. Multiple
    types of blocking are supported, including spatial- and group-blocking of folds. Multiple model types
    are supported as well, i.e., two-stage hurdle and direct regression/classification models.

    Parameters
    ----------
    model : sklearn-like
    data : pandas.DataFrame
    block_type : string
    num_folds : integer
    group_col : string
    spatial_spacing : integer
    fit_args : dictionary
    pp_args : dictionary
    class_metrics : dictionary
    reg_metrics : dictionary
    verbose : boolean
    random_state : integer
    sklearn_submodels : boolean
    back_transform : boolean
    direct : string
    tune_hurdle_thresh : boolean

    Returns
    -------
    metric_dict : dictionary
    """

    # Setting mutable defaults
    assert (class_metrics is not None) or (reg_metrics is not None), 'Please provide at least one classification or regression metric.'

    if class_metrics is None:
        class_metrics = {'per_class' : {}, 'overall' : {}}
    if reg_metrics is None:
        reg_metrics = {}
    if fit_args is None and direct is not None:
        fit_args = {}
    if pp_args is None:
        pp_args = {}

    # Establishing k-fold parameters
    if block_type is None:
        if verbose:
            print('Using standard cross-validation')
        groups = None
        kfold = KFold(n_splits = num_folds, random_state = random_state, shuffle = True)
    elif block_type == 'group':
        assert group_col is not None, 'If using group-based blocking, a group column must be specified using "group_col."'

        if verbose:
            print(f'Using group blocking on column {group_col}')
        groups = data[group_col].values
        kfold = GroupKFold(n_splits = num_folds)
    elif block_type == 'spatial':
        if verbose:
            print(f'Using spatial blocking on with spacing {spatial_spacing} degrees')
        groups = None
        kfold = BlockKFold(spacing = spatial_spacing, n_splits = num_folds, shuffle = True, random_state = random_state)

    # Data structures for saving results
    classes = {0 : 'low', 1 : 'medium', 2 : 'high'}
    metric_dict = {}

    for m in class_metrics['per_class']:
        metric_dict[m] = {classes[c] : [] for c in classes}
    for m in class_metrics['overall']:
        metric_dict[m] = []
    for m in reg_metrics:
        metric_dict[m] = []

    # Running the k-fold cross-validation
    coords = data[['X', 'Y']].values
    for i, (train_idx, test_idx) in enumerate(kfold.split(coords, groups = groups)):
        if verbose:
            print(f'Fold {i}:')

        train_test_idxs = {'train' : train_idx, 'test' : test_idx}
        pp_data = preprocess_data(data, standardize = True, train_test_idxs = train_test_idxs, **pp_args)

        # Fitting/predicting differently for direct classification/regression vs. hurdle models
        if direct is None:
            train_data, test_data = pp_data.iloc[train_idx].copy(deep = True), pp_data.iloc[test_idx].copy(deep = True)

            #  clone the model to ensure it fits from scratch... Pymer submodels do this through the wrapper class
            #   at fit time and AutoML instances do this when "keep_search_state" is False
            if sklearn_submodels:
                model = clone(model)

            #  train the model
            with warnings.catch_warnings(action = 'ignore'):
                if verbose:
                    print('  training model')
                model.fit(train_data, fit_args)

            #  optionally tuning the probability threshold for the zero component of the hurdle model
            if tune_hurdle_thresh:
                assert isinstance(model, HurdleModelEstimator), 'Threshold tuning only applies to two-stage hurdle models.'

                X_zero, y_zero, _, _ = get_zero_nonzero_datasets(train_data, extirp_pos = model.extirp_pos,
                                                                 pred = False, **model.data_args)
                y_pred = model.zero_model.predict_proba(X_zero)[ : , 1]

                opt_thresh, _ = test_thresholds(y_pred, y_zero)
                model.prob_thresh = round(opt_thresh, 3)

                if verbose:
                    print(f'  optimal threshold was found to be {model.prob_thresh}')

            #  predicting on the test set
            y_pred = model.predict(test_data)
            y_test = test_data['ratio'].copy(deep = True)

            #  back-transforming to go from RRs --> ratios
            if back_transform:
                y_pred[y_pred != 0] = np.exp(y_pred[y_pred != 0])
        else:
            assert direct in ['classification', 'regression'], 'The "direct" argument must either be "classification" or "regression."'

            #  getting the data split
            X_train, y_train, X_test, y_test = direct_train_test(pp_data, task = direct, already_pp = True,
                                                                 train_test_idxs = train_test_idxs)

            #  training the model + perform model search
            model.fit(X_train = X_train, y_train = y_train, **fit_args)

            #  predicting on the test set
            y_pred = model.predict(X_test)

        # Get predictions and targets
        if verbose:
            print('  getting test metrics')

        # Discretize ratios for regression models to get classification metrics
        #  - case where our predictions are already in the form of DI categories
        if direct == 'classification':
            true_DI_cats = y_test
            pred_DI_cats = y_pred
        #  - case where our predictions are in the form of ratios
        elif len(class_metrics) != 0:
            true_DI_cats = ratios_to_DI_cats(y_test)
            pred_DI_cats = ratios_to_DI_cats(y_pred)

        # Get regression test metrics for this train/test split
        for metric in reg_metrics.keys():
            kws = reg_metrics[metric]['kwargs']
            metric_dict[metric].append(reg_metrics[metric]['function'](y_test, y_pred, **kws))

        # Get per-class classification metrics
        for c in classes:
            #  binarizing the true/pred labels
            true = (true_DI_cats == c).astype(int)
            pred = (pred_DI_cats == c).astype(int)

            for metric in class_metrics['per_class'].keys():
                kws = class_metrics['per_class'][metric]['kwargs']
                metric_dict[metric][classes[c]].append(class_metrics['per_class'][metric]['function'](true, pred, **kws))

        # Get overall classification metrics
        for metric in class_metrics['overall'].keys():
            kws = class_metrics['overall'][metric]['kwargs']
            metric_dict[metric].append(class_metrics['overall'][metric]['function'](true_DI_cats, pred_DI_cats, **kws))

    return metric_dict

def format_cv_results(metric_dict_sub, class_metrics, reg_metrics, result_type = 'per_class'):

    """
    A helper function to handle the re-formatting of different subsets of the metric dictionary.

    Parameters
    ----------
    metrics_dict_sub : dictionary
    class_metrics : dictionary
    reg_metrics : dictionary
    results_type : string

    Returns
    -------
    results : pandas.DataFrame
    """

    # Re-structuring per-class classification metrics
    if result_type == 'per_class':
        metrics = pd.DataFrame(metric_dict_sub)
        metrics = pd.concat([metrics[m].explode() for m in class_metrics['per_class']], axis = 1).reset_index()
        metrics = metrics.rename(columns = {'index' : 'DI_category'})
        metrics = metrics.melt(id_vars = ['DI_category'], value_vars = class_metrics['per_class'], var_name = 'metric')

        results = metrics.groupby(['DI_category', 'metric']).mean()
        results = results.rename(columns = {'value' : 'mean'})
        results = pd.concat((results, metrics.groupby(['DI_category', 'metric']).std()), axis = 1)
        results = results.rename(columns = {'value' : 'std'})

    # Re-structuring overall classification metrics
    elif result_type == 'overall':
        metrics = pd.DataFrame(metric_dict_sub)
        metrics = metrics.melt(id_vars = [], value_vars = class_metrics['overall'], var_name = 'metric')

        results = metrics.groupby('metric').describe()['value'][['mean', 'std']]

    # Re-structuring regression metrics
    elif result_type == 'regression':
        if 'mean_absolute_error_0-1' in metric_dict_sub:
            metric_dict_sub['mean_absolute_error_0-1'] = [mae for mae, _ in metric_dict_sub['mean_absolute_error_0-1']]

        metrics = pd.DataFrame(metric_dict_sub)
        metrics = metrics.melt(id_vars = [], value_vars = reg_metrics, var_name = 'metric')

        results = metrics.groupby('metric').describe()['value'][['mean', 'std']]

    # Incorrect input value for result_type arg
    else:
        raise ValueError('The "result_type" argument must be one of "per_class," "overall," or "regression."')

    return results

def save_cv_results(metrics_dict, model_name, fp, class_metrics = None, reg_metrics = None):

    """
    A helper function to wrap the re-formatting of the results dictionary and saving the resulting
    dataframe as a CSV file after cross-validation.

    Parameters
    ----------
    metrics_dict : dictionary
    model_name : string
    fp : string
    class_metrics : dictionary
    reg_metrics : dictionary

    Returns
    -------
    final_results : pandas.DataFrame
    """

    # Setting mutable defaults
    assert (class_metrics is not None) or (reg_metrics is not None), 'Make sure one of "class_metrics" or "reg_metrics" is non-empty.'
    assert len(metrics_dict) > 0, 'The inputted "metrics_dict" has no entries.'

    if class_metrics is None:
        class_metrics = {'per_class' : {}, 'overall' : {}}
    if reg_metrics is None:
        reg_metrics = {}

    # Cleaning per-class classification metrics
    per_class_dict = {m : metrics_dict[m] for m in class_metrics['per_class']}
    if len(per_class_dict) > 0:
        per_class_results = format_cv_results(per_class_dict, class_metrics, reg_metrics, result_type = 'per_class')
    else:
        per_class_results = None

    # Cleaning overall classification metrics
    overall_dict = {m : metrics_dict[m] for m in class_metrics['overall']}
    if len(overall_dict) > 0:
        overall_results = format_cv_results(overall_dict, class_metrics, reg_metrics, result_type = 'overall')
    else:
        overall_results = None

    # Cleaning regression metrics
    reg_dict = {m : metrics_dict[m] for m in reg_metrics}
    if len(reg_dict) > 0:
        reg_results = format_cv_results(reg_dict, class_metrics, reg_metrics, result_type = 'regression')
    else:
        reg_results = None

    # Merging the cleaned results dataframes
    columns = ['DI_category', 'metric', 'mean', 'std']
    final_results = pd.DataFrame({c : [np.nan] for c in columns}) # empty dataframe to add on to

    if per_class_results is not None:
        final_results = pd.concat((final_results, per_class_results.reset_index()))
    if overall_results is not None:
        final_results = pd.concat((final_results, overall_results.reset_index()))
    if reg_results is not None:
        final_results = pd.concat((final_results, reg_results.reset_index()))

    final_results = final_results.reset_index(drop = True).dropna(axis = 0, how = 'all')

    #  adding on some last few bits of information
    final_results = final_results.rename(columns = {'std' : 'standard_deviation'})
    final_results['DI_category'] = final_results['DI_category'].fillna('overall')
    final_results['date'] = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    final_results['model_name'] = model_name

    # Saving to the inputted file
    if os.path.isfile(fp):
        existing_results = pd.read_csv(fp)
        all_results = pd.concat((existing_results, final_results))
        all_results.to_csv(fp, index = False)
    else:
        final_results.to_csv(fp, index = False)

    return final_results
