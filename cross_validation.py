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
        a model that behaves like an sklearn model (i.e., has fit and predict
        functions)
    data : pandas.DataFrame
        a dataframe holding the raw predictors and the response variable (i.e.,
        not yet preprocessed)
    block_type : string
        the type of cross-validation blocking to perform, one of 'spatial' or 'group,'
        defaults to random splitting
    num_folds : integer
        the number of folds to use for k-fold cross-validation
    group_col : string
        the column in the dataframe to use for group membership in group blocking
    spatial_spacing : integer
        the size of grid cells in degrees for spatial blocking
    fit_args : dictionary
        keyword arguments to pass to the model during fitting
    pp_args : dictionary
        keyword arguments to pass to the preprocessing function
    class_metrics : dictionary
        classification metrics to use (either 'overall' or 'per-class'), each entry
        should be a dictionary with 'function' and 'kwargs' entries
    reg_metrics : dictionary
        regression metrics to use, each entry should be a dictionary with
        'function' and 'kwargs' entries
    verbose : boolean
        should we print out progress indicators during cross-validation?
    random_state : integer
        the random state used in all interior random operations
    sklearn_submodels : boolean
        are we using a hurdle model with sklearn sub-models?
    back_transform : boolean
        should we back-transform from response ratios to abundance ratios during
        prediction?
    direct : string
        the type of direct model being used, either 'classification' or 'regression'
    tune_hurdle_thresh : boolean
        should we tune the probability threshold for the zero model component of
        the hurdle model?

    Returns
    -------
    metric_dict : dictionary
        the obtained test metrics for each fold, each passed in metric gets one
        entry
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
    if block_type == 'random':
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
            print(f'Using spatial blocking with spacing {spatial_spacing} degrees')
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

    # Getting data splits for cross-validation
    lat, lon = ('Y', 'X') if pp_args['dataset'] in ['mammals', 'both'] else ('Latitude', 'Longitude')
    coords = data[[lon, lat]].values

    #  some logic for data splitting based on the dataset
    if pp_args['dataset'] == 'both':
        mammal_mask = data['Class'] == 'Mammalia'
        split_mammals = kfold.split(coords[mammal_mask], groups = groups[mammal_mask] if block_type == 'group' else groups)

        bird_mask = data['Class'] == 'Aves'
        split_birds = kfold.split(coords[bird_mask], groups = groups[bird_mask] if block_type == 'group' else groups)

        splits = zip(split_mammals, split_birds)
    else:
        splits = kfold.split(coords, groups = groups)

    # Data structure for saving predictions
    all_preds = {'index' : [], 'fold' : [], 'actual' : [], 'predicted' : []}

    # Running the k-fold cross-validation
    for i, split in enumerate(splits):
        if verbose:
            print(f'Fold {i}:')

        #  preprocessing data - a bit different if using both datasets, so that we
        #   are sure to get the same splits as the single-dataset case!
        if pp_args['dataset'] == 'both':
            (mammal_train_idx, mammal_test_idx), (bird_train_idx, bird_test_idx) = split
            bird_train_idx, bird_test_idx = bird_train_idx + mammal_mask.sum(), bird_test_idx + mammal_mask.sum()

            train_test_idxs = {'train' : list(mammal_train_idx) + list(bird_train_idx),
                               'test' : list(mammal_test_idx) + list(bird_test_idx)}
        else:
            train_idx, test_idx = split
            train_test_idxs = {'train' : train_idx, 'test' : test_idx}

        pp_data = preprocess_data(data, standardize = True, train_test_idxs = train_test_idxs, **pp_args)

        # Fitting/predicting differently for direct classification/regression vs. hurdle models
        if verbose:
            print('  training model')
        if direct is None:
            train_data, test_data = pp_data.iloc[train_test_idxs['train']].copy(deep = True), pp_data.iloc[train_test_idxs['test']].copy(deep = True)

            #  clone the model to ensure it fits from scratch... Pymer submodels do this through the wrapper class
            #   at fit time and AutoML instances do this when "keep_search_state" is False
            if sklearn_submodels:
                model = clone(model)

            #  train the model
            with warnings.catch_warnings(action = 'ignore'):
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

            resp_col = 'ratio' if pp_args['dataset'] in ['mammals', 'both'] else 'RR'
            if pp_args['dataset'] == 'mammals_extended':
                resp_col = 'Ratio'

            y_test = test_data[resp_col].copy(deep = True)

            #  back-transforming to go from RRs --> ratios
            if back_transform:
                y_pred[y_pred != 0] = np.exp(y_pred[y_pred != 0])
        else:
            assert direct in ['classification', 'regression'], 'The "direct" argument must either be "classification" or "regression."'
            assert pp_args['dataset'] != 'both', 'Training/testing on both datasets is only supported for hurdle models.'

            #  getting the data split
            X_train, y_train, X_test, y_test = direct_train_test(pp_data, task = direct, already_pp = True,
                                                                 train_test_idxs = train_test_idxs,
                                                                 dataset = pp_args['dataset'])

            #  training the model + perform model search
            model.fit(X_train, y_train, **fit_args)

            #  predicting on the test set
            y_pred = model.predict(X_test)

        # Discretize ratios for regression models to get classification metrics
        #  - case where our predictions are already in the form of DI categories
        if direct == 'classification':
            true_DI_cats = y_test
            pred_DI_cats = y_pred
        #  - case where our predictions are in the form of ratios
        elif len(class_metrics) != 0:
            true_DI_cats = ratios_to_DI_cats(y_test)
            pred_DI_cats = ratios_to_DI_cats(y_pred)

        # Save predictions and true values for this test set
        all_preds['index'].extend(train_test_idxs['test'])
        all_preds['fold'].extend([i for k in range(len(y_test))])
        all_preds['actual'].extend(y_test)
        all_preds['predicted'].extend(y_pred)

        # Get metrics
        if pp_args['dataset'] != 'both':
            if verbose:
                print('  getting test metrics')

            #  get regression test metrics for this train/test split
            for metric in reg_metrics.keys():
                kws = reg_metrics[metric]['kwargs']
                metric_dict[metric].append(reg_metrics[metric]['function'](y_test, y_pred, **kws))

            #  get per-class classification metrics
            for c in classes:
                #  binarizing the true/pred labels
                true = (true_DI_cats == c).astype(int)
                pred = (pred_DI_cats == c).astype(int)

                for metric in class_metrics['per_class'].keys():
                    kws = class_metrics['per_class'][metric]['kwargs']
                    metric_dict[metric][classes[c]].append(class_metrics['per_class'][metric]['function'](true, pred, **kws))

            #  get overall classification metrics
            for metric in class_metrics['overall'].keys():
                kws = class_metrics['overall'][metric]['kwargs']
                metric_dict[metric].append(class_metrics['overall'][metric]['function'](true_DI_cats, pred_DI_cats, **kws))

    # Formatting prediction saves and adding to the metric dictionary
    all_preds = pd.DataFrame(all_preds)
    all_preds = all_preds.set_index('index').sort_index()
    if pp_args['dataset'] == 'both':
        all_preds['dataset'] = (['mammals'] * len(data[mammal_mask])) + (['birds'] * len(data[bird_mask]))

    metric_dict['raw_preds'] = all_preds

    return metric_dict

def format_cv_results(metric_dict_sub, class_metrics, reg_metrics, result_type = 'per_class'):

    """
    A helper function to handle the re-formatting of different subsets of the metric dictionary.

    Parameters
    ----------
    metrics_dict_sub : dictionary
        a subset of the metric dictionary post-cross-validation
    class_metrics : dictionary
        the classification metrics used (see 'run_cross_val' above)
    reg_metrics : dictionary
        the regression metrics used (see 'run_cross_val' above)
    results_type : string
        the type of results contained in 'metrics_dict_sub,' one of 'per_class'
        (classification), 'overall' (classification), or 'regression'

    Returns
    -------
    results : pandas.DataFrame
        a formatted dataframe with the same information as 'metrics_dict_sub'
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

def save_cv_results(metrics_dict, model_name, save_fp, cross_val_params, class_metrics = None,
                    reg_metrics = None, vals_to_save = None, dataset = 'mammals'):

    """
    A helper function to wrap the re-formatting and aggregation of the results dictionary
    obtained via cross-validation. Results are also saved as a CSV file.

    Parameters
    ----------
    metrics_dict : dictionary
        the metric dictionary recieved from cross-validation
    model_name : string
        the name of the model used
    save_fp : string
        a folder for saving the metric results and the raw predictions
    cross_val_params : dictionary
        the parameters used for cross-validation, should have entries for 'num_folds,'
        'block_type,' 'spatial_spacing,' and 'group_col'
    class_metrics : dictionary
        the classification metrics used (see 'run_cross_val' above)
    reg_metrics : dictionary
        the regression metrics used (see 'run_cross_val' above)
    vals_to_save : list
        a list of the values to save, can include 'metrics' and 'raw'
    dataset : string
        the dataset being used, either 'mammals' (Benitez-Lopez et al., 2019) or
        'birds' (Ferreiro-Arias et al., 2024)

    Returns
    -------
    final_results : pandas.DataFrame
        the final results aggregated (mean and standard deviation across folds)
        in a dataframe format
    """

    block_name = cross_val_params['block_type'] if cross_val_params['block_type'] is not None else 'random'

    if dataset != 'both':
        # Setting mutable defaults
        assert (class_metrics is not None) or (reg_metrics is not None), 'Make sure one of "class_metrics" or "reg_metrics" is not None.'
        assert len(metrics_dict) > 0, 'The inputted "metrics_dict" has no entries.'

        if class_metrics is None:
            class_metrics = {'per_class' : {}, 'overall' : {}}
        if reg_metrics is None:
            reg_metrics = {}
        if vals_to_save is None:
            vals_to_save = ['metrics', 'raw']

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
        final_results['dataset'] = dataset

        #  adding info about the cross-val setup
        final_results['num_folds'] = cross_val_params['num_folds']
        final_results['block_type'] = block_name
        final_results['spatial_spacing'] = cross_val_params['spatial_spacing'] if cross_val_params['block_type'] == 'spatial' else np.nan
        final_results['group_col'] = cross_val_params['group_col'] if cross_val_params['block_type'] == 'group' else np.nan

        # Saving to the inputted file
        if 'metrics' in vals_to_save:
            save_filename = os.path.join(save_fp, 'cross_val_results.csv')
            if os.path.isfile(save_filename):
                existing_results = pd.read_csv(save_filename)
                all_results = pd.concat((existing_results, final_results))
                all_results.to_csv(save_filename, index = False)
            else:
                final_results.to_csv(save_filename, index = False)

    # Saving the raw predictions
    if 'raw' in vals_to_save:
        raw_preds = metrics_dict['raw_preds']

        save_filename = f'{model_name}_{dataset}_{cross_val_params["num_folds"]}-fold_{block_name}-blocking'
        if cross_val_params['block_type'] == 'spatial':
            save_filename += f'_{cross_val_params["spatial_spacing"]}-degree'
        elif cross_val_params['block_type'] == 'group':
            save_filename += f'_{cross_val_params["group_col"].lower()}'
        save_filename += '.csv'

        preds_fp = os.path.join(save_fp, 'raw_predictions', save_filename)
        raw_preds.to_csv(preds_fp)

    if dataset != 'both':
        return final_results
