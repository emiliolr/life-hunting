import sys
import os
import argparse
from datetime import datetime

sys.path.append('..')

import pandas as pd
import numpy as np

from custom_metrics import *

def get_run_info_from_fname(filename):

    """
    Extract all relevant info for a model run from the model run's filename.
    """

    # Some initial checking to remove runs which can't be parsed in the same way
    if ('SPECIAL' in filename) or ('CLIP' in filename) or (filename == '.DS_Store'):
        return None

    # Parsing the file name
    bits = filename.split('_')

    model = bits[0]
    assert bits[0] in ['pymer', 'FLAML', 'xgboost', 'rf', 'dummy'], f'Model "{bits[0]}" is not currently implemented in this function'
    
    if model == 'pymer':
        if ('rebalance' in filename) and ('tune' in filename):
            model_name = bits[0 : 6]
            i = 6
        elif ('rebalance' in filename) or ('tune' in filename):
            model_name = bits[0 : 5]
            i = 5
        else:
            model_name = bits[0 : 4]
            i = 4
    elif model in ['FLAML', 'xgboost', 'rf']:
        if ('rebalance' in filename) or ('tune' in filename) or ('ensemble' in filename):
            model_name = bits[0 : 4]
            i = 4
        else:
            model_name = bits[0 : 3]
            i = 3
    elif model == 'dummy':
        model_name = bits[0 : 3]
        i = 3
    model_name = '_'.join(model_name)

    #  ignoring direct regression & classification models
    if (model_name.startswith('FLAML_classification')) or (model_name.startswith('FLAML_regression')):
        return None

    if bits[i + 1] in ['extended', 'recreated']:
        dataset = bits[i : i + 2]
        dataset = '_'.join(dataset)
        i += 2
    else:
        dataset = bits[i]
        i += 1
    
    num_folds = bits[i]
    num_folds = int(num_folds.removesuffix('-fold'))
    i += 1

    block_type = bits[i]
    block_type = block_type.removesuffix('.csv').removesuffix('-blocking')
    i += 1

    spatial_spacing = None
    group_col = None
    
    if block_type == 'spatial':
        spatial_spacing = bits[i]
        spatial_spacing = int(spatial_spacing.removesuffix('-degree.csv'))
    elif block_type == 'group':
        group_col = bits[i]
        group_col = group_col.removesuffix('.csv')

    return_dict = {'model_name' : model_name, 
                   'dataset' : dataset, 
                   'num_folds' : num_folds, 
                   'block_type' : block_type,
                   'spatial_spacing' : spatial_spacing,
                   'group_col' : group_col,
                   'filename' : filename}

    return return_dict

def get_metric_CV(raw_preds, metric, submodel = None, **kwargs):

    """
    Utility function to apply a given metric to each fold and then get the average and 
    standard deviation over folds.
    """

    if submodel == 'nonzero':
        raw_preds = raw_preds[raw_preds['actual'] != 0].copy(deep = True) # tossing zero entries when just evaluating the nonzero model
    raw_preds = raw_preds.groupby('fold')

    if submodel is None:
        metric_by_group = raw_preds.apply(lambda x: metric(x['actual'], x['predicted'], **kwargs), 
                                          include_groups = False)
    elif submodel == 'zero':
        metric_by_group = raw_preds.apply(lambda x: metric(x['actual_zero'], x['predicted_zero'], **kwargs), 
                                          include_groups = False)
    elif submodel == 'nonzero':
        metric_by_group = raw_preds.apply(lambda x: metric(x['actual'], x['predicted_nonzero'], **kwargs), 
                                          include_groups = False)
    
    metric_mean = metric_by_group.mean()
    metric_std = metric_by_group.std()

    return metric_mean, metric_std

def metrics_to_use():

    """
    Defining the list of metrics to use, their arguments, and which types of models they should 
    be applied to.
    """

    metrics = []

    # CLASSIFICATION:
    #  balanced accuracy on intuitive DI categories
    ba_DI_cats = {'function' : balanced_accuracy_DI_cats, 
                  'kwargs' : {'neighborhood' : 0.05},
                  'name' : 'balanced_accuracy_DI-%s',
                  'kwarg_name_fill' : 'neighborhood',
                  'valid' : 'all'}
    metrics.append(ba_DI_cats)

    # REGRESSION:
    #  mean absolute error over full range
    mae_inf = {'function' : mean_absolute_error, 
               'kwargs' : {},
               'name' : 'mean_absolute_error-inf',
               'valid' : 'all'}
    metrics.append(mae_inf)

    #  mean absolute error in the 0-1 range
    mae_01 = {'function' : mean_absolute_error_range, 
              'kwargs' : {'lower_bound' : 0, 'upper_bound' : 1, 'return_pct_kept' : False},
              'name' : 'mean_absolute_error-1',
              'valid' : 'all'}
    metrics.append(mae_01)

    #  median absolute error over full range
    med_ae_inf = {'function' : median_absolute_error, 
                  'kwargs' : {},
                  'name' : 'median_absolute_error-inf', 
                  'valid' : 'all'}
    metrics.append(med_ae_inf)

    #  median absolute error in the 0-1 range
    med_ae_01 = {'function' : median_absolute_error_range, 
                 'kwargs' : {'lower_bound' : 0, 'upper_bound' : 1, 'return_pct_kept' : False},
                 'name' : 'median_absolute_error-1',
                 'valid' : 'all'}
    metrics.append(med_ae_01)

    #  mean absolute percentage error, excluding small values
    mape_tau = {'function' : mean_absolute_percent_error_tau, 
                'kwargs' : {'tau' : 0.05},
                'kwarg_name_fill' : 'tau',
                'name' : 'mean_absolute_percentage_error-%s',
                'valid' : 'all'}
    metrics.append(mape_tau)
    
    # DISTRIBUTIONAL:
    #  wasserstein distance for the 0-2 range
    wd_eta = {'function' : wasserstein_distance_range,
              'kwargs' : {'lower_bound' : 0, 'upper_bound' : 2},
              'kwarg_name_fill' : 'upper_bound',
              'name' : 'wasserstein_distance-%s',
              'valid' : 'all'}
    metrics.append(wd_eta)

    # HURDLE COMPONENTS:
    #  balanced accuracy for the local extirpation model
    ba_local_extirp = {'function' : balanced_accuracy_score,
                       'kwargs' : {},
                       'name' : 'balanced_accuracy_local_extirpation',
                       'valid' : 'hurdle', 
                       'submodel' : 'zero'}
    metrics.append(ba_local_extirp)

    #  wasserstein distance for the continuous RR model, in 0-2 range
    wd_cont_eta = {'function' : wasserstein_distance_range,
                   'kwargs' : {'lower_bound' : 0, 'upper_bound' : 2},
                   'name' : 'wasserstein_distance_continuous-%s',
                   'kwarg_name_fill' : 'upper_bound',
                   'valid' : 'hurdle', 
                   'submodel' : 'nonzero'}
    metrics.append(wd_cont_eta)

    return metrics

def main(args):
    # Read in the raw predictions from the indicated directory
    runs_to_eval = os.listdir(args.pred_dir)
    runs_to_eval = [r for r in runs_to_eval if ' ' not in r] # getting rid of duplicate prediction files

    #  get run info and read in raw prediction dataframe
    runs_to_eval = [get_run_info_from_fname(r) for r in runs_to_eval]
    runs_to_eval = [r for r in runs_to_eval if r is not None]

    for r in runs_to_eval:
        fp = os.path.join(args.pred_dir, r['filename'])

        df = pd.read_csv(fp, index_col = 'index')
        if 'predicted_zero' in df.columns:
            df['actual_zero'] = (df['actual'] == 0).astype(int) if args.extirp_pos else (df['actual'] != 0).astype(int) 

        date = os.path.getmtime(fp)
        date = datetime.fromtimestamp(date).strftime('%Y-%m-%d %H:%M:%S')
        
        r['raw_preds'] = df
        r['new_metrics'] = {}
        r['date'] = date

    # Getting the list of metrics to use
    metrics = metrics_to_use()

    # Applying the metrics to the raw predictions, only applying hurdle-specific metrics to hurdle models
    for r in runs_to_eval:
        for m in metrics:
            submodel = None
            if m['valid'] == 'hurdle':
                if ('hurdle' in r['model_name']) and ('predicted_zero' in r['raw_preds'].columns):
                    submodel = m['submodel']
                else:
                    continue

            metric_name = m['name'] if '%' not in m['name'] else (m['name'] % m['kwargs'][m['kwarg_name_fill']])
            
            metric_mean, metric_std = get_metric_CV(r['raw_preds'], m['function'], submodel = submodel, **m['kwargs'])
            r['new_metrics'][metric_name] = {'mean' : metric_mean, 'standard_deviation' : metric_std}

    # Turning new metric results into a dataframe
    new_metrics = pd.DataFrame(columns = ['metric', 'mean', 'standard_deviation', 'model_name', 'dataset', 
                                          'date', 'num_folds', 'block_type', 'spatial_spacing', 'group_col'])

    i = 0
    for r in runs_to_eval:
        new_metrics_vals = r['new_metrics']
        for m in new_metrics_vals.keys():
            row = [m, new_metrics_vals[m]['mean'], new_metrics_vals[m]['standard_deviation'], r['model_name'], r['dataset'], 
                   r['date'], r['num_folds'], r['block_type'], r['spatial_spacing'], r['group_col']]
            
            new_metrics.loc[i] = row
            i += 1

    new_metrics = new_metrics.sort_values(by = ['model_name', 'dataset', 'block_type'])

    # Optionally saving results
    if args.save_results:
        new_metrics.to_csv(os.path.join(args.save_dir, 'cross_val_results_all_metrics.csv'), index = False)

    return new_metrics
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Collect command line arguments
    parser.add_argument('--pred_dir', type = str)
    parser.add_argument('--save_results', type = int, choices = [0, 1])

    parser.add_argument('--save_dir', type = str, default = '../phd_results')
    parser.add_argument('--extirp_pos', type = int, default = 0, choices = [0, 1])
    
    #  parse argument inputs and fixing types
    args = parser.parse_args()

    args.extirp_pos = bool(args.extirp_pos)
    args.save_results = bool(args.save_results)

    # Compute metrics
    new_metrics = main(args)
    print(new_metrics.head())