import json
import os
import sys
import warnings
import argparse

sys.path.append('..')
warnings.simplefilter(action = 'ignore', category = FutureWarning)

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import balanced_accuracy_score, recall_score, mean_absolute_error, root_mean_squared_error
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV, LinearRegression, LogisticRegression
from sklearn.dummy import DummyRegressor

from pymer4 import Lmer
from flaml import AutoML

from utils import read_csv_non_utf
from model_utils import HurdleModelEstimator, PymerModelWrapper
from custom_metrics import balanced_accuracy_FLAML, median_absolute_error_FLAML, mean_absolute_error_range
from cross_validation import run_cross_val, save_cv_results

def read_data(args):
    # Loading in general configuration
    with open('../config.json', 'r') as f:
        config = json.load(f)

    # Getting filepaths
    if args.gdrive:
        gdrive_fp = config['gdrive_path']
        LIFE_fp = config['LIFE_folder']
        dataset_fp = config['datasets_path']

        benitez_lopez2019 = config['indiv_data_paths']['benitez_lopez2019']
        ferreiro_arias2024 = config['indiv_data_paths']['ferreiro_arias2024']

        ferreiro_arias2024_ext = config['indiv_data_paths']['ferreiro_arias2024_extended']
        benitez_lopez2019_ext = config['indiv_data_paths']['benitez_lopez2019_extended']

        benitez_lopez2019_rec = config['indiv_data_paths']['benitez_lopez2019_recreated']
        
        ben_lop_path = os.path.join(gdrive_fp, LIFE_fp, dataset_fp, benitez_lopez2019)
        fer_ari_path = os.path.join(gdrive_fp, LIFE_fp, dataset_fp, ferreiro_arias2024)
        fer_ari_ext_path = os.path.join(gdrive_fp, LIFE_fp, dataset_fp, ferreiro_arias2024_ext)
        ben_lop_ext_path = os.path.join(gdrive_fp, LIFE_fp, dataset_fp, benitez_lopez2019_ext)
        ben_lop_rec_path = os.path.join(gdrive_fp, LIFE_fp, dataset_fp, benitez_lopez2019_rec)
    else:
        ben_lop_path = config['remote_machine_paths']['benitez_lopez2019']
        fer_ari_path = config['remote_machine_paths']['ferreiro_arias2024']
        fer_ari_ext_path = config['remote_machine_paths']['ferreiro_arias2024_extended']
        ben_lop_ext_path = config['remote_machine_paths']['benitez_lopez2019_extended']

    # Reading in the dataset
    if args.dataset == 'birds':
        data = pd.read_csv(fer_ari_path)
    elif args.dataset == 'birds_extended':
        data = pd.read_csv(fer_ari_ext_path)

        #  bit of recoding on some categorical variables
        data['Trophic_Niche'] = data['Trophic_Niche'].apply(lambda x: x if x in ['Frugivore', 'Invertivore', 'Omnivore'] else 'Other')
        data['IUCN_Is_Threatened'] = data['IUCN_Category'].apply(lambda x: 1 if x == 'threatened or near threatened' else 0)
        data['Habitat_Is_Dense'] = data['Habitat_Density'].apply(lambda x: 1 if x == 'Dense' else 0)
        
        data = data.drop(columns = ['IUCN_Category', 'Habitat_Density'])
    elif args.dataset == 'mammals':
        data = read_csv_non_utf(ben_lop_path)
    elif args.dataset == 'mammals_recreated':
        data = pd.read_csv(ben_lop_rec_path)
    elif args.dataset == 'mammals_extended':
        data = pd.read_csv(ben_lop_ext_path)

        #  recoding IUCN category to make it a numeric indicator
        data['IUCN_Is_Threatened'] = data['IUCN_Category'].apply(lambda x: 1 if x == 'threatened or near threatened' else 0)

        data = data.drop(columns = ['IUCN_Category', 'Generation_Time']) # getting rid of generation time, too many missing values
    elif args.dataset == 'both':
        ben_lop2019 = read_csv_non_utf(ben_lop_path)
        fer_ari2024 = pd.read_csv(fer_ari_path)

        cols = ['Order', 'Family', 'Species', 'ratio', 'X', 'Y', 'Country', 'BM', 'DistKm', 'PopDens', 
                'Stunting', 'TravTime', 'LivestockBio', 'Reserve']
        ben_lop2019 = ben_lop2019[cols]
        ben_lop2019['Class'] = 'Mammalia'

        cols = ['Order', 'Family', 'Species', 'RR', 'Latitude', 'Longitude', 'Country', 'Body_Mass', 
                'Dist_Hunters', 'PopDens', 'Stunting', 'TravDist', 'FoodBiomass', 'Reserve']
        fer_ari2024 = fer_ari2024[cols]
        fer_ari2024['Class'] = 'Aves'
        fer_ari2024['Reserve'] = fer_ari2024['Reserve'].replace({0 : 'No', 1 : 'Yes'}) # aligning the coding of this binary columns to the mammal dataset
        
        fer_ari2024 = fer_ari2024.rename(columns = {'RR' : 'ratio', 'Longitude' : 'X', 'Latitude' : 'Y',
                                                    'Dist_Hunters' : 'DistKm', 'TravDist' : 'TravTime',
                                                    'FoodBiomass' : 'LivestockBio', 'Body_Mass' : 'BM'})

        data = pd.concat((ben_lop2019, fer_ari2024), join = 'inner', axis = 0, ignore_index = True)

    return data

def set_eval_metrics():
    # Defining the metrics to use
    class_metrics = {'per_class' : {'balanced_accuracy' : {'function' : balanced_accuracy_score,
                                                           'kwargs' : {}
                                                          },
                                    'sensitivity' : {'function' : recall_score,
                                                     'kwargs' : {'pos_label' : 1}
                                                    },
                                    'specificity' : {'function' : recall_score,
                                                     'kwargs' : {'pos_label' : 0}
                                                    }
                                   },
                     'overall' : {'balanced_accuracy_overall' : {'function' : balanced_accuracy_score,
                                                                 'kwargs' : {}
                                                                }
                                 }
                    }
    reg_metrics = {'mean_absolute_error' : {'function' : mean_absolute_error,
                                            'kwargs' : {}
                                           },
                   'root_mean_squared_error' : {'function' : root_mean_squared_error,
                                                'kwargs' : {}
                                               },
                   'mean_absolute_error_0-1' : {'function' : mean_absolute_error_range,
                                                 'kwargs' : {'lower_bound' : 0,
                                                             'upper_bound' : 1
                                                            }
                                               }
                  }
    
    return class_metrics, reg_metrics
    
def set_up_and_run_cross_val(args, data, class_metrics, reg_metrics):
    # Pymer hurdle model, for sanity checking
    if args.model_to_use == 'pymer':
        #  setting up the equations for each model
        if args.dataset == 'mammals':
            formula_zero = 'local_extirpation ~ BM + DistKm + I(DistKm^2) + PopDens + Stunting + Reserve + (1|Country) + (1|Species) + (1|Study)'
            formula_nonzero = 'RR ~ BM + DistKm + I(DistKm^2) + PopDens + I(PopDens^2) + BM*DistKm + (1|Country) + (1|Species) + (1|Study)'
        elif args.dataset == 'birds':
            formula_zero = 'local_extirpation ~ Body_Mass + Dist_Hunters + TravDist + PopDens + Stunting + NPP + Reserve + Body_Mass*Dist_Hunters + Body_Mass*TravDist + Body_Mass*Stunting + NPP*Dist_Hunters + (1|Country) + (1|Species)'
            formula_nonzero = 'RR ~ Body_Mass + Dist_Hunters + TravDist + PopDens + Stunting + NPP + Reserve + Body_Mass*Dist_Hunters + Body_Mass*TravDist + Body_Mass*Stunting + NPP*Dist_Hunters + (1|Country) + (1|Species)'
        elif args.dataset == 'both':
            formula_zero = 'local_extirpation ~ BM + DistKm + I(DistKm^2) + TravTime + PopDens + Stunting + Reserve + BM*DistKm + BM*TravTime + BM*Stunting + (1|Country) + (1|Species)'
            formula_nonzero = 'RR ~ BM + DistKm + I(DistKm^2) + TravTime + PopDens + I(PopDens^2) + Stunting + Reserve + BM*DistKm + BM*TravTime + BM*Stunting + (1|Country) + (1|Species)'
        else:
            raise ValueError('Dataset {args.dataset} not implemented for pymer hurdle model')

        if args.dataset == 'both':
            control_str = "optimizer='bobyqa', optCtrl=list(maxfun=1e6)"
        else:
            control_str = "optimizer='bobyqa', optCtrl=list(maxfun=1e5)"

        #  hurdle model params
        extirp_pos = False

        if args.outlier_cutoff is None:
            args.outlier_cutoff = 15 if args.dataset == 'mammals' else 5
        data_args = {'outlier_cutoff' : args.outlier_cutoff, 
                     'dataset' : args.dataset, 
                     'rebalance_dataset' : args.rebalance_dataset}

        #  setting up the hurdle model
        zero_model = PymerModelWrapper(Lmer, formula = formula_zero, family = 'binomial', control_str = control_str, 
                                       use_rfx = args.use_rfx)
        nonzero_model = PymerModelWrapper(Lmer, formula = formula_nonzero, family = 'gaussian', control_str = control_str, 
                                          use_rfx = args.use_rfx)

        model = HurdleModelEstimator(zero_model, nonzero_model, extirp_pos = extirp_pos, data_args = data_args)

        #  cross-validation params
        back_transform = True
        sklearn_submodels = False
        direct = None
        
        fit_args = None
        pp_args = {'include_indicators' : False,
                   'include_categorical' : True,
                   'polynomial_features' : 0,
                   'log_trans_cont' : True,
                   'dataset' : args.dataset}

        #  results saving params
        model_name = 'pymer_hurdle'
        model_name += '_w_rfx' if args.use_rfx else '_wo_rfx'
        
    # Sklearn fixed-effects hurdle model
    elif args.model_to_use == 'sklearn':
        #  hurdle model params
        extirp_pos = False
        verbose = False
        
        if args.dataset == 'mammals':
            zero_columns = ['BM', 'DistKm', 'PopDens', 'Stunting', 'TravTime', 'LivestockBio', 'Literacy', 'Reserve']
        elif args.dataset == 'birds':
            zero_columns = ['Dist_Hunters', 'TravDist', 'PopDens', 'Stunting', 'FoodBiomass', 'Forest_cover', 'NPP', 'Body_Mass']
        nonzero_columns = zero_columns
        indicator_columns = []
        
        data_args = {'indicator_columns' : indicator_columns,
                     'nonzero_columns' : nonzero_columns,
                     'zero_columns' : zero_columns,
                     'dataset' : args.dataset}

        #  cross-validation params for tuning zero/nonzero model hyperparams
        grid_cv = 5
        logistic_penalty = 'l1'
        l1_ratio = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1]
        Cs = 20

        #  setting up the hurdle model
        nonzero_model = ElasticNetCV(cv = grid_cv, l1_ratio = l1_ratio, max_iter = 5000)
        zero_model = LogisticRegressionCV(cv = grid_cv, Cs = Cs, penalty = logistic_penalty, solver = 'saga', max_iter = 500)
        model = HurdleModelEstimator(zero_model, nonzero_model, extirp_pos = extirp_pos, verbose = verbose,
                                    data_args = data_args)

        #  cross-validation params
        back_transform = True
        sklearn_submodels = True
        direct = None
        
        fit_args = None
        pp_args = {'include_indicators' : False,
                   'include_categorical' : False,
                   'polynomial_features' : 2,
                   'log_trans_cont' : False,
                   'dataset' : args.dataset}
        
        #  results saving params
        model_name = 'sklearn_hurdle'

    # FLAML AutoML hurdle model
    elif args.model_to_use == 'FLAML_hurdle':
        #  automl params
        base_path = os.path.join('..', 'model_saves')
        
        zero_metric = balanced_accuracy_FLAML
        nonzero_metric = median_absolute_error_FLAML

        #  hurdle model params
        verbose = 0
        extirp_pos = False
        
        if args.dataset in ['mammals', 'both']:
            zero_columns = ['BM', 'DistKm', 'PopDens', 'Stunting', 'TravTime', 'LivestockBio', 'Reserve'] 
            zero_columns = zero_columns + (['Literacy'] if args.dataset == 'mammals' else [])
        elif args.dataset == 'birds':
            zero_columns = ['Dist_Hunters', 'TravDist', 'PopDens', 'Stunting', 'FoodBiomass', 'Forest_cover', 'NPP', 
                            'Body_Mass', 'Reserve']
        elif args.dataset in ['birds_extended', 'mammals_extended', 'mammals_recreated']:
            zero_columns = None # just using defaults here, which is all available predictors...
        nonzero_columns = zero_columns
        indicator_columns = []
        
        #  setting up the zero and nonzero models
        zero_model = AutoML()
        nonzero_model = AutoML()
        
        #  specify fitting paramaters
        zero_models_to_try = args.flaml_single_model
        if zero_models_to_try is None:
            zero_models_to_try = ['lgbm', 'xgboost', 'xgb_limitdepth', 'rf', 'extra_tree', 'kneighbor', 'lrl1', 'lrl2']

        zero_settings = {
            'time_budget' : args.time_budget_mins * 60,  # in seconds
            'metric' : zero_metric,
            'task' : 'classification',
            'log_file_name' : os.path.join(base_path, f'{args.dataset}_nonlinear_hurdle_ZERO.log'),
            'seed' : 1693,
            'estimator_list' : zero_models_to_try,
            'early_stop' : True,
            'verbose' : verbose,
            'keep_search_state' : True,
            'eval_method' : 'cv'
        }
        
        nonzero_models_to_try = args.flaml_single_model
        if nonzero_models_to_try is None:
            nonzero_models_to_try = ['lgbm', 'xgboost', 'xgb_limitdepth', 'rf', 'extra_tree', 'kneighbor']

        nonzero_settings = {
            'time_budget' : args.time_budget_mins * 60,  # in seconds
            'metric' : nonzero_metric,
            'task' : 'regression',
            'log_file_name' : os.path.join(base_path, f'{args.dataset}_nonlinear_hurdle_NONZERO.log'),
            'seed' : 1693,
            'estimator_list' : nonzero_models_to_try,
            'early_stop' : True,
            'verbose' : verbose,
            'keep_search_state' : True,
            'eval_method' : 'cv'
        }

        #  optionally, adding ability to ensemble via stacked generalization
        if args.ensemble:
            zero_settings['ensemble'] = {'passthrough' : False, 'final_estimator' : LogisticRegression()}
            nonzero_settings['ensemble'] = {'passthrough' : False, 'final_estimator' : LinearRegression()}
        
        #  dumping everything into the hurdle model wrapper
        data_args = {'indicator_columns' : indicator_columns,
                     'nonzero_columns' : nonzero_columns,
                     'zero_columns' : zero_columns,
                     'dataset' : args.dataset,
                     'embeddings_to_use' : args.embeddings_to_use,
                     'rebalance_dataset' : args.rebalance_dataset,
                     'outlier_cutoff' : args.outlier_cutoff if args.outlier_cutoff is not None else np.Inf}
        model = HurdleModelEstimator(zero_model, nonzero_model, extirp_pos = extirp_pos, 
                                     data_args = data_args, verbose = False)

        #  cross-validation params
        back_transform = True
        sklearn_submodels = False
        direct = None
        
        fit_args = {'zero' : zero_settings, 'nonzero' : nonzero_settings}
        pp_args = {'include_indicators' : True if 'extended' in args.dataset else False,
                   'include_categorical' : False,
                   'polynomial_features' : 0,
                   'log_trans_cont' : False,
                   'dataset' : args.dataset,
                   'embeddings_to_use' : args.embeddings_to_use,
                   'embeddings_args' : args.embeddings_args}

        #  results saving params
        if args.flaml_single_model is None:
            model_name = f'FLAML_hurdle_{args.time_budget_mins}mins'
        else:
            model_name = f'{args.flaml_single_model[0]}_hurdle_{args.time_budget_mins}mins'

        if args.embeddings_to_use is not None:
            if (zero_columns is not None) and (nonzero_columns is not None):
                if (len(zero_columns) == 0) and (len(nonzero_columns) == 0):
                    model_name += '_JUST'
            model_name += f'_{'+'.join(args.embeddings_to_use)}'

        if args.ensemble:
            model_name += '_ensemble'

    # FLAML AutoML direct regression model
    elif args.model_to_use == 'FLAML_regression':
        #  initialize automl instance
        model = AutoML()
        
        #  specify paramaters
        base_path = os.path.join('..', 'model_saves', f'direct_regression')
        
        automl_settings = {
            'time_budget' : args.time_budget_mins * 60,  # in seconds
            'metric' : 'mse',
            'task' : 'regression',
            'log_file_name' : os.path.join(base_path, f'{args.dataset}_direct_regression.log'),
            'seed' : 1693,
            'estimator_list' : ['lgbm', 'xgboost', 'xgb_limitdepth', 'rf', 'extra_tree', 'kneighbor'],
            'early_stop' : True,
            'verbose' : 0,
            'eval_method' : 'cv'
        }

        #  cross-validation params
        back_transform = False
        sklearn_submodels = False
        direct = 'regression'
        args.tune_thresh = False

        fit_args = automl_settings
        pp_args = {'include_indicators' : False,
                   'include_categorical' : False,
                   'polynomial_features' : 0,
                   'log_trans_cont' : False,
                   'dataset' : args.dataset}

        #  results saving params
        model_name = f'FLAML_regression_{args.time_budget_mins}mins'

    # FLAML AutoML direct classification model
    elif args.model_to_use == 'FLAML_classification':
        #  initialize the automl instance
        model = AutoML()
        
        #  specify paramaters
        base_path = os.path.join('..', 'model_saves', f'direct_classification')
        
        automl_settings = {
            'time_budget' : args.time_budget_mins * 60,  # in seconds
            'metric' : balanced_accuracy_FLAML,
            'task' : 'classification',
            'log_file_name' : os.path.join(base_path, f'{args.dataset}_direct_classification.log'),
            'seed' : 1693,
            'estimator_list' : ['lgbm', 'xgboost', 'xgb_limitdepth', 'rf', 'extra_tree', 'kneighbor'],
            'early_stop' : True,
            'verbose' : 0,
            'eval_method' : 'cv'
        }

        #  cross-validation params
        back_transform = False
        sklearn_submodels = False
        direct = 'classification'
        args.tune_thresh = False
        reg_metrics = None

        fit_args = automl_settings
        pp_args = {'include_indicators' : False,
                'include_categorical' : False,
                'polynomial_features' : 0,
                'log_trans_cont' : False,
                'dataset' : args.dataset}

        #  results saving params
        model_name = f'FLAML_classification_{args.time_budget_mins}mins'

    # Dummy regressor
    elif args.model_to_use == 'dummy_regressor':
        model = DummyRegressor(strategy = args.dummy_strat)
        
        #  cross-validation params
        back_transform = False
        sklearn_submodels = False
        direct = 'regression'
        args.tune_thresh = False

        fit_args = None
        pp_args = {'include_indicators' : False,
                   'include_categorical' : False,
                   'polynomial_features' : 0,
                   'log_trans_cont' : False,
                   'dataset' : args.dataset}

        #  results saving params
        model_name = f'dummy_regressor_{args.dummy_strat}'

    # Some general configuration stuff
    if args.rebalance_dataset:
        model_name += '_rebalance-classes'
    if args.tune_thresh:
        model_name += '_tune-thresh'

    # Making sure the save directory exists for autoML training
    if args.model_to_use.startswith('FLAML'):
        if not os.path.exists(base_path):
            os.mkdir(base_path)

    # Printing some run info before running cross-validation
    print(f'Training/testing on {args.dataset} dataset{'s' if args.dataset == 'both' else ''}\n')
    print(f'Using {model_name} with{" no" if args.outlier_cutoff is None else ""} outlier cutoff {"of " + str(args.outlier_cutoff) if args.outlier_cutoff is not None else ""}\n')

    if (args.model_to_use == 'FLAML_hurdle') and (args.flaml_single_model is not None):
        print(f'Using FLAML to hyperparameter tune just {args.flaml_single_model[0]}\n')

    if args.dataset != 'both':
        all_metric_names = list(class_metrics['per_class']) + list(class_metrics['overall']) + (list(reg_metrics.keys()) if reg_metrics is not None else [])
        print(f'Metrics: {all_metric_names}\n')

    # Run the cross-validation using the inputted params
    metrics_dict = run_cross_val(model, data, block_type = args.block_type, num_folds = args.num_folds, 
                                 group_col = args.group_col, spatial_spacing = args.spatial_spacing, fit_args = fit_args, 
                                 pp_args = pp_args, class_metrics = class_metrics, reg_metrics = reg_metrics, 
                                 verbose = True, random_state = 1693, sklearn_submodels = sklearn_submodels, 
                                 back_transform = back_transform, direct = direct, tune_hurdle_thresh = args.tune_thresh)
    
    return metrics_dict, model_name
    
def save_results(args, metrics_dict, model_name, class_metrics, reg_metrics):
    # Saving and displaying results
    cross_val_params = {'num_folds' : args.num_folds,
                        'block_type' : args.block_type,
                        'spatial_spacing' : args.spatial_spacing,
                        'group_col' : args.group_col}

    #  making sure the save directory structure is set up correctly
    if not os.path.exists(args.save_fp):
        os.mkdir(args.save_fp)
        os.mkdir(os.path.join(args.save_fp, 'raw_predictions'))
        os.mkdir(os.path.join(args.save_fp, 'full_experiment_params'))

    save_cv_results(metrics_dict, model_name, args.save_fp, cross_val_params, class_metrics, reg_metrics,
                    args.vals_to_save, args.dataset, args)
    print(f'Saved results at {args.save_fp}')

def main(args):
    # Get the dataset
    data = read_data(args)

    # Get evaluation metrics
    class_metrics, reg_metrics = set_eval_metrics()

    # Run the cross validation with the chosen parameters, metrics, and dataset
    metrics_dict, model_name = set_up_and_run_cross_val(args, data, class_metrics, reg_metrics)

    # Save the result of the cross validation
    save_results(args, metrics_dict, model_name, class_metrics, reg_metrics)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # DATASET PARAMS
    parser.add_argument('--gdrive', type = int, default = 1)
    parser.add_argument('--dataset', type = str, default = 'birds', choices = ['mammals', 'birds', 'birds_extended', 
                                                                               'mammals_extended', 'mammals_recreated', 
                                                                               'both'])
    parser.add_argument('--rebalance_dataset', type = int, default = 0)
    parser.add_argument('--tune_thresh', type = int, default = 1)
    parser.add_argument('--outlier_cutoff', type = float, default = 1000)

    # MODEL PARAMS
    parser.add_argument('--model_to_use', type = str, default = 'FLAML_hurdle', choices = ['pymer', 'sklearn', 'FLAML_hurdle', 
                                                                                           'FLAML_regression', 'FLAML_classification', 
                                                                                           'dummy_regressor'])
    parser.add_argument('--vals_to_save', type = str, nargs = '*', default = ['metrics'], choices = ['metrics', 'raw'])

    # CROSS-VALIDATION PARAMS
    parser.add_argument('--num_folds', type = int, default = 5)
    parser.add_argument('--block_type', type = str, default = 'random', choices = ['random', 'group', 'spatial'])
    parser.add_argument('--group_col', type = str, default = 'species', choices = ['species'])
    parser.add_argument('--spatial_spacing', type = int, default = 5)

    # RESULTS SAVE PARAMS
    parser.add_argument('--save_fp', type = str, default = '../phd_results')

    # LINEAR RANDOM-EFFECTS MODEL PARAMS
    parser.add_argument('--use_rfx', type = int, default = 0)

    # NONLINEAR FLAML MODELS PARAMS
    parser.add_argument('--time_budget_mins', type = float, default = 0.1)
    parser.add_argument('--ensemble', type = int, default = 0)
    parser.add_argument('--flaml_single_model', type = str, default = '', choices = ['rf', 'xgboost'])

    # EMBEDDING PARAMS
    parser.add_argument('--embeddings_to_use', type = str, nargs = '*', default = [], choices = ['SatCLIP', 'BioCLIP'])

    # DUMMY REGRESSOR PARAMS
    parser.add_argument('--dummy_strat', type = str, default = 'mean', choices = ['mean', 'median'])

    # Parse args and fix some types
    args = parser.parse_args()

    if (args.group_col == 'species') and (args.dataset.startswith('birds') or (args.dataset.startswith('mammals'))):
        args.group_col = 'Species'

    if args.outlier_cutoff == 1000:
        args.outlier_cutoff = None

    if args.flaml_single_model == '':
        args.flaml_single_model = None
    else:
        args.flaml_single_model = [args.flaml_single_model]

    args.gdrive = bool(args.gdrive)
    args.use_rfx = bool(args.use_rfx)
    args.ensemble = bool(args.ensemble)
    args.rebalance_dataset = bool(args.rebalance_dataset)
    args.tune_thresh = bool(args.tune_thresh)

    if args.embeddings_to_use == []:
        args.embeddings_to_use = None
    args.embeddings_args = {'pca' : True, 'var_cutoff' : 0.9, 'satclip_L' : 10}

    # Run the cross-val loop using the inputted params
    main(args)