import os
import sys
import warnings
import argparse
import pickle

sys.path.append('..')
warnings.simplefilter(action = 'ignore', category = FutureWarning)

import numpy as np

from pymer4 import Lmer
from flaml import AutoML

from utils import read_csv_non_utf, preprocess_data, get_zero_nonzero_datasets, test_thresholds
from model_utils import HurdleModelEstimator, PymerModelWrapper, ThreePartModel
from custom_metrics import balanced_accuracy_FLAML, median_absolute_error_FLAML
from run_cross_validation import read_data

def setup_and_train_model(args, data):
    # ML hurdle model, w/hyperparam tuning through FLAML
    if args.model_to_use == 'FLAML':
        #  setting up infrastructure to save FLAML logs
        base_path = os.path.join(args.save_fp, 'flaml_history')
        if not os.path.exists(base_path):
            os.mkdir(base_path)

        m = args.flaml_single_model[0] if args.flaml_single_model is not None else 'FLAML'
        zero_history_fp = f'{args.dataset}_{m}_ZERO.log'
        nonzero_history_fp = f'{args.dataset}_{m}_NONZERO.log'
        
        #  FLAML objectives
        zero_metric = balanced_accuracy_FLAML
        nonzero_metric = median_absolute_error_FLAML

        #  hurdle model params
        verbose = 0
        extirp_pos = False
        pca_cols = None

        if args.dataset in ['mammals']:
            zero_columns = ['BM', 'DistKm', 'PopDens', 'Stunting', 'TravTime', 'LivestockBio', 'Reserve', 'Literacy'] 
        elif args.dataset in ['mammals_extended']:
            zero_columns = None
        elif args.dataset == 'mammals_recreated':
            if args.flaml_single_model == ['rf']:
                zero_columns = ['Body_Mass', 'Stunting_Pct', 'Literacy_Rate', 'Dist_Settlement_KM', 
                                'Travel_Time_Large', 'Livestock_Biomass', 'Population_Density', 
                                'Percent_Settlement_50km', 'Protected_Area']
            elif args.flaml_single_model == ['rf-gov']:
                zero_columns = ['Body_Mass', 'Stunting_Pct', 'Literacy_Rate', 'Dist_Settlement_KM', 
                                'Travel_Time_Large', 'Livestock_Biomass', 'Population_Density', 
                                'Percent_Settlement_50km', 'Protected_Area', 'Corruption', 
                                'Government_Effectiveness', 'Political_Stability', 'Regulation', 
                                'Rule_of_Law', 'Accountability']
            elif args.flaml_single_model == ['rf-pca']:
                zero_columns = ['Body_Mass', 'Stunting_Pct', 'Literacy_Rate', 'Dist_Settlement_KM', 
                                'Travel_Time_Large', 'Livestock_Biomass', 'Population_Density', 
                                'Percent_Settlement_50km', 'Protected_Area', 'PC']
                pca_cols = ['Corruption', 'Government_Effectiveness', 'Political_Stability', 'Regulation', 
                            'Rule_of_Law', 'Accountability']
            else:
                zero_columns = None

        nonzero_columns = zero_columns
        indicator_columns = []
        
        #  setting up the zero and nonzero models
        zero_model = AutoML()
        nonzero_model = AutoML()
        
        #  specify fitting paramaters
        zero_models_to_try = [args.flaml_single_model[0].replace('-pca', '').replace('-gov', '')]
        if zero_models_to_try is None:
            zero_models_to_try = ['rf', 'lgbm', 'xgboost']

        zero_settings = {
            'time_budget' : args.time_budget_mins * 60,  # in seconds
            'metric' : zero_metric,
            'task' : 'classification',
            'log_file_name' : os.path.join(base_path, zero_history_fp),
            'seed' : 1693,
            'estimator_list' : zero_models_to_try,
            'early_stop' : True,
            'verbose' : verbose,
            'keep_search_state' : True,
            'eval_method' : 'cv'
        }
        
        nonzero_models_to_try = [args.flaml_single_model[0].replace('-pca', '').replace('-gov', '')]
        if nonzero_models_to_try is None:
            nonzero_models_to_try = ['rf', 'lgbm', 'xgboost']

        nonzero_settings = {
            'time_budget' : args.time_budget_mins * 60,  # in seconds
            'metric' : nonzero_metric,
            'task' : 'regression',
            'log_file_name' : os.path.join(base_path, nonzero_history_fp),
            'seed' : 1693,
            'estimator_list' : nonzero_models_to_try,
            'early_stop' : True,
            'verbose' : verbose,
            'keep_search_state' : True,
            'eval_method' : 'cv'
        }
        
        #  dumping everything into the hurdle model wrapper
        data_args = {'indicator_columns' : indicator_columns,
                     'nonzero_columns' : nonzero_columns,
                     'zero_columns' : zero_columns,
                     'dataset' : args.dataset,
                     'embeddings_to_use' : None,
                     'rebalance_dataset' : args.rebalance_dataset,
                     'outlier_cutoff' : args.outlier_cutoff if args.outlier_cutoff is not None else np.Inf}
        model = HurdleModelEstimator(zero_model, nonzero_model, extirp_pos = extirp_pos, 
                                     data_args = data_args, verbose = args.verbose)

        #  preprocessing + model fitting params
        fit_args = {'zero' : zero_settings, 'nonzero' : nonzero_settings}
        pp_args = {'include_indicators' : True if ('extended' in args.dataset) or ('recreated' in args.dataset) else False,
                   'include_categorical' : False,
                   'polynomial_features' : 0,
                   'log_trans_cont' : False,
                   'dataset' : args.dataset,
                   'embeddings_to_use' : None,
                   'embeddings_args' : None,
                   'pca_cols' : pca_cols}

        #  results saving params
        if args.flaml_single_model is None:
            args.model_name = f'FLAML_hurdle_{args.time_budget_mins}mins'
        else:
            args.model_name = f'{args.flaml_single_model[0]}_hurdle_{args.time_budget_mins}mins'

    # Linear mixed-effects hurdle model
    elif args.model_to_use == 'pymer':
        #  setting up the equations for each model
        if args.dataset == 'mammals':
            if args.pymer_zero_formula == '':
                args.pymer_zero_formula = 'local_extirpation ~ BM + DistKm + I(DistKm^2) + PopDens + Stunting + Reserve + (1|Country) + (1|Species) + (1|Study)'
            if args.pymer_nonzero_formula == '':
                args.pymer_nonzero_formula = 'RR ~ BM + DistKm + I(DistKm^2) + PopDens + I(PopDens^2) + BM*DistKm + (1|Country) + (1|Species) + (1|Study)'
        elif args.dataset == 'mammals_recreated':
            if args.pymer_zero_formula == '':
                args.pymer_zero_formula = 'local_extirpation ~ Body_Mass + Dist_Settlement_KM + I(Dist_Settlement_KM^2) + Population_Density + Stunting_Pct + Protected_Area + (1|Country) + (1|Species) + (1|Study)'
            if args.pymer_nonzero_formula == '':
                args.pymer_nonzero_formula = 'RR ~ Body_Mass + Dist_Settlement_KM + I(Dist_Settlement_KM^2) + Population_Density + I(Population_Density^2) + Body_Mass*Dist_Settlement_KM + (1|Country) + (1|Species) + (1|Study)'
        else:
            raise ValueError('Dataset {args.dataset} not implemented for pymer hurdle model')

        control_str = "optimizer='bobyqa', optCtrl=list(maxfun=1e5)"

        #  hurdle model params
        extirp_pos = False
        indicator_columns = ['Country', 'Species'] + (['Study'] if args.dataset in ['mammals', 'mammals_recreated'] else [])

        if args.outlier_cutoff is None:
            args.outlier_cutoff = 15 if args.dataset == 'mammals' else 5
        data_args = {'outlier_cutoff' : args.outlier_cutoff, 
                     'dataset' : args.dataset, 
                     'rebalance_dataset' : args.rebalance_dataset,
                     'indicator_columns' : indicator_columns}

        #  setting up the hurdle model
        zero_model = PymerModelWrapper(Lmer, formula = args.pymer_zero_formula, family = 'binomial', 
                                       control_str = control_str, use_rfx = False, reml = args.pymer_reml)
        nonzero_model = PymerModelWrapper(Lmer, formula = args.pymer_nonzero_formula, family = 'gaussian', 
                                          control_str = control_str, use_rfx = False, reml = args.pymer_reml)

        model = HurdleModelEstimator(zero_model, nonzero_model, extirp_pos = extirp_pos, 
                                     data_args = data_args, verbose = args.verbose)

        #  preprocessing + model fitting params
        fit_args = None
        pp_args = {'include_indicators' : False,
                   'include_categorical' : True,
                   'polynomial_features' : 0,
                   'log_trans_cont' : True,
                   'dataset' : args.dataset}

        #  results saving params
        args.model_name = 'pymer_hurdle'

    elif args.model_to_use == 'FLAML_three_part':
        #  automl params
        base_path = os.path.join(args.save_fp, 'flaml_history')
        if not os.path.exists(base_path):
            os.mkdir(base_path)
        
        class_metric = balanced_accuracy_FLAML
        reg_metric = median_absolute_error_FLAML

        #  three-part model params
        verbose = 0
        args.tune_hurdle_thresh = False
        
        if args.dataset == 'mammals_recreated':
            pred_cols = ['Body_Mass', 'Stunting_Pct', 'Literacy_Rate', 'Dist_Settlement_KM', 
                         'Travel_Time_Large', 'Livestock_Biomass', 'Population_Density', 
                         'Percent_Settlement_50km', 'Protected_Area', 'PC']
            pca_cols = ['Corruption', 'Government_Effectiveness', 'Political_Stability', 'Regulation', 
                        'Rule_of_Law', 'Accountability']
        else:
            raise ValueError('Only "mammals_recreated" is supported right now for the three-part model.')

        classifier_columns = pred_cols
        decrease_columns = pred_cols
        increase_columns = pred_cols
        indicator_columns = []
        
        #  setting up the three constituent models
        class_model = AutoML()
        decrease_model = AutoML()
        increase_model = AutoML()
        
        #  specify fitting paramaters
        if args.flaml_single_model is None:
            class_models_to_try = ['rf', 'xgboost']
        else:
            class_models_to_try = [args.flaml_single_model[0].replace('-pca', '').replace('-gov', '')]

        class_settings = {
            'time_budget' : args.time_budget_mins * 60,  # in seconds
            'metric' : class_metric,
            'task' : 'classification',
            'log_file_name' : os.path.join(base_path, f'{args.dataset}_three_part_classifier.log'),
            'seed' : 1693,
            'estimator_list' : class_models_to_try,
            'early_stop' : True,
            'verbose' : verbose,
            'keep_search_state' : True,
            'eval_method' : 'cv'
        }
        
        if args.flaml_single_model is None:
            reg_models_to_try = ['rf', 'xgboost']
        else:
            reg_models_to_try = [args.flaml_single_model[0].replace('-pca', '').replace('-gov', '')]

        decrease_settings = {
            'time_budget' : (args.time_budget_mins / 2) * 60,  # in seconds
            'metric' : reg_metric,
            'task' : 'regression',
            'log_file_name' : os.path.join(base_path, f'{args.dataset}_three_part_decrease.log'),
            'seed' : 1693,
            'estimator_list' : reg_models_to_try,
            'early_stop' : True,
            'verbose' : verbose,
            'keep_search_state' : True,
            'eval_method' : 'cv'
        }

        increase_settings = {
            'time_budget' : (args.time_budget_mins / 2) * 60,  # in seconds
            'metric' : reg_metric,
            'task' : 'regression',
            'log_file_name' : os.path.join(base_path, f'{args.dataset}_three_part_increase.log'),
            'seed' : 1693,
            'estimator_list' : reg_models_to_try,
            'early_stop' : True,
            'verbose' : verbose,
            'keep_search_state' : True,
            'eval_method' : 'cv'
        }
        
        #  dumping everything into the three-part model wrapper
        data_args = {'indicator_columns' : indicator_columns,
                     'classifier_columns' : classifier_columns,
                     'decrease_columns' : decrease_columns,
                     'increase_columns' : increase_columns,
                     'dataset' : args.dataset,
                     'rebalance_dataset' : args.rebalance_dataset,
                     'outlier_cutoff' : args.outlier_cutoff if args.outlier_cutoff is not None else np.Inf,
                     'neighborhood' : 0.1}
        model = ThreePartModel(class_model, decrease_model, increase_model, data_args = data_args, 
                               classes_enc = None, verbose = verbose)

        #  preprocessing + model fitting params
        fit_args = {'classifier' : class_settings, 
                    'decrease' : decrease_settings,
                    'increase' : increase_settings}
        pp_args = {'include_indicators' : True,
                   'include_categorical' : False,
                   'polynomial_features' : 0,
                   'log_trans_cont' : False,
                   'dataset' : args.dataset,
                   'pca_cols' : pca_cols}

        #  results saving params
        if args.flaml_single_model is None:
            args.model_name = f'FLAML_three_part_{args.time_budget_mins}mins'
        else:
            args.model_name = f'{args.flaml_single_model[0]}_three_part_{args.time_budget_mins}mins'

    # Preprocess the data
    pp_data = preprocess_data(data, standardize = True, **pp_args)

    # Train the hurdle model
    if args.verbose:
        print(f'Training the {args.model_to_use}{" hurdle" if "three_part" not in args.model_to_use else ""} model on {args.dataset}')
        if args.flaml_single_model is not None:
            print(f'  (single FLAML model - {args.flaml_single_model[0]})')
    train_model(args, model, pp_data, fit_args)

    return model

def train_model(args, model, pp_data, fit_args):
    # Train the model
    with warnings.catch_warnings(action = 'ignore'):
        model.fit(pp_data, fit_args)

    # Optionally tuning the probability threshold for the zero component of the hurdle model
    if args.tune_thresh:
        X_zero, y_zero, _, _ = get_zero_nonzero_datasets(pp_data, extirp_pos = model.extirp_pos,
                                                         pred = False, **model.data_args)
        y_pred = model.zero_model.predict_proba(X_zero)[ : , 1]

        opt_thresh, _ = test_thresholds(y_pred, y_zero)
        model.prob_thresh = round(opt_thresh, 3)

        if args.verbose:
            print(f'  optimal threshold was found to be {model.prob_thresh}')

def save_model(args, trained_model):
    # Pickling the model and saving in the correct directory
    fp = os.path.join(args.save_fp, f'{args.model_name}.pkl')

    with open(fp, 'wb') as f:
        pickle.dump(trained_model, f)

    if args.verbose:
        print(f'Saved trained model at {args.save_fp}')

    return fp

def main(args):
    # Get the dataset
    data = read_data(args)

    # Make sure the save directory exists
    if not os.path.exists(args.save_fp):
        os.mkdir(args.save_fp)

    # Train the model on the full dataset
    trained_model = setup_and_train_model(args, data)

    # Save the model to the indicated directory, returning save filepath for testing
    if args.save_trained_model:
        fp = save_model(args, trained_model)

    if args.return_model:
        return trained_model
    return fp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # DATASET PARAMS
    parser.add_argument('--gdrive', type = int, default = 1)
    parser.add_argument('--dataset', type = str, default = 'birds', choices = ['mammals', 'mammals_extended', 'mammals_recreated'])
    parser.add_argument('--rebalance_dataset', type = int, default = 0)
    parser.add_argument('--tune_thresh', type = int, default = 1)
    parser.add_argument('--outlier_cutoff', type = float, default = 1000)

    # MODEL PARAMS
    parser.add_argument('--model_to_use', type = str, default = 'FLAML', choices = ['pymer', 'FLAML', 'FLAML_three_part'])

    # RESULTS SAVE PARAMS
    parser.add_argument('--save_fp', type = str, default = '../final_models')

    # NONLINEAR FLAML MODELS PARAMS
    parser.add_argument('--time_budget_mins', type = float, default = 0.1)
    parser.add_argument('--flaml_single_model', type = str, default = '', choices = ['rf', 'xgboost', 'lgbm', 'rf-gov', 'rf-pca'])

    # LINEAR PYMER MODEL PARAMS
    parser.add_argument('--pymer_zero_formula', type = str, default = '')
    parser.add_argument('--pymer_nonzero_formula', type = str, default = '')

    # Parse args and fix some types
    args = parser.parse_args()

    if args.flaml_single_model == '':
        args.flaml_single_model = None
    else:
        args.flaml_single_model = [args.flaml_single_model]

    args.gdrive = bool(args.gdrive)
    args.rebalance_dataset = bool(args.rebalance_dataset)
    args.tune_thresh = bool(args.tune_thresh)

    #  a few arguments to set this apart from the feature selection script
    args.verbose = True
    args.return_model = True
    args.save_trained_model = True
    args.pymer_reml = True # for fitting final models, always use restricted max likelihood

    # Train the model on the full dataset using the inputted parameters
    model = main(args)