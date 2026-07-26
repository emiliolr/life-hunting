import os
import sys
import warnings
import argparse
import pickle
import json

import numpy as np
import pandas as pd

from run_cross_validation import read_data
from train_final_model import setup_and_train_model
from per_species_aoh_diffs import collect_aoh_info_one_species

class ArgsHolder:
    def __init__(self):
        pass

def get_args(params):
    # Loading parameters into an ArgsHolder wrapper for compatibility with
    #  other scripts
    args = ArgsHolder()

    #  params for bootstrapping
    args.num_resamples = params['num_resamples']

    #  params for reading the dataset
    args.gdrive = True if (mode == 'local') else False
    args.dataset = 'mammals_recreated'
    args.block_type = None

    #  params for training the final models (only 3-part RF)
    assert params['model_to_use'] == 'rf-3part', 'Only the three-part RF model is supported for study-level bootstrapping'
    args.model_to_use = 'FLAML_three_part'
    args.flaml_single_model = ['rf']
    args.time_budget_mins = 0.01
    args.trait_null = False

    args.rebalance_dataset = False
    args.tune_thresh = False
    args.outlier_cutoff = 3

    args.save_fp = '../testing_governance/bootstrapping'
    args.verbose = False

    return args

def resample_dataset(data, args):
    # Resample by study, aiming for approx the same amount of data each time
    studies = data['Study'].unique()
    tol = data['Study'].value_counts().max() + 5

    all_res_idxs = []
    for i in range(args.num_resamples):
        #  get a random resample of more studies than needed
        np.random.seed(i)
        studies_res = np.random.choice(studies, size = studies.shape[0] * 2, replace = True)
        
        #  starting with a good guess of how many of the(resampled) studies will be 
        #   needed, progressively add or subtract a study until we're at roughly
        #   the same amount of data as the full dataset
        correct_amt_of_data = False
        num_studies_from_res = studies.shape[0]
        while not correct_amt_of_data:
            studies_sub = studies_res[ : num_studies_from_res] # get the subset of resampled studies for this iter
        
            #  gather all the dataset indices for all studies study (with possible repeats!)
            idxs = []
            for s in studies_sub:
                idx_s = data[data['Study'] == s].index.to_list()
                idxs.extend(idx_s)
        
            #  check if we're inside the dataset size tolerances and if not, adjust accordingly
            not_too_much = (len(idxs) <= data.shape[0] + tol)
            not_too_little = (len(idxs) >= data.shape[0] - tol)

            if not_too_much and not_too_little:
                correct_amt_of_data = True # we can exit
            elif not not_too_much and not_too_little:
                num_studies_from_res -= 1
            elif not_too_much and not not_too_little:
                num_studies_from_res += 1

        all_res_idxs.append(np.array(idxs))

    return all_res_idxs

def train_resampled_models(data, resample_idxs, args):
    # Loop through the data resamples + train a model for each
    all_models = []
    for r in resample_idxs:
        data_res = data.iloc[r].copy(deep = True).reset_index(drop = True)
        trained_model = setup_and_train_model(args, data_res)
        all_models.append(trained_model)

    return all_models

def main(params):
    # Get the arguments to pass into other scripts
    args = get_args(params)

    # Read the data
    data = read_data(args).reset_index(drop = True)

    #  get the resampled indices to subset the dataset
    resample_idxs = resample_dataset(data, args)

    # Train the models on each resampled dataset
    models = train_resampled_models(data, resample_idxs, args)

if __name__ == '__main__':
    # Read in parameters
    with open('experiments/study_bootstrap.json', 'r') as f:
        params = json.load(f)

    # Choosing either "local" or "remote"
    mode = 'local'
    print(f'Running in {mode} mode\n')

    main(params)