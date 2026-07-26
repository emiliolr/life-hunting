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
    # Resample the rows (indices)
    np.random.seed(1693)

    all_res_idxs = []
    idxs = [i for i in range(data.shape[0])]
    for i in range(args.num_resamples):
        res_idxs = np.random.choice(idxs, size = data.shape[0], replace = True)
        all_res_idxs.append(res_idxs)

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
    data = read_data(args)

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