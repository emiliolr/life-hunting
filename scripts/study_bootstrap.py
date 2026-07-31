import os
import sys
import json
import glob
import pickle
from itertools import chain

import numpy as np
import pandas as pd

from joblib import Parallel, delayed

import rioxarray as rxr
import xarray as xr
import geopandas as gpd

from run_cross_validation import read_data
from train_final_model import setup_and_train_model
from model_projection import read_predictor_stack

class ArgsHolder:
    def __init__(self):
        pass

def get_args(params, mode):
    # Loading parameters into an ArgsHolder wrapper for compatibility with
    #  other scripts
    args = ArgsHolder()
    filepaths = params['filepaths'][mode]

    #  params for bootstrapping
    args.num_resamples = params['num_resamples']
    args.retrain_models = bool(params['retrain_models'])
    args.run_model_proj = bool(params['run_model_proj'])
    args.model_save_dir = filepaths['model_save_dir']

    #  params for reading the dataset
    args.gdrive = True if (mode == 'local') else False
    args.dataset = 'mammals_recreated'
    args.block_type = None

    #  params for training the final models (only 3-part RF)
    assert params['model_to_use'] == 'rf-3part', 'Only the three-part RF model is supported for study-level bootstrapping'
    args.model_to_use = 'FLAML_three_part'
    args.flaml_single_model = ['rf']
    args.time_budget_mins = 1
    args.trait_null = False

    args.rebalance_dataset = False
    args.tune_thresh = False
    args.outlier_cutoff = 3

    args.save_fp = '../testing_governance/bootstrapping'
    args.verbose = False

    #  params for model projection
    args.hybrid_hab_map = params['hybrid_hab_map']

    args.predictor_stack_fp = filepaths['predictor_stack_fp']
    args.mammals_data_fp = filepaths['mammals_data_fp']
    args.tropical_mammals_fp = filepaths['tropical_mammals_fp']
    args.tropical_zone_fp = filepaths['tropical_zone_fp']
    
    args.cur_aoh_dir = filepaths['current_aoh_dir'] % (filepaths['hybrid_dir'] if args.hybrid_hab_map else filepaths['non_hybrid_dir'])
    args.hum_abs_aoh_dir = filepaths['human_absent_aoh_dir'] % (filepaths['hybrid_dir'] if args.hybrid_hab_map else filepaths['non_hybrid_dir'])

    args.iucn_ids = params['iucn_id_subset']
    args.apply_standardization = False # we're going to do this model-by-model

    #  params for collecting AOH stats
    args.no_increase = params['no_increase']
    args.just_tropical_forest = params['just_tropical_forest']
    args.num_cores = params['num_cores']
    
    args.hunting_preds_dir = filepaths['hunting_preds_dir']

    return args

def resample_dataset(data, args):
    # Resample by study, aiming for approx the same amount of data each time
    studies = data['Study'].unique()
    tol = int(data['Study'].value_counts().max() / 2) + 1

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
                correct_amt_of_data = True # we can exit the loop
            elif not not_too_much and not_too_little:
                num_studies_from_res -= 1
            elif not_too_much and not not_too_little:
                num_studies_from_res += 1

        all_res_idxs.append(np.array(idxs))

    return all_res_idxs

def train_resampled_models(data, resample_idxs, args):
    # Checking to see what's already been trained
    file_pattern = os.path.join(args.model_save_dir, f'rf-3part_{args.time_budget_mins}_*.pkl')
    model_fps = glob.glob(file_pattern)

    #  case where we're retaining: loop through the data resamples + train a model for each
    all_models = []
    if args.retrain_models:
        resample_idxs = resample_idxs[len(model_fps) : ] # only training models that we don't already have

        for i, r in zip(range(len(model_fps), args.num_resamples), resample_idxs):
            print(f'  training model {i}')

            data_res = data.iloc[r].copy(deep = True).reset_index(drop = True)
            trained_model = setup_and_train_model(args, data_res)
            trained_model.resampled_idxs = r

            #  adding to our running list + saving to file
            all_models.append(trained_model)
            fp = os.path.join(args.model_save_dir, f'rf-3part_{args.time_budget_mins}_{i}.pkl')
            with open(fp, 'wb') as f:
                pickle.dump(trained_model, f)

    #  case where we're retrieving: get existing pre-trained model saves
    else:
        assert len(model_fps) >= args.num_resamples, 'Didn\'t find enough saved models for the requested number of resamples'

        for i in range(args.num_resamples):
            with open(model_fps[i], 'rb') as f:
                model = pickle.load(f)
                all_models.append(model)

    return all_models

def apply_all_models_one_species(species, tropical_mammals, predictor_stack, tropical_zone, mammals_data, model_list,
                                 aoh_dir, hybrid_hab_map):
    # Reading in the relevant AOH
    if hybrid_hab_map:
        aoh_poss_fps = glob.glob(os.path.join(aoh_dir, f'aoh_T{species}A*_RESIDENT.tif'))
        if len(aoh_poss_fps) != 1:
            return -2, None
        aoh_fp = aoh_poss_fps[0]
    else:
        aoh_fp = os.path.join(aoh_dir, f'{species}_RESIDENT.tif')

    if not os.path.isfile(aoh_fp):
      return -2, None

    aoh = rxr.open_rasterio(aoh_fp)

    # Clipping the predictor rasters to the bounds of the AOH
    predictor_stack_clipped = predictor_stack.rio.clip_box(*aoh.rio.bounds())

    #  making sure the predictor stack is perfectly aligned w/AOH
    predictor_stack_clipped = predictor_stack_clipped.rio.reproject_match(aoh)

    # Masking predictions outside of the AOH & tropical forest zone (the intersection of the two)
    aoh_in_forest_zone = aoh.rio.clip(tropical_zone, all_touched = True).fillna(0) # making sure to set NAs back to 0

    #  applying to the predictor stack
    predictor_stack_clipped = predictor_stack_clipped.where(aoh_in_forest_zone != 0)

    # Calculating the area overlap of AOH & tropical forest as a percent of total AOH
    aoh_total = float(aoh.sum())
    aoh_in_forest = float(aoh_in_forest_zone.sum())

    #  handling the case where there's no AOH at all...
    if aoh_total == 0:
        return -2, None

    pct_overlap = aoh_in_forest / aoh_total

    #  skip making predictions if there's no overlap w/tropical forest
    if pct_overlap == 0:
        return 0, None

    # Extracting the data to numpy + reshaping to get it in a "tabular" format
    predictor_stack_np = predictor_stack_clipped.to_array().variable.values.squeeze(axis = 3)

    num_y, num_x = predictor_stack_np[0].shape
    predictors_tabular = predictor_stack_np.reshape(predictor_stack_np.shape[0], num_y * num_x).transpose()

    #  tossing nan rows, but keeping track of where they are for reshaping back to raster later
    nan_mask = np.any(np.isnan(predictors_tabular), axis = 1)
    predictors_tabular_no_nan = predictors_tabular[~nan_mask, : ]

    #  error handling for no pixels to predict on - should only be an issue if the predictor 
    #   rasters have gaps that preclude model prediction (or poor alignment)
    if predictors_tabular_no_nan.shape[0] == 0:
        return -1, None

    # Putting data in a Pandas DataFrame so the predict function of the hurdle model can grab the right vars
    predictors_tabular_no_nan = pd.DataFrame(predictors_tabular_no_nan, columns = list(predictor_stack_clipped.keys()))
    
    # Applying all resampled models one by one (each applies their own normalization to data)
    pred_list = [] # TODO: probably more space-efficient if this isn't held in a last, maybe xarray stack?
    for model in model_list:
        #  getting a copy of the prediction data + the resampled training data for stats
        predictors_tabular_no_nan_i = predictors_tabular_no_nan.copy(deep = True)
        mammals_data_i = mammals_data.iloc[model.resampled_idxs].copy(deep = True).reset_index(drop = True)

        #  normalizing the spatial predictors
        for c in predictors_tabular_no_nan_i.columns:
            if c == 'Protected_Area':
                continue

            c_mean, c_std = float(mammals_data_i[c].mean()), float(mammals_data_i[c].std())
            predictors_tabular_no_nan_i[c] = ((predictors_tabular_no_nan_i[c] - c_mean) / c_std)

        #  adding the same standardized body mass value to each row
        bm = mammals_data_i['Body_Mass']
        bm_mean, bm_std = bm.mean(), bm.std()

        species_bm = tropical_mammals[tropical_mammals['iucn_id'] == species]['combine_body_mass'].iloc[0]
        species_bm = (species_bm - bm_mean) / bm_std

        predictors_tabular_no_nan_i['Body_Mass'] = species_bm

        #  apply the trained hurdle model to each pixel iteratively
        pred = model.predict(predictors_tabular_no_nan_i)

        #  putting the dataset all back together in a predicted raster
        pred_tabular = np.empty(shape = predictors_tabular.shape[0])
        pred_tabular.fill(np.nan)
        pred_tabular[~nan_mask] = pred # one prediction for each pixel, w/nans put back in the right place

        #  reshaping back to raster format + converting back to xarray
        pred_raster = pred_tabular.transpose().reshape(num_y, num_x)
        pred_raster_xr = xr.zeros_like(aoh_in_forest_zone)
        pred_raster_xr.values = np.expand_dims(pred_raster, axis = 0)

        pred_list.append(pred_raster_xr)
    
    return pred_list, aoh

def get_aoh_stats_one_species(args, species, tropical_mammals, predictor_stack, tropical_zone, mammals_data, 
                              model_list):
    # Predict current + human-absent HP maps FOR EACH MODEL
    cur_hp_maps, current_aoh = apply_all_models_one_species(species, tropical_mammals, predictor_stack, 
                                                            tropical_zone, mammals_data, model_list, args.cur_aoh_dir, 
                                                            args.hybrid_hab_map)
    hum_abs_hp_maps, human_absent_aoh, = apply_all_models_one_species(species, tropical_mammals, predictor_stack, 
                                                                      tropical_zone, mammals_data, model_list, 
                                                                      args.hum_abs_aoh_dir, args.hybrid_hab_map)
    
    #  skip species that were filtered out during model projection for various reasons
    if isinstance(cur_hp_maps, int) or isinstance(hum_abs_hp_maps, int):
        return [{'species' : species}]

    # Calculate eAOH stats (human-absent (+ w/HP) and current (+ w/HP))
    if args.just_tropical_forest:
        human_absent_aoh = human_absent_aoh.rio.clip(tropical_zone, all_touched = True).fillna(0) # making sure to set NAs back to 0
        current_aoh = current_aoh.rio.clip(tropical_zone, all_touched = True).fillna(0)

    #  getting human-absent + current AOH totals (same for all models)
    human_absent_aoh_total = float(human_absent_aoh.sum())
    current_aoh_total = float(current_aoh.sum())

    aoh_stats = []
    for current_hp, human_absent_hp in zip(cur_hp_maps, hum_abs_hp_maps):
        #  optionally, capping RRs at 1 (no change)
        if args.no_increase:
            current_hp = current_hp.clip(max = 1)
            human_absent_hp = human_absent_hp.clip(max = 1)

        #  ensure hunting pressure maps align precisely w/respective AOHs
        human_absent_hp = human_absent_hp.rio.reproject_match(human_absent_aoh)
        current_hp = current_hp.rio.reproject_match(current_aoh)

        #  putting RR=1 (no hunting effect) in AOH areas with no predictions for hunting maps
        no_pred_mask = ((human_absent_aoh != 0) & (xr.ufuncs.isnan(human_absent_hp)))
        human_absent_hp = human_absent_hp.where(~no_pred_mask, other = 1)

        no_pred_mask = ((current_aoh != 0) & (xr.ufuncs.isnan(current_hp)))
        current_hp = current_hp.where(~no_pred_mask, other = 1)

        #  element-wise multiplications w/predicted hunting pressure maps to get
        #    scenarios with hunting
        human_absent_aoh_w_hunting = human_absent_aoh * human_absent_hp
        human_absent_aoh_w_hunting_total = float(human_absent_aoh_w_hunting.sum())

        current_aoh_w_hunting = current_aoh * current_hp
        current_aoh_w_hunting_total = float(current_aoh_w_hunting.sum())

        #  add to the running list of eAOH stats
        aoh_dict = {'species' : species,
                    'human_absent_aoh_total' : human_absent_aoh_total,
                    'current_aoh_total' : current_aoh_total, 
                    'human_absent_aoh_w_hunting_total' : human_absent_aoh_w_hunting_total,
                    'current_aoh_w_hunting_total' : current_aoh_w_hunting_total}
        aoh_stats.append(aoh_dict)
    
    return aoh_stats

def calculate_resampled_aoh_stats(args, models, tropical_mammals, predictor_stack, mammals_data):
    # Grabbing the subset of IDs to run over
    if isinstance(args.iucn_ids, list):
        if len(args.iucn_ids) == 0:
            args.iucn_ids = tropical_mammals['iucn_id'].to_list()
    elif isinstance(args.iucn_ids, int):
        args.iucn_ids = tropical_mammals['iucn_id'].iloc[ : args.iucn_ids].to_list()
    print(f'\nPredicting over {len(args.iucn_ids)} species with {len(models)} resamples')

    # Reading the tropical forest extent polygon for masking non-forest pixels
    tropical_zone = gpd.read_file(args.tropical_zone_fp)
    tropical_zone = [tropical_zone.geometry.iloc[0]]

    # Get AOH stats for all species over the models trained on the resampled data
    all_aoh_stats = Parallel(n_jobs = args.num_cores, verbose = 10)(delayed(get_aoh_stats_one_species)(args, 
                                                                                                       sp, 
                                                                                                       tropical_mammals, 
                                                                                                       predictor_stack, 
                                                                                                       tropical_zone, 
                                                                                                       mammals_data, 
                                                                                                       models) for sp in args.iucn_ids)

    #  turn the AOH stats data into a proper dataframe
    resampled_aoh_stats = pd.DataFrame(chain(*all_aoh_stats))

    return resampled_aoh_stats

def main(params, mode):
    # Get the arguments to pass into other scripts
    args = get_args(params, mode)

    # Read the data
    print('Reading and resampling the data')
    data = read_data(args).reset_index(drop = True)

    #  get the resampled indices to subset the dataset
    resample_idxs = resample_dataset(data, args)

    # Train a model for each resampled version of the dataset
    print(f'{"Training" if args.retrain_models else "Retrieving"} bootstrapped models ({args.time_budget_mins} min)')
    models = train_resampled_models(data, resample_idxs, args)

    #  exiting early if not doing model projection
    if not args.run_model_proj:
        print('\nSkipping model projection')
        sys.exit()

    # Read the predictor stack + training data to normalize vars
    print('\nReading predictor stack')
    mammals_data = pd.read_csv(args.mammals_data_fp).reset_index(drop = True)
    predictor_stack = read_predictor_stack(args.predictor_stack_fp, args.model_to_use, mammals_data, 
                                           args.apply_standardization)

    # Reading in the tropical mammal body mass data
    tropical_mammals = pd.read_csv(args.tropical_mammals_fp)

    # Calculate resampled AOH stats using the resampled models + save to CSV
    resampled_aoh_stats = calculate_resampled_aoh_stats(args, models, tropical_mammals, predictor_stack, 
                                                        mammals_data)
    resampled_aoh_stats.to_csv(os.path.join(args.hunting_preds_dir, f'effective_aoh_info_RESAMPLED_{params["model_to_use"]}{"_just-tropical-forest" if args.just_tropical_forest else ""}{"_no-increase" if args.no_increase else ""}{"_hybrid" if args.hybrid_hab_map else ""}.csv'), index = False)

if __name__ == '__main__':
    # Read in parameters
    with open('experiments/study_bootstrap.json', 'r') as f:
        params = json.load(f)

    # Choosing either "local" or "remote"
    mode = 'local'
    print(f'Running in {mode} mode\n')

    main(params, mode)