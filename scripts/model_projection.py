import pickle
import sys
import os
import os.path
import warnings
import json

sys.path.append('..')
warnings.simplefilter(action = 'ignore', category = FutureWarning)
warnings.simplefilter(action = 'ignore', category = UserWarning)
os.environ['PYTHONWARNINGS'] = 'ignore::FutureWarning,ignore::UserWarning' # to be sure warnings don't pop back up in parallel processes...

from joblib import Parallel, delayed

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

import rioxarray as rxr
import xarray as xr
import geopandas as gpd

def apply_model_one_species(species, tropical_mammals, predictor_stack, tropical_zone, mammals_data, model,
                            model_to_use, aoh_dir, save_raster, save_dir, error_fp):
    # Get the species' name + body mass
    species_row = tropical_mammals[tropical_mammals['iucn_id'] == species]
    species_bm = species_row['combine_body_mass'].iloc[0]

    # Reading in the relevant AOH
    aoh_fp = os.path.join(aoh_dir, f'{species}_RESIDENT.tif')
    if not os.path.isfile(aoh_fp):
      return species, -2

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
        return species, -2 

    pct_overlap = aoh_in_forest / aoh_total

    #  skip making predictions if there's no overlap w/tropical forest
    if pct_overlap == 0:
        return species, pct_overlap

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
        return species, -1

    # Putting data in a Pandas DataFrame so the predict function of the hurdle model can grab the right vars
    predictors_tabular_no_nan = pd.DataFrame(predictors_tabular_no_nan, columns = list(predictor_stack_clipped.keys()))

    #  adding the same (standardized) body mass value to each row
    if model_to_use == 'pymer':
        bm = np.log10(mammals_data['Body_Mass'])
        bm_mean, bm_std = bm.mean(), bm.std()
        
        species_bm_std = (np.log10(species_bm) - bm_mean) / bm_std
    else:
        bm = mammals_data['Body_Mass']
        bm_mean, bm_std = bm.mean(), bm.std()
        
        species_bm_std = (species_bm - bm_mean) / bm_std

    predictors_tabular_no_nan['Body_Mass'] = species_bm_std

    #  apply the trained hurdle model to each pixel iteratively
    pred = model.predict(predictors_tabular_no_nan)
    if '3part' not in model_to_use:
        pred[pred != 0] = np.exp(pred[pred != 0]) # back-transforming log RRs

    # Putting the dataset all back together in a predicted raster
    pred_tabular = np.empty(shape = predictors_tabular.shape[0])
    pred_tabular.fill(np.nan)
    pred_tabular[~nan_mask] = pred # one prediction for each pixel, w/nans put back in the right place

    #  reshaping back to raster format + converting back to xarray
    pred_raster = pred_tabular.transpose().reshape(num_y, num_x)
    pred_raster_xr = xr.zeros_like(aoh_in_forest_zone)
    pred_raster_xr.values = np.expand_dims(pred_raster, axis = 0)

    # Saving back to TIF using rasterio
    if save_raster:
        pred_raster_xr.rio.to_raster(os.path.join(save_dir, f'{species}_hunting_pred_{model_to_use}.tif'), dtype = 'float32')

    # Returning tuples of (IUCN ID, AOH percent overlap)
    return species, pct_overlap

def main(params, mode):
    # Parsing parameters passed in via JSON
    model_to_use = params['model_to_use'] 
    save_raster = bool(params['save_raster'])

    num_cores = params['num_cores']

    current_aoh = params['current_aoh']

    #  a subset of iucn IDs, for testing (if empty, will loop over all IDs)
    iucn_ids = params['iucn_id_subset']

    #  file paths
    filepaths = params['filepaths'][mode]
    base_fp = filepaths['base_fp']

    predictor_stack_fp = os.path.join(base_fp, filepaths['predictor_stack_fp'])
    if model_to_use == 'rf-gov':
        predictor_stack_fp = predictor_stack_fp.replace('_pca', '')

    tropical_zone_fp = os.path.join(base_fp, filepaths['tropical_zone_fp'])

    mammals_data_fp = os.path.join(base_fp, filepaths['mammals_data_fp'])
    tropical_mammals_fp = os.path.join(base_fp, filepaths['tropical_mammals_fp'])

    model_dir = os.path.join(base_fp, filepaths['model_dir'])

    error_fp = filepaths['error_fp']

    #  either applying to current AOHs or human-absent
    if current_aoh:
        aoh_dir = filepaths['current_aoh_dir']
        save_dir = os.path.join(filepaths['save_dir'], 'current')
    else:
        aoh_dir = filepaths['human_absent_aoh_dir']
        save_dir = os.path.join(filepaths['save_dir'], 'human_absent')

    # Reading in the predictor raster stack
    print('Reading predictor stack')

    predictor_stack = rxr.open_rasterio(predictor_stack_fp, band_as_variable = True)

    #  correcting the variable names
    predictor_stack = predictor_stack.rename({band : predictor_stack[band].attrs['long_name'] for band in predictor_stack})

    # Reading the tropical forest extent polygon for masking non-forest pixels
    tropical_zone = gpd.read_file(tropical_zone_fp)
    tropical_zone = [tropical_zone.geometry.iloc[0]]

    # Reading the full mammal_recreated dataset for z-score stats
    mammals_data = pd.read_csv(mammals_data_fp)

    #  columns to use
    cols_to_normalize = list(predictor_stack.keys())
    cols_to_normalize = [c for c in cols_to_normalize if (c not in 'Protected_Area') and (not c.startswith('IUCN_Country_Region'))]
    
    gov_vars = ['Corruption', 'Government_Effectiveness', 'Political_Stability', 'Regulation', 'Rule_of_Law', 
                'Accountability', 'PC_0', 'PC_1']
    if model_to_use == 'pymer':
        cols_to_normalize = [c for c in cols_to_normalize if c not in gov_vars]

    #  extract columns means + standard deviations
    mammals_cols_to_normalize = mammals_data[cols_to_normalize]
    if model_to_use == 'pymer':
        mammals_cols_to_normalize = mammals_cols_to_normalize.replace(0, 0.1)
        mammals_cols_to_normalize = np.log10(mammals_cols_to_normalize)
    
    col_means = mammals_cols_to_normalize.mean(axis = 0)
    col_stds = mammals_cols_to_normalize.std(axis = 0)

    # Applying data preprocessing to predictor rasters
    print('Normalizing predictors')

    for pred in cols_to_normalize:
        temp = predictor_stack[pred]
        
        if pred == 'Dist_Settlement_KM':
            temp = temp / 1000 # converting to actual km

        #  log10-transforming continuous vars (just linear hurdle)
        if model_to_use == 'pymer':
            temp = temp.where(temp != 0, other = 0.1) # making sure there aren't issues w/taking the log
            temp = xr.ufuncs.log10(temp)
        
        #  z-score normalization
        temp = temp - col_means[pred]
        temp = temp / col_stds[pred]

        #  slotting the preprocessed version back into the dataset
        predictor_stack[pred] = temp

    # Reading in the saved predictive model
    print('Reading saved model')
    
    model_fps = {'rf' : 'rf_hurdle_10.0mins.pkl',
                 'rf-gov' : 'rf-gov_hurdle_10.0mins.pkl',
                 'rf-pca' : 'rf-pca_hurdle_10.0mins.pkl',
                 'xgboost' : 'xgboost_hurdle_10.0mins.pkl',
                 'pymer' : 'pymer_hurdle.pkl', 
                 'xgboost-3part' : 'xgboost_three_part_10.0mins.pkl'}
    model_fp = os.path.join(model_dir, model_fps[model_to_use])
    
    with open(model_fp, 'rb') as f:
        model = pickle.load(f)

    # Reading in the tropical mammal body mass data
    tropical_mammals = pd.read_csv(tropical_mammals_fp)

    #  grabbing the subset of IDs to run over
    if isinstance(iucn_ids, list):
        if len(iucn_ids) == 0:
            iucn_ids = tropical_mammals['iucn_id'].to_list()
    elif isinstance(iucn_ids, int):
        iucn_ids = tropical_mammals['iucn_id'].iloc[ : iucn_ids].to_list()

    # Looping over the tropical mammal species and applying the predictive model IN PARALLEL
    print('Making hunting predictions\n')
    aoh_pct_overlap = Parallel(n_jobs = num_cores, verbose = 10)(delayed(apply_model_one_species)(species, 
                                                                                                  tropical_mammals, 
                                                                                                  predictor_stack, 
                                                                                                  tropical_zone, 
                                                                                                  mammals_data, 
                                                                                                  model,
                                                                                                  model_to_use,
                                                                                                  aoh_dir,
                                                                                                  save_raster,
                                                                                                  save_dir, 
                                                                                                  error_fp) for species in iucn_ids)
    
    # Adding the percent overlap stats to the tropical mammal dataset + saving
    aoh_pct_overlap = pd.DataFrame(aoh_pct_overlap, columns = ['iucn_id', 'aoh_pct_overlap'])
    aoh_pct_overlap = tropical_mammals[['iucn_id']].merge(aoh_pct_overlap, on = 'iucn_id', how = 'left')
    aoh_pct_overlap.to_csv(os.path.join(save_dir, 'tropical_mammals_aoh_overlap.csv'), index = False)

if __name__ == '__main__':
    # Read in parameters
    with open('experiments/model_projection.json', 'r') as f:
        params = json.load(f)

    # Choosing either "local" or "remote"
    mode = 'local'
    print(f'Running in {mode} mode\n')

    #  running the projection procedure over the tropical mammal IUCN IDs
    main(params, mode)
