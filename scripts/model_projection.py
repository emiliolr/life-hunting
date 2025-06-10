import pickle
import sys
import os
import warnings

sys.path.append('..')
warnings.simplefilter(action = 'ignore', category = FutureWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

import rioxarray as rxr
import xarray as xr
import geopandas as gpd

def main():
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
    cols_to_normalize.remove('Protected_Area')

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
                 'xgboost' : 'xgboost_hurdle_10.0mins.pkl',
                 'pymer' : 'pymer_hurdle.pkl'}
    model_fp = os.path.join(model_base_path, model_fps[model_to_use])

    with open(model_fp, 'rb') as f:
        model = pickle.load(f)

    # Reading in the tropical mammal body mass data
    tropical_mammals = pd.read_csv(tropical_mammals_fp)
    # iucn_ids = tropical_mammals['iucn_id']

    print()

    # Looping over the tropical mammal species and applying the predictive model
    # TODO: make this parallel
    for species in iucn_ids:
        print(f'Working on {species}...')

        # Get the species' name + body mass
        species_row = tropical_mammals[tropical_mammals['iucn_id'] == species]
        species_name = species_row['scientific_name'].iloc[0]
        species_bm = species_row['combine_body_mass'].iloc[0]

        # Reading in the relevant AOH
        aoh_fp = os.path.join(aoh_base_fp, f'{species}_RESIDENT.tif')
        aoh = rxr.open_rasterio(aoh_fp)

        # Clipping the predictor rasters to the bounds of the AOH
        predictor_stack_clipped = predictor_stack.rio.clip_box(*aoh.rio.bounds())

        #  making sure the predictor stack is perfectly aligned w/AOH
        predictor_stack_clipped = predictor_stack_clipped.rio.reproject_match(aoh)

        # Masking predictions outside of the AOH & tropical forest zone (the intersection of the two)
        aoh_in_forest_zone = aoh.rio.clip(tropical_zone).fillna(0) # making sure to set NAs back to 0

        # initial_num_pixels = int(aoh.where(aoh != 0).count())
        # new_num_pixels = int(aoh_in_forest_zone.where(aoh_in_forest_zone != 0).count())
        # print(f'{initial_num_pixels - new_num_pixels} pixels dropped (in AOH, but outside of tropical forest zone)')

        #  TODO: calculate the area overlap of AOH/tropical forest as a pct of total AOH

        #  applying to the predictor stack
        predictor_stack_clipped = predictor_stack_clipped.where(aoh_in_forest_zone != 0)

        # Extracting the data to numpy + reshaping to get it in a "tabular" format
        predictor_stack_np = predictor_stack_clipped.to_array().variable.values.squeeze()
        num_y, num_x = predictor_stack_np[0].shape
        predictors_tabular = predictor_stack_np.reshape(predictor_stack_np.shape[0], num_y * num_x).transpose()
        # print(f'Originally {predictors_tabular.shape[0]} pixels to predict on (w/NAs)')

        #  tossing nan rows, but keeping track of where they are for reshaping back to raster later
        nan_mask = np.any(np.isnan(predictors_tabular), axis = 1)
        predictors_tabular_no_nan = predictors_tabular[~nan_mask, : ]

        # pixels_left = predictors_tabular_no_nan.shape[0]
        # print(f'Now only {pixels_left} pixels to predict on (removing all NAs)')

        #  checking if we drop any pixels within the AOH
        # if pixels_left < new_num_pixels:
        #     print()
        #     print(f'DROPPED {int(new_num_pixels - pixels_left)} PIXELS!')

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
            # species_name = species_name.replace(' ', '_')
            pred_raster_xr.rio.to_raster(os.path.join(save_dir, f'{species}_hunting_pred_{model_to_use}.tif'), dtype = 'float32')

if __name__ == '__main__':
    # Parameters
    model_to_use = 'pymer' # "pymer" or "rf" or "xgboost"
    save_raster = True

    #  iucn IDs, for initial testing
    iucn_ids = [181007989, 181008073, 7140]

    #  file paths
    life_gdrive_fp = '/Users/emiliolr/Google Drive/My Drive/LIFE/'

    predictor_stack_fp = os.path.join(life_gdrive_fp, 'datasets/derived_datasets/hunting_predictor_stack/hunting_predictor_stack_buffered.tif')
    tropical_zone_fp = os.path.join(life_gdrive_fp, 'datasets/derived_datasets/tropical_forest_extent/tropical_forest_extent.shp')
    aoh_base_fp = '/Users/emiliolr/Desktop/phd-exploratory-work/data/elephants'

    mammals_data_fp = os.path.join(life_gdrive_fp, 'datasets/derived_datasets/benitez_lopez2019_recreated/benitez_lopez2019_recreated_w_original.csv')
    tropical_mammals_fp = os.path.join(life_gdrive_fp, 'datasets/derived_datasets/tropical_species/tropical_mammals_taxonomic_info_w_body_mass.csv')

    model_base_path = os.path.join(life_gdrive_fp, 'hunting_analysis/final_models')

    save_dir = '/Users/emiliolr/Desktop/hunting_testing'

    #  running the projection procedure over the tropical mammal IUCN IDs
    main()
