import sys
import os
import warnings

sys.path.append('..')
warnings.simplefilter(action = 'ignore', category = FutureWarning)
warnings.simplefilter(action = 'ignore', category = UserWarning)
os.environ['PYTHONWARNINGS'] = 'ignore::FutureWarning,ignore::UserWarning' # to be sure warnings don't pop back up in parallel processes...

from joblib import Parallel, delayed

import pandas as pd
import numpy as np

import rioxarray as rxr
import xarray as xr

def collect_aoh_info_one_species(species, current_aoh_dir, human_absent_aoh_dir, hunting_preds_dir, model_to_use):
    # Reading in the four needed rasters: (1) human-absent AOH, (2) current AOH, (3 + 4) hunting pressure maps
    human_absent_aoh = rxr.open_rasterio(os.path.join(human_absent_aoh_dir, f'{species}_RESIDENT.tif'))
    human_absent_hp = rxr.open_rasterio(os.path.join(hunting_preds_dir, 'human_absent', f'{species}_hunting_pred_{model_to_use}.tif'))

    current_aoh = rxr.open_rasterio(os.path.join(current_aoh_dir, f'{species}_RESIDENT.tif'))
    current_hp = rxr.open_rasterio(os.path.join(hunting_preds_dir, 'current', f'{species}_hunting_pred_{model_to_use}.tif'))

    #  ensure hunting pressure maps align precisely w/respective AOHs
    human_absent_hp = human_absent_hp.rio.reproject_match(human_absent_aoh)
    current_hp = current_hp.rio.reproject_match(current_aoh)

    # Putting RR=1 (no hunting effect) in AOH areas with no predictions for hunting maps
    no_pred_mask = ((human_absent_aoh != 0) & (xr.ufuncs.isnan(human_absent_hp)))
    human_absent_hp = human_absent_hp.where(~no_pred_mask, other = 1)

    no_pred_mask = ((current_aoh != 0) & (xr.ufuncs.isnan(current_hp)))
    current_hp = current_hp.where(~no_pred_mask, other = 1)

    # Getting different needed AOH quantities
    #  first: human-absent + current AOHs
    human_absent_aoh_total = float(human_absent_aoh.sum())
    current_aoh_total = float(current_aoh.sum())

    #  next: element-wise multiplications w/predicted hunting pressure maps to get
    #   scenarios with hunting
    human_absent_aoh_w_hunting = human_absent_aoh * human_absent_hp
    human_absent_aoh_w_hunting_total = float(human_absent_aoh_w_hunting.sum())

    current_aoh_w_hunting = current_aoh * current_hp
    current_aoh_w_hunting_total = float(current_aoh_w_hunting.sum())

    # Packaging everything in a dictionary (a single row for a dataframe)
    return_dict = {'species' : species,
                   'human_absent_aoh_total' : human_absent_aoh_total,
                   'current_aoh_total' : current_aoh_total, 
                   'human_absent_aoh_w_hunting_total' : human_absent_aoh_w_hunting_total,
                   'current_aoh_w_hunting_total' : current_aoh_w_hunting_total}

    return return_dict

def main(tropical_mammals_fp, hunting_preds_dir, current_aoh_dir, human_absent_aoh_dir, iucn_ids, model_to_use):
    # Reading in the tropical mammal body mass data
    tropical_mammals = pd.read_csv(tropical_mammals_fp)

    #  grabbing the subset of IDs to run over
    if isinstance(iucn_ids, list):
        if len(iucn_ids) == 0:
            iucn_ids = tropical_mammals['iucn_id'].to_list()
    elif isinstance(iucn_ids, int):
        iucn_ids = tropical_mammals['iucn_id'].iloc[ : iucn_ids].to_list()

    # Reading in the AOH percent overlap file to filter out species
    aoh_overlap_current = pd.read_csv(os.path.join(hunting_preds_dir, 'current', 'tropical_mammals_aoh_overlap.csv'))
    aoh_overlap_human_absent = pd.read_csv(os.path.join(hunting_preds_dir, 'human_absent', 'tropical_mammals_aoh_overlap.csv'))

    filtered_iucn_ids = []
    for sp in iucn_ids:
        pct_overlap_current = aoh_overlap_current.loc[aoh_overlap_current['iucn_id'] == sp, 'aoh_pct_overlap'].iloc[0]
        pct_overlap_human_absent = aoh_overlap_human_absent.loc[aoh_overlap_human_absent['iucn_id'] == sp, 'aoh_pct_overlap'].iloc[0]
        
        if (pct_overlap_current > 0) and (pct_overlap_human_absent > 0):
            filtered_iucn_ids.append(sp)

    # Collecting AOH info for each species in parallel:
    #  1. Human-absent AOH
    #  2. Current AOH (human-absent + habitat loss)
    #  3. Human-absent AOH + hunting
    #  4. Current AOH + hunting
    aoh_info_df = pd.DataFrame(columns = ['species', 'human_absent_aoh_total', 'current_aoh_total',
                                          'human_absent_aoh_w_hunting_total', 'current_aoh_w_hunting_total'])

    for i, sp in enumerate(filtered_iucn_ids):
        row = collect_aoh_info_one_species(sp, current_aoh_dir, human_absent_aoh_dir, hunting_preds_dir, model_to_use)
        aoh_info_df.loc[i] = row

    # Saving the data frame containing different bits of AOH info
    aoh_info_df.to_csv(os.path.join(hunting_preds_dir, 'effective_aoh_info.csv'), index = False)

if __name__ == '__main__':
    tropical_mammals_fp = '/Users/emiliolr/Google Drive/My Drive/LIFE/datasets/derived_datasets/tropical_species/tropical_mammals_taxonomic_info_w_body_mass.csv'
    iucn_ids = [7140, 181007989, 181008073]
    
    model_to_use = 'pymer'

    hunting_preds_dir = '/Users/emiliolr/Desktop/hunting_testing'
    current_aoh_dir = '/Users/emiliolr/Desktop/phd-exploratory-work/data/elephants'
    human_absent_aoh_dir = '/Users/emiliolr/Desktop/phd-exploratory-work/data/elephants/human_absent'

    main(tropical_mammals_fp, hunting_preds_dir, current_aoh_dir, human_absent_aoh_dir, iucn_ids, model_to_use)