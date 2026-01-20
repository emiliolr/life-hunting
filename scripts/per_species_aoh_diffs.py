import sys
import os
import warnings
import json

sys.path.append('..')
warnings.simplefilter(action = 'ignore', category = FutureWarning)
warnings.simplefilter(action = 'ignore', category = UserWarning)
os.environ['PYTHONWARNINGS'] = 'ignore::FutureWarning,ignore::UserWarning' # to be sure warnings don't pop back up in parallel processes...

from joblib import Parallel, delayed

import pandas as pd
import numpy as np

import rioxarray as rxr
import xarray as xr
import geopandas as gpd

def collect_aoh_info_one_species(species, current_aoh_dir, human_absent_aoh_dir, hunting_preds_dir, model_to_use, 
                                 no_increase, hybrid_hab_map, tropical_zone):
    # Reading in the four needed rasters: (1) human-absent AOH, (2) current AOH, (3 + 4) hunting pressure maps
    human_absent_aoh = rxr.open_rasterio(os.path.join(human_absent_aoh_dir, f'{species}_RESIDENT.tif'))
    human_absent_hp = rxr.open_rasterio(os.path.join(hunting_preds_dir, 'human_absent' + ('_hybrid' if hybrid_hab_map else ''), f'{species}_hunting_pred_{model_to_use}.tif'))

    current_aoh = rxr.open_rasterio(os.path.join(current_aoh_dir, f'{species}_RESIDENT.tif'))
    current_hp = rxr.open_rasterio(os.path.join(hunting_preds_dir, 'current' + ('_hybrid' if hybrid_hab_map else ''), f'{species}_hunting_pred_{model_to_use}.tif'))

    #  optionally, limiting to just tropical forest portions of AOH
    if tropical_zone is not None:
        human_absent_aoh = human_absent_aoh.rio.clip(tropical_zone, all_touched = True).fillna(0) # making sure to set NAs back to 0
        current_aoh = current_aoh.rio.clip(tropical_zone, all_touched = True).fillna(0)

    #  optionally, capping RRs at 1 (no change)
    if no_increase:
        current_hp = current_hp.clip(max = 1)
        human_absent_hp = human_absent_hp.clip(max = 1)

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

def main(params, mode):
    # Parsing parameters passed in via JSON
    model_to_use = params['model_to_use']
    iucn_ids = params['iucn_id_subset']
    num_cores = params['num_cores']
    no_increase = params['no_increase']
    hybrid_hab_map = bool(params['hybrid_hab_map'])
    just_tropical_forest = bool(params['just_tropical_forest'])

    #  file paths
    filepaths = params['filepaths'][mode]

    tropical_mammals_fp = filepaths['tropical_mammals_fp']
    tropical_zone_fp = filepaths['tropical_zone_fp']

    hunting_preds_dir = filepaths['hunting_preds_dir']
    current_aoh_dir = filepaths['current_aoh_dir'] % (filepaths['hybrid_dir'] if hybrid_hab_map else filepaths['non_hybrid_dir'])
    human_absent_aoh_dir = filepaths['human_absent_aoh_dir'] % (filepaths['hybrid_dir'] if hybrid_hab_map else filepaths['non_hybrid_dir'])

    # Reading in the tropical mammal body mass data
    tropical_mammals = pd.read_csv(tropical_mammals_fp)

    #  grabbing the subset of IDs to run over
    if isinstance(iucn_ids, list):
        if len(iucn_ids) == 0:
            iucn_ids = tropical_mammals['iucn_id'].to_list()
    elif isinstance(iucn_ids, int):
        iucn_ids = tropical_mammals['iucn_id'].iloc[ : iucn_ids].to_list()

    # Reading in the AOH percent overlap file to filter out species
    aoh_overlap_current = pd.read_csv(os.path.join(hunting_preds_dir, 'current' + ('_hybrid' if hybrid_hab_map else ''), 'tropical_mammals_aoh_overlap.csv'))
    aoh_overlap_human_absent = pd.read_csv(os.path.join(hunting_preds_dir, 'human_absent' + ('_hybrid' if hybrid_hab_map else ''), 'tropical_mammals_aoh_overlap.csv'))

    filtered_iucn_ids = []
    for sp in iucn_ids:
        pct_overlap_current = aoh_overlap_current.loc[aoh_overlap_current['iucn_id'] == sp, 'aoh_pct_overlap'].iloc[0]
        pct_overlap_human_absent = aoh_overlap_human_absent.loc[aoh_overlap_human_absent['iucn_id'] == sp, 'aoh_pct_overlap'].iloc[0]
        
        if (pct_overlap_current > 0) and (pct_overlap_human_absent > 0):
            filtered_iucn_ids.append(sp)

    # Reading the tropical forest extent polygon for masking non-forest pixels
    print(f'Computing differences {"across full" if not just_tropical_forest else "within tropical forest"} AOH\n')

    if just_tropical_forest:
        tropical_zone = gpd.read_file(tropical_zone_fp)
        tropical_zone = [tropical_zone.geometry.iloc[0]]
    else:
        tropical_zone = None

    # Collecting AOH info for each species in parallel:
    #  1. Human-absent AOH,
    #  2. Current AOH (human-absent + habitat loss),
    #  3. Human-absent AOH + hunting, and
    #  4. Current AOH + hunting
    aoh_dicts = Parallel(n_jobs = num_cores, verbose = 10)(delayed(collect_aoh_info_one_species)(sp, 
                                                                                                 current_aoh_dir, 
                                                                                                 human_absent_aoh_dir, 
                                                                                                 hunting_preds_dir, 
                                                                                                 model_to_use,
                                                                                                 no_increase,
                                                                                                 hybrid_hab_map, 
                                                                                                 tropical_zone) for sp in filtered_iucn_ids)

    # Saving the data frame containing different bits of AOH info
    aoh_info_df = pd.DataFrame(aoh_dicts)
    aoh_info_df.to_csv(os.path.join(hunting_preds_dir, f'effective_aoh_info_{model_to_use}{"_just-tropical-forest"}{"_no-increase" if no_increase else ''}{"_hybrid" if hybrid_hab_map else ""}.csv'), index = False)

if __name__ == '__main__':
    # Read in parameters
    with open('experiments/per_species_aoh_diffs.json', 'r') as f:
        params = json.load(f)

    # Choosing either "local" or "remote"
    mode = 'local'
    print(f'Running in {mode} mode\n')

    main(params, mode)
