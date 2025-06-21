import sys
import os
import json
import time

sys.path.append('..')

import pandas as pd
import geopandas as gpd

import rioxarray as rxr
import xarray as xr

def main(params, mode):
    # Parsing parameters passed in via JSON
    iucn_ids = params['iucn_id_subset']
    map_type = params['map_type']
    model_to_use = params['model_to_use']

    assert map_type in ['species_richness', 'hunting_pressure'], f'{map_type} not currently supported'

    #  file paths
    filepaths = params['filepaths'][mode]

    tropical_mammals_fp = filepaths['tropical_mammals_fp']
    template_raster_fp = filepaths['template_raster_fp']
    tropical_zone_fp = filepaths['tropical_zone_fp']
    aoh_dir = filepaths['aoh_dir']
    hunting_preds_dir = filepaths['hunting_preds_dir']
    pred_stack_fp = filepaths['pred_stack_fp']

    if map_type == 'species_richness':
        save_fp = os.path.join(filepaths['save_dir'], 'tropical_species_richness_map.tif')
    elif map_type == 'hunting_pressure':
        save_fp = os.path.join(filepaths['save_dir'], f'tropical_species_aggregate_hunting_pressure_{model_to_use}.tif')

    # Reading in the tropical mammal data
    tropical_mammals = pd.read_csv(tropical_mammals_fp)

    #  grabbing the subset of IDs to run over
    if isinstance(iucn_ids, list):
        if len(iucn_ids) == 0:
            iucn_ids = tropical_mammals['iucn_id'].to_list()
    elif isinstance(iucn_ids, int):
        iucn_ids = tropical_mammals['iucn_id'].iloc[ : iucn_ids].to_list()

    # Reading in the AOH percent overlap file to filter out species
    aoh_overlap_current = pd.read_csv(os.path.join(hunting_preds_dir, 'current', 'tropical_mammals_aoh_overlap.csv'))

    filtered_iucn_ids = []
    for sp in iucn_ids:
        pct_overlap_current = aoh_overlap_current.loc[aoh_overlap_current['iucn_id'] == sp, 'aoh_pct_overlap'].iloc[0]
        
        if pct_overlap_current > 0:
            filtered_iucn_ids.append(sp)

    # Reading in the template raster (extent of tropical forest zone, resolution + projection of AOHs)
    template_raster = rxr.open_rasterio(template_raster_fp)

    # Iteratively processing AOHs/hunting pressure maps for each tropical species
    print(f'Aggregating across {len(filtered_iucn_ids)} species')
    start = time.time()

    for sp in filtered_iucn_ids:
        if map_type == 'species_richness':
            sp_fp = os.path.join(aoh_dir, f'{sp}_RESIDENT.tif')
        elif map_type == 'hunting_pressure':
            sp_fp = os.path.join(hunting_preds_dir, 'current', f'{sp}_hunting_pred_{model_to_use}.tif')

        sp_raster = rxr.open_rasterio(sp_fp)
        sp_raster = sp_raster.rio.reproject_match(template_raster).fillna(0) # reproject to match template exactly

        #  turn into a binary AOH map, only for species richness
        if map_type == 'species_richness':
            sp_raster = (sp_raster > 0).astype(int)

        template_raster = template_raster + sp_raster # add to running aggregated raster

    print(f'    processing time: {time.time() - start}')

    # Cropping the aggregated raster to the forest zone polygon boundaries
    tropical_zone = gpd.read_file(tropical_zone_fp)
    tropical_zone = [tropical_zone.geometry.iloc[0]]
    agg_raster = template_raster.rio.clip(tropical_zone, all_touched = True)

    if map_type == 'species_richness':
        agg_raster = agg_raster.fillna(0)

    # If doing hunting pressure, divide through by the number of species per cell to get a mean RR
    if map_type == 'hunting_pressure':
        spp_richness = rxr.open_rasterio(os.path.join(filepaths['save_dir'], 'tropical_species_richness_map.tif'))
        spp_richness = spp_richness.where(spp_richness != 0) # ensure no divide by 0
        agg_raster = agg_raster / spp_richness

    #  also, mask by nan mask from predictor raster to remove artifacts where there shouldn't be
    #   any predictions
    if map_type == 'hunting_pressure':
        pred_stack = rxr.open_rasterio(pred_stack_fp)
        pred_stack = xr.ufuncs.isnan(pred_stack).astype(int).sum(dim = 'band') # see where there are nans in any band
        pred_stack = pred_stack.rio.reproject_match(agg_raster)

        agg_raster = agg_raster.where(pred_stack == 0)

    # Saving the final aggregated raster
    dtype = 'uint16' if map_type == 'species_richness' else 'float32'
    agg_raster.rio.to_raster(save_fp, dtype = dtype)

if __name__ == '__main__':
    # Read in parameters
    with open('experiments/spatial_aggregate_maps.json', 'r') as f:
        params = json.load(f)

    # Choosing either "local" or "remote"
    mode = 'local'
    print(f'Running in {mode} mode\n')

    main(params, mode)
