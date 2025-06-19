import sys
import os
import json
import time

sys.path.append('..')

import pandas as pd
import rioxarray as rxr
import geopandas as gpd

def main(params, mode):
    # Parsing parameters passed in via JSON
    iucn_ids = params['iucn_id_subset']

    #  file paths
    filepaths = params['filepaths'][mode]

    tropical_mammals_fp = filepaths['tropical_mammals_fp']
    template_raster_fp = filepaths['template_raster_fp']
    tropical_zone_fp = filepaths['tropical_zone_fp']
    aoh_dir = filepaths['aoh_dir']

    save_fp = filepaths['save_fp']

    # Reading in the tropical mammal body mass data
    tropical_mammals = pd.read_csv(tropical_mammals_fp)

    #  grabbing the subset of IDs to run over
    if isinstance(iucn_ids, list):
        if len(iucn_ids) == 0:
            iucn_ids = tropical_mammals['iucn_id'].to_list()
    elif isinstance(iucn_ids, int):
        iucn_ids = tropical_mammals['iucn_id'].iloc[ : iucn_ids].to_list()

    # Reading in the template raster (extent of tropical forest zone, resolution + projection of AOHs)
    template_raster = rxr.open_rasterio(template_raster_fp)

    # Iteratively reading in the AOHs for tropical mammals + recording where they fall w/in the
    #  tropical forest zone
    print(f'Aggregating across {len(iucn_ids)} species')
    start = time.time()

    for sp in iucn_ids:
        aoh = rxr.open_rasterio(os.path.join(aoh_dir, f'{sp}_RESIDENT.tif'))
        aoh = aoh.rio.reproject_match(template_raster).fillna(0) # reproject to match template exactly
        aoh = (aoh > 0).astype(int) # turn into a binary AOH map

        template_raster = template_raster + aoh # add to running species richness map

    print(f'Processing time: {time.time() - start}')

    # Cropping the aggregated richness map to the forest zone polygon boundaries
    tropical_zone = gpd.read_file(tropical_zone_fp)
    tropical_zone = [tropical_zone.geometry.iloc[0]]
    species_richness = template_raster.rio.clip(tropical_zone, all_touched = True).fillna(0)

    # Saving the final species richness raster
    species_richness.rio.to_raster(save_fp, dtype = 'uint16')

if __name__ == '__main__':
    # Read in parameters
    with open('experiments/species_richness_map.json', 'r') as f:
        params = json.load(f)

    # Choosing either "local" or "remote"
    mode = 'local'
    print(f'Running in {mode} mode\n')

    main(params, mode)
