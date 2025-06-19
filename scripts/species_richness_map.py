import sys
import os
import shutil
import json
import time

sys.path.append('..')

from joblib import Parallel, delayed

import pandas as pd
import numpy as np

import rioxarray as rxr
import geopandas as gpd

def process_one_species_chunk(species_chunk, chunk_num, template_raster, aoh_dir, cache_dir):
    # Make a deep copy of the template for this chunk's species richness map
    spp_richness = template_raster.copy(deep = True)

    # Run across all species in the chunk
    for sp in species_chunk:
        aoh = rxr.open_rasterio(os.path.join(aoh_dir, f'{sp}_RESIDENT.tif'))
        aoh = aoh.rio.reproject_match(spp_richness).fillna(0) # reproject to match template exactly
        aoh = (aoh > 0).astype(int) # turn into a binary AOH map

        spp_richness = spp_richness + aoh # add to running species richness map
    
    # Save species richness sub-raster for this chunk
    spp_richness.rio.to_raster(os.path.join(cache_dir, f'{chunk_num}_species_richness.tif'))

def main(params, mode):
    # Parsing parameters passed in via JSON
    iucn_ids = params['iucn_id_subset']
    num_chunks = params['num_chunks']
    num_cores = params['num_cores']

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

    # Chunking species into groups for parallel processing
    species_chunks = np.array_split(np.array(iucn_ids), indices_or_sections = num_chunks)

    # Reading in the template raster (extent of tropical forest zone, resolution + projection of AOHs)
    template_raster = rxr.open_rasterio(template_raster_fp)

    # Creating the directory for the chunk intermediate cache
    cache_dir = os.path.join(os.path.dirname(save_fp), 'species_richness_cache')

    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.mkdir(cache_dir)

    # Process the species chunks in parallel
    print(f'Aggregating across {len(iucn_ids)} species')
    start = time.time()

    Parallel(n_jobs = num_cores, verbose = 10)(delayed(process_one_species_chunk)(species_chunk, i, template_raster, 
                                                                                  aoh_dir, cache_dir) for i, species_chunk in enumerate(species_chunks))

    # Aggregating all sub-rasters from chunks
    for i, _ in enumerate(species_chunks):
        spp_richness = rxr.open_rasterio(os.path.join(cache_dir, f'{i}_species_richness.tif'))
        template_raster = template_raster + spp_richness

    print(f'Processing time: {time.time() - start}')

    #  cleanup: removing the cache directory
    shutil.rmtree(cache_dir)

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
    mode = 'remote'
    print(f'Running in {mode} mode\n')

    main(params, mode)
