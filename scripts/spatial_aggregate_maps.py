import sys
import os
import json
import time

sys.path.append('..')

from tqdm import tqdm

import pandas as pd
import geopandas as gpd
import numpy as np

import rioxarray as rxr
import xarray as xr

def shapley_ish_value(aoh_hum_abs, aoh_interest, aoh_other, aoh_joint):
    return (1 / 2) * ((aoh_interest - aoh_hum_abs) + (aoh_joint - aoh_other))

def bm_cats(bm):
    if bm < 0.1:
        return 'very small'
    elif (bm >= 0.1) and (bm < 1):
        return 'small'
    elif (bm >= 1) and (bm < 10):
        return 'medium'
    else:
        return 'large'

def aoh_extinction_curve(aoh_hum_abs, aoh_new, curve_type = 'power_law', z = 0.25, a = 2.5, b = -14.5, 
                         alpha = 1, np_careful = False):
    assert curve_type in ['power_law', 'gompertz'], f'Curve type "{curve_type}" not supported'

    #  everything here v carefully uses Float64 for operations, to avoid errors w/small numbers
    if curve_type == 'power_law':
        if np_careful:
            prob_extinction = np.subtract(np.float64(1), np.float_power(np.divide(aoh_new, aoh_hum_abs), z))
        else:
            prob_extinction = 1 - ((aoh_new / aoh_hum_abs) ** z)
    elif curve_type == 'gompertz':
        if np_careful:
            prob_extinction = np.subtract(np.float64(1), np.exp(np.multiply(np.float64(-1), np.exp(np.add(np.float64(a), np.multiply(np.float64(b), np.float_power(np.divide(aoh_new, aoh_hum_abs), alpha)))))))
        else:
            prob_extinction = 1 - (np.exp(-1 * np.exp(a + b * ((aoh_new / aoh_hum_abs) ** alpha))))

    #  clip values to 0-1 range so they're probabilities
    prob_extinction = np.clip(prob_extinction, a_min = 0, a_max = 1)

    return prob_extinction

def main(params, mode):
    # Parsing parameters passed in via JSON
    iucn_ids = params['iucn_id_subset']
    map_type = params['map_type']
    model_to_use = params['model_to_use']
    current = bool(params['current'])
    no_increase = bool(params['no_increase'])
    hybrid_hab_map = bool(params['hybrid_hab_map'])

    facet_body_mass = bool(params['facet_body_mass'])
    if facet_body_mass:
        assert map_type in ['hunting_pressure', 'species_richness'], f'Faceting by body mass not supported for map type {map_type}.'

    valid_map_types = ['species_richness', 'hunting_pressure', 'joint_aoh_effect', 'partial_aoh_effects', 'restore_and_abate']
    assert map_type in valid_map_types, f'{map_type} not currently supported.'

    #  file paths
    filepaths = params['filepaths'][mode]

    tropical_mammals_fp = filepaths['tropical_mammals_fp']
    template_raster_fp = filepaths['template_raster_fp']
    tropical_zone_fp = filepaths['tropical_zone_fp']

    cur_aoh_dir = os.path.join(filepaths['aoh_dir'] % (filepaths['hybrid_dir'] if hybrid_hab_map else filepaths['non_hybrid_dir']), 'current')
    hum_abs_aoh_dir = os.path.join(filepaths['aoh_dir'] % (filepaths['hybrid_dir'] if hybrid_hab_map else filepaths['non_hybrid_dir']), 'pnv')

    if mode == 'remote':
        cur_aoh_dir += '/MAMMALIA'
        hum_abs_aoh_dir += '/MAMMALIA'

    hunting_preds_dir = filepaths['hunting_preds_dir']
    pred_stack_fp = filepaths['pred_stack_fp']
    if model_to_use == 'rf-gov':
        predictor_stack_fp = predictor_stack_fp.replace('_pca', '')

    save_dir = filepaths['save_dir'] if not facet_body_mass else os.path.join(filepaths['save_dir'], 'body_mass_facet')
    if map_type == 'species_richness':
        save_fp = os.path.join(save_dir, f'tropical_species_richness_map_{"current" if current else "human_absent"}%s{"_hybrid" if hybrid_hab_map else ""}.tif')
    elif map_type == 'hunting_pressure':
        save_fp = os.path.join(save_dir, f'tropical_species_aggregate_hunting_pressure_{model_to_use}%s{"_no-increase" if no_increase else ""}{"_hybrid" if hybrid_hab_map else ""}.tif')
    elif map_type == 'joint_aoh_effect':
        save_fp = os.path.join(save_dir, f'tropical_species_aggregate_joint_effect_{model_to_use}%s{"_no-increase" if no_increase else ""}{"_hybrid" if hybrid_hab_map else ""}.tif')
    elif map_type == 'partial_aoh_effects':
        save_fps = {'hunting' : os.path.join(save_dir, f'tropical_species_aggregate_partial_hunting_effect_{model_to_use}%s{"_no-increase" if no_increase else ""}{"_hybrid" if hybrid_hab_map else ""}.tif'),
                    'habitat_loss' : os.path.join(save_dir, f'tropical_species_aggregate_partial_hab_loss_effect_{model_to_use}%s{"_no-increase" if no_increase else ""}{"_hybrid" if hybrid_hab_map else ""}.tif')}
    elif map_type == 'restore_and_abate':
        save_fps = {'restore_and_abate' : os.path.join(save_dir, f'tropical_species_aggregate_restore_and_abate_{model_to_use}%s{"_no-increase" if no_increase else ""}{"_hybrid" if hybrid_hab_map else ""}.tif'),
                    'restore' : os.path.join(save_dir, f'tropical_species_aggregate_restore_{model_to_use}%s{"_no-increase" if no_increase else ""}{"_hybrid" if hybrid_hab_map else ""}.tif'),
                    'abate' : os.path.join(save_dir, f'tropical_species_aggregate_abate_{model_to_use}%s{"_no-increase" if no_increase else ""}{"_hybrid" if hybrid_hab_map else ""}.tif'),
                    'restore_baseline' : os.path.join(save_dir, f'tropical_species_aggregate_restore_baseline_{model_to_use}%s{"_no-increase" if no_increase else ""}{"_hybrid" if hybrid_hab_map else ""}.tif')}

    # Reading in the tropical mammal data
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

    # Get body mass categories, if grouping by body mass
    if facet_body_mass:
        tropical_mammals['body_mass_cat'] = (tropical_mammals['combine_body_mass'] / 1000).apply(bm_cats)
        
        body_mass_cats = []
        for sp in filtered_iucn_ids:
            bm_cat = tropical_mammals[tropical_mammals['iucn_id'] == sp]['body_mass_cat'].iloc[0]
            body_mass_cats.append(bm_cat)

    # Reading in the template raster (extent of tropical forest zone, resolution + projection of AOHs)
    if map_type in ['species_richness', 'hunting_pressure', 'joint_aoh_effect']:
        if not facet_body_mass:
            template_raster = rxr.open_rasterio(template_raster_fp)
        else:
            templates = {bm_cat : rxr.open_rasterio(template_raster_fp) for bm_cat in np.unique(body_mass_cats)}
    elif map_type == 'partial_aoh_effects':
        template_raster_hunt, template_raster_hab = rxr.open_rasterio(template_raster_fp), rxr.open_rasterio(template_raster_fp)
    elif map_type == 'restore_and_abate':
        template_raster_abate = rxr.open_rasterio(template_raster_fp).astype('float64')
        template_raster_restore = rxr.open_rasterio(template_raster_fp).astype('float64')
        template_raster_restore_baseline = rxr.open_rasterio(template_raster_fp).astype('float64')
        template_raster_abate_restore = rxr.open_rasterio(template_raster_fp).astype('float64')

    # Iteratively processing AOHs/hunting pressure maps for each tropical species
    print(f'Aggregating across {len(filtered_iucn_ids)} species for map type "{map_type}"' + (f' ({"current" if current else "human_absent"})' if map_type == 'species_richness' else ''))
    if facet_body_mass:
        print('  faceting by body mass categories')

    for i in tqdm(range(len(filtered_iucn_ids))):
        sp = filtered_iucn_ids[i]
        bm_cat = body_mass_cats[i] if facet_body_mass else None

        if map_type in ['species_richness', 'hunting_pressure']:
            if map_type == 'species_richness':
                sp_fp = os.path.join(cur_aoh_dir if current else hum_abs_aoh_dir, f'{sp}_RESIDENT.tif')
            elif map_type == 'hunting_pressure':
                sp_fp = os.path.join(hunting_preds_dir, 'current' + ('_hybrid' if hybrid_hab_map else ''), f'{sp}_hunting_pred_{model_to_use}.tif')

            sp_raster = rxr.open_rasterio(sp_fp)
        elif map_type in ['joint_aoh_effect', 'partial_aoh_effects', 'restore_and_abate']:
            #  read in needed rasters
            cur_aoh_fp = os.path.join(cur_aoh_dir, f'{sp}_RESIDENT.tif')
            hum_abs_aoh_fp = os.path.join(hum_abs_aoh_dir, f'{sp}_RESIDENT.tif')
            hp_cur_fp = os.path.join(hunting_preds_dir, 'current' + ('_hybrid' if hybrid_hab_map else ''), f'{sp}_hunting_pred_{model_to_use}.tif')
            hp_abs_fp = os.path.join(hunting_preds_dir, 'human_absent' + ('_hybrid' if hybrid_hab_map else ''), f'{sp}_hunting_pred_{model_to_use}.tif')

            cur_aoh = rxr.open_rasterio(cur_aoh_fp)
            hum_abs_aoh = rxr.open_rasterio(hum_abs_aoh_fp)
            hp_cur = rxr.open_rasterio(hp_cur_fp)
            hp_abs = rxr.open_rasterio(hp_abs_fp)

            #  optionally, capping RRs at 1 (no change)
            if no_increase:
                hp_cur = hp_cur.clip(max = 1)
                hp_abs = hp_abs.clip(max = 1)

            #  get joint effect of hunting and habitat loss
            hp_cur = hp_cur.rio.reproject_match(cur_aoh)

            no_pred_mask = ((cur_aoh != 0) & (xr.ufuncs.isnan(hp_cur)))
            hp_cur = hp_cur.where(~no_pred_mask, other = 1)

            effective_aoh_cur = cur_aoh * hp_cur
            effective_aoh_cur = effective_aoh_cur.rio.reproject_match(hum_abs_aoh).fillna(0)

            delta_aoh_tot = effective_aoh_cur - hum_abs_aoh

            #  get the joint delta AOH + represent as a percentage of human-absent AOH
            if map_type == 'joint_aoh_effect':
                sp_raster = delta_aoh_tot.where(hum_abs_aoh != 0) / hum_abs_aoh
            #  partial out the joint effect to hunting + habitat loss based on independent effects
            elif map_type in ['partial_aoh_effects', 'restore_and_abate']:
                #  get AOH effect of just hunting
                hp_abs = hp_abs.rio.reproject_match(hum_abs_aoh)

                no_pred_mask = ((hum_abs_aoh != 0) & (xr.ufuncs.isnan(hp_abs)))
                hp_abs = hp_abs.where(~no_pred_mask, other = 1)

                effective_aoh_abs = hum_abs_aoh * hp_abs

                #  align everything with human absent AOH so we can do cell-wise operations
                effective_aoh_abs = effective_aoh_abs.rio.reproject_match(hum_abs_aoh).fillna(0)
                cur_aoh = cur_aoh.rio.reproject_match(hum_abs_aoh).fillna(0)

                if map_type == 'partial_aoh_effects':
                    #  get partial effects using shapley(-ish) values
                    sp_raster_hunt = shapley_ish_value(hum_abs_aoh, effective_aoh_abs, cur_aoh, effective_aoh_cur)
                    sp_raster_hab = shapley_ish_value(hum_abs_aoh, cur_aoh, effective_aoh_abs, effective_aoh_cur)

                    #  divide through by the total delta AOH (summed across AOH)
                    sp_raster_hunt = sp_raster_hunt.where(hum_abs_aoh != 0) / hum_abs_aoh
                    sp_raster_hab = sp_raster_hab.where(hum_abs_aoh != 0) / hum_abs_aoh
                elif map_type == 'restore_and_abate':
                    #  cast everything to Float64 to ensure precision in operations & make sure that when we 
                    #   take the difference, comparisons are made everywhere (doesn't happen if one layers has null values 
                    #   but not the other)
                    hum_abs_aoh = hum_abs_aoh.astype('float64').fillna(0)
                    cur_aoh = cur_aoh.astype('float64').fillna(0)
                    effective_aoh_abs = effective_aoh_abs.astype('float64').fillna(0)
                    effective_aoh_cur = effective_aoh_cur.astype('float64').fillna(0)

                    #  the delta AOH in each pixel
                    delta_aoh_tot = hum_abs_aoh - effective_aoh_cur
                    delta_aoh_restore = effective_aoh_abs - effective_aoh_cur
                    delta_aoh_abate = cur_aoh - effective_aoh_cur
                    delta_aoh_restore_baseline = hum_abs_aoh - cur_aoh

                    #  getting needed total AOHs
                    hum_abs_aoh_tot = hum_abs_aoh.sum()
                    cur_aoh_tot = cur_aoh.sum()
                    effective_aoh_cur_tot = effective_aoh_cur.sum()
                    p0 = aoh_extinction_curve(hum_abs_aoh_tot, effective_aoh_cur_tot, np_careful = True)

                    #  the effect when you restore/abate in each pixel
                    abate_restore_aoh_tot = effective_aoh_cur_tot + delta_aoh_tot
                    p1 = aoh_extinction_curve(hum_abs_aoh_tot, abate_restore_aoh_tot, np_careful = True)
                    sp_raster_abate_restore = p1 - p0

                    #  the effect of JUST abating in each pixel
                    abate_aoh_tot = effective_aoh_cur_tot + delta_aoh_abate
                    p1 = aoh_extinction_curve(hum_abs_aoh_tot, abate_aoh_tot, np_careful = True)
                    sp_raster_abate = p1 - p0

                    #  the effect of JUST restoring in each pixel
                    restore_aoh_tot = effective_aoh_cur_tot + delta_aoh_restore
                    p1 = aoh_extinction_curve(hum_abs_aoh_tot, restore_aoh_tot, np_careful = True)
                    sp_raster_restore = p1 - p0

                    #  the effect of JUST restoring in a world w/o hunting, as a baseline
                    restore_baseline_aoh_tot = cur_aoh_tot + delta_aoh_restore_baseline
                    p0 = aoh_extinction_curve(hum_abs_aoh_tot, cur_aoh_tot, np_careful = True)
                    p1 = aoh_extinction_curve(hum_abs_aoh_tot, restore_baseline_aoh_tot, np_careful = True)
                    sp_raster_restore_baseline = p1 - p0

        if map_type not in ['partial_aoh_effects', 'restore_and_abate']:
            sp_raster = sp_raster.rio.reproject_match(template_raster if not facet_body_mass else templates[bm_cat]).fillna(0) # reproject to match template exactly

            #  turn into a binary AOH map, only for species richness
            if map_type == 'species_richness':
                sp_raster = (sp_raster > 0).astype(int)

            #  optionally, capping RRs at 1 (no change)
            if no_increase and map_type == 'hunting_pressure':
                sp_raster = sp_raster.clip(max = 1)

            if not facet_body_mass:
                template_raster = template_raster + sp_raster # add to running aggregated raster
            else:
                templates[bm_cat] = templates[bm_cat] + sp_raster # add to raster of the body mass category
        elif map_type == 'partial_aoh_effects':
            sp_raster_hunt = sp_raster_hunt.rio.reproject_match(template_raster_hunt).fillna(0)
            template_raster_hunt = template_raster_hunt + sp_raster_hunt

            sp_raster_hab = sp_raster_hab.rio.reproject_match(template_raster_hab).fillna(0)
            template_raster_hab = template_raster_hab + sp_raster_hab
        elif map_type == 'restore_and_abate':
            sp_raster_abate_restore = sp_raster_abate_restore.rio.reproject_match(template_raster_abate_restore).fillna(0)
            template_raster_abate_restore = template_raster_abate_restore + sp_raster_abate_restore

            sp_raster_abate = sp_raster_abate.rio.reproject_match(template_raster_abate).fillna(0)
            template_raster_abate = template_raster_abate + sp_raster_abate

            sp_raster_restore = sp_raster_restore.rio.reproject_match(template_raster_restore).fillna(0)
            template_raster_restore = template_raster_restore + sp_raster_restore

            sp_raster_restore_baseline = sp_raster_restore_baseline.rio.reproject_match(template_raster_restore_baseline).fillna(0)
            template_raster_restore_baseline = template_raster_restore_baseline + sp_raster_restore_baseline

    print()
    print('Applying postprocessing')

    # Cropping the aggregated raster to the forest zone polygon boundaries
    tropical_zone = gpd.read_file(tropical_zone_fp)
    tropical_zone = [tropical_zone.geometry.iloc[0]]

    if map_type not in ['partial_aoh_effects', 'restore_and_abate']:
        if not facet_body_mass:
            agg_raster = template_raster.rio.clip(tropical_zone, all_touched = True)
        else:
            for k in templates.keys():
                templates[k] = templates[k].rio.clip(tropical_zone, all_touched = True)
    elif map_type == 'partial_aoh_effects':
        agg_raster_hunt = template_raster_hunt.rio.clip(tropical_zone, all_touched = True)
        agg_raster_hab = template_raster_hab.rio.clip(tropical_zone, all_touched = True)
    elif map_type == 'restore_and_abate':
        agg_raster_abate_restore = template_raster_abate_restore.rio.clip(tropical_zone, all_touched = True)
        agg_raster_abate = template_raster_abate.rio.clip(tropical_zone, all_touched = True)
        agg_raster_restore = template_raster_restore.rio.clip(tropical_zone, all_touched = True)
        agg_raster_restore_baseline = template_raster_restore_baseline.rio.clip(tropical_zone, all_touched = True)

    if map_type == 'species_richness':
        if not facet_body_mass:
            agg_raster = agg_raster.fillna(0)
        else:
            for k in templates.keys():
                templates[k] = templates[k].fillna(0)

    # Divide through by the number of species per cell to get a mean RR (or change in AOH)
    if map_type in ['hunting_pressure', 'joint_aoh_effect', 'partial_aoh_effects']:
        if not facet_body_mass:
            spp_richness = rxr.open_rasterio(os.path.join(save_dir, f'tropical_species_richness_map_{"current" if map_type == "hunting_pressure" else "human_absent"}{"_hybrid" if hybrid_hab_map else ""}.tif'))
            spp_richness = spp_richness.where(spp_richness != 0) # ensure no divide by 0
        else:
            spp_richness_cats = {}
            for k in templates.keys():
                sr_cat = rxr.open_rasterio(os.path.join(save_dir, f'tropical_species_richness_map_current_{k.replace(" ", "_")}{"_hybrid" if hybrid_hab_map else ""}.tif'))
                spp_richness_cats[k] = sr_cat.where(sr_cat != 0)

        if map_type != 'partial_aoh_effects':
            if not facet_body_mass:
                agg_raster = agg_raster / spp_richness
            else:
                for k in templates.keys():
                    templates[k] = templates[k] / spp_richness_cats[k]
        else:
            agg_raster_hunt = agg_raster_hunt / spp_richness
            agg_raster_hab = agg_raster_hab / spp_richness

    # Also, mask by nan mask from predictor raster to remove artifacts where there shouldn't be
    #   any predictions
    if map_type != 'species_richness':
        pred_stack = rxr.open_rasterio(pred_stack_fp)
        pred_stack = xr.ufuncs.isnan(pred_stack).astype(int).sum(dim = 'band') # see where there are nans in any band

        if map_type not in ['partial_aoh_effects', 'restore_and_abate']:
            if not facet_body_mass:
                pred_stack = pred_stack.rio.reproject_match(agg_raster)
                agg_raster = agg_raster.where(pred_stack == 0)
            else:
                for k in templates.keys():
                    pred_stack = pred_stack.rio.reproject_match(templates[k])
                    templates[k] = templates[k].where(pred_stack == 0)
        elif map_type == 'partial_aoh_effects':
            pred_stack = pred_stack.rio.reproject_match(agg_raster_hunt)

            agg_raster_hunt = agg_raster_hunt.where(pred_stack == 0)
            agg_raster_hab = agg_raster_hab.where(pred_stack == 0)
        elif map_type == 'restore_and_abate':
            pred_stack = pred_stack.rio.reproject_match(agg_raster_abate_restore)

            agg_raster_abate_restore = agg_raster_abate_restore.where(pred_stack == 0)
            agg_raster_abate = agg_raster_abate.where(pred_stack == 0)
            agg_raster_restore = agg_raster_restore.where(pred_stack == 0)
            agg_raster_restore_baseline = agg_raster_restore_baseline.where(pred_stack == 0)

    # Divide through by the cell area (in km^2), only for abate/restore layers 
    #  to make cell values directly comparable (non-equal area grid)
    if map_type == 'restore_and_abate':
        template_cell_area = rxr.open_rasterio(template_raster_fp.replace('.tif', '_cell_area.tif'))
        
        agg_raster_abate_restore = agg_raster_abate_restore / template_cell_area
        agg_raster_abate = agg_raster_abate / template_cell_area
        agg_raster_restore = agg_raster_restore / template_cell_area
        agg_raster_restore_baseline = agg_raster_restore_baseline / template_cell_area

    # Saving the final aggregated raster
    print('Saving results')

    if map_type == 'species_richness':
        dtype = 'uint16'
    elif map_type in ['hunting_pressure', 'joint_aoh_effect', 'partial_aoh_effects']:
        dtype = 'float32'
    elif map_type == 'restore_and_abate':
        dtype = 'float64'

    if map_type not in ['partial_aoh_effects', 'restore_and_abate']:
        if not facet_body_mass:
            agg_raster.rio.to_raster(save_fp % '', dtype = dtype)
        else:
            for k in templates.keys():
                templates[k].rio.to_raster(save_fp % ('_' + k.replace(' ', '_')), dtype = dtype)
    elif map_type == 'partial_aoh_effects':
        agg_raster_hunt.rio.to_raster(save_fps['hunting'] % '', dtype = dtype)
        agg_raster_hab.rio.to_raster(save_fps['habitat_loss'] % '', dtype = dtype)
    elif map_type == 'restore_and_abate':
        agg_raster_abate_restore.rio.to_raster(save_fps['restore_and_abate'] % '', dtype = dtype)
        agg_raster_abate.rio.to_raster(save_fps['abate'] % '', dtype = dtype)
        agg_raster_restore.rio.to_raster(save_fps['restore'] % '', dtype = dtype)
        agg_raster_restore_baseline.rio.to_raster(save_fps['restore_baseline'] % '', dtype = dtype)

if __name__ == '__main__':
    # Read in parameters
    with open('experiments/spatial_aggregate_maps.json', 'r') as f:
        params = json.load(f)

    # Choosing either "local" or "remote"
    mode = 'local'
    print(f'Running in {mode} mode\n')

    main(params, mode)
