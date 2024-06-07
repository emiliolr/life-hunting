import os
import json
import copy
from itertools import chain

from tqdm import tqdm

import pandas as pd
import numpy as np

import open_clip
from huggingface_hub import hf_hub_download

from pytaxize import gn, Ids, itis

from utils import read_csv_non_utf, get_species_embeddings, get_species_names

def multi_species_extraction(species_names):
    if ', ' in species_names:
        species_names = species_names.split(',')
        species_names = [s.split('and ') for s in species_names]
        species_names = list(chain(*species_names))
    elif 'and ' in species_names:
        species_names = species_names.split('and ')
    elif 'or ' in species_names:
        species_names = species_names.split('or ')
    else:
        return [species_names.replace('spp', '*').replace('.', '')]

    # Removing whitespace and empty strings
    species_names = [s.strip() for s in species_names]
    species_names = [s for s in species_names if s != '']

    # Special cases for fixing the binomials
    if species_names[0] == 'Sciurus spadiceus':
        species_names = ['Sciurus spadiceus', 'Sciurus sanborni']
    elif species_names[0] == 'Saguinus mystax':
        species_names = ['Saguinus mystax', 'Saguinus imperator']
    elif species_names[0] == 'Potos flavus':
        species_names = ['Potos flavus', 'Bassaricyon *']

    # General case
    else:
        for i in range(1, len(species_names)):
            if (species_names[i][0] == species_names[0][0]) and (species_names[i][1] == '.'):
                new_name = species_names[i].split(' ')[1]
                new_name = species_names[0].split(' ')[0] + ' ' + new_name

                species_names[i] = new_name

    # Removing any unnecessary periods
    species_names = [s.replace('.', '') for s in species_names]

    return species_names

def read_dataset():
    # Loading in general configuration
    with open('config.json', 'r') as f:
        config = json.load(f)

    # Getting filepaths
    gdrive_fp = config['gdrive_path']
    LIFE_fp = config['LIFE_folder']
    dataset_fp = config['datasets_path']

    # Grabbing Benitez-Lopez
    benitez_lopez2019 = config['indiv_data_paths']['benitez_lopez2019']
    ben_lop_path = os.path.join(gdrive_fp, LIFE_fp, dataset_fp, benitez_lopez2019)
    ben_lop2019 = read_csv_non_utf(ben_lop_path)

    return ben_lop2019

def get_higher_level_dicts(higher_tax_level):
    higher_tax_full = []

    for name in tqdm(higher_tax_level):
        name = name.split(' ')[0]

        #  special cases
        if name == 'Cebus':
            sel_id = 572816
        elif name == 'Ateles':
            sel_id = 572812
        elif name == 'Sciurus':
            sel_id = 180171
        elif name == 'Dasyprocta':
            sel_id = 584623

        #  general case
        else:
            tax_id = Ids(name)
            tax_id.itis(type = 'scientific')
            ids = tax_id.extract_ids()

            sel_id = int(ids[name][0]) # the first ID is the direct record, after are children...

        level = itis.rank_name(sel_id)['rankName'] # checking the taxonomic level
        level_to_pass = level.replace('Sub', '')

        #  cleaning up the output to reflect the fact that we only have genus info
        tax_names = get_species_names(itis_id = sel_id, level = level_to_pass)
        higher_tax_full.append(tax_names)

    return higher_tax_full

def get_species_dicts(full_species, species_resolved):
    species_itis = {}
    for s, s_dict in zip(full_species, species_resolved):
        if s == 'Smutsia gigantea':
            tax_id = Ids('Manis gigantea') # this is the correct entry for the giant pangolin
            tax_id.itis(type = 'scientific')
            ids = tax_id.extract_ids()
            sel_id = int(ids['Manis gigantea'][0])

            species_itis[s] = sel_id
        else:
            s_dict = s_dict[0]
            species_itis[s] = int(s_dict['current_taxon_id']) if 'current_taxon_id' in s_dict.keys() else int(s_dict['taxon_id'])

    # Querying ITIS for full taxonomic hierachy using our existing function
    full_names = []
    for species, itis_id in tqdm(species_itis.items()):
        full_names.append(get_species_names(species, itis_id))

    return full_names

def main():
    # Getting the dataset
    ben_lop2019 = read_dataset()

    # Grabbing all unique species
    ben_lop2019_species = ben_lop2019['Species'].apply(multi_species_extraction)
    ben_lop2019_species = set(chain(*list(ben_lop2019_species)))

    full_species = [s for s in ben_lop2019_species if '*' not in s]
    higher_tax_level = [s for s in ben_lop2019_species if '*' in s]

    # Getting full names - GENUS (or higher level) ONLY CASE!
    print(f'Getting full names for {len(higher_tax_level)} higher taxonomic levels')
    higher_tax_full = get_higher_level_dicts(higher_tax_level)

    # Getting full names - FULL SPECIES CASE!
    print(f'Getting full names for {len(full_species)} species')
    species_resolved = gn.resolve(full_species, best_match_only = True, source = [3])
    species_full = get_species_dicts(full_species, species_resolved)

    # Reading in the pre-trained BioCLIP model
    model, _, preprocess_val = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')
    tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip')

    # Processing w/BioCLIP and saving the resulting embeddings
    print(f'Getting BioCLIP embeddings for {len(higher_tax_full) + len(species_full)} records')
    species_embeddings = get_species_embeddings(species_full, model, tokenizer, full_hierarchy = True,
                                                common_name = True)
    higher_tax_embeddings = get_species_embeddings(higher_tax_full, model, tokenizer, full_hierarchy = True,
                                                   common_name = True, names_to_use = higher_tax_level)

    #  merging the embedding dicts
    all_embeddings = copy.deepcopy(species_embeddings)
    for entry in higher_tax_embeddings.keys():
        all_embeddings[entry] = higher_tax_embeddings[entry]

    #  saving the unified embedding dict
    with open('embeddings/bioclip_embeddings.json', 'w') as f:
        json.dump(all_embeddings, f)

if __name__ == '__main__':
    main()
