import os
import sys
import json
import copy
from itertools import chain

from tqdm import tqdm

import pandas as pd
import numpy as np

import open_clip
import torch
from huggingface_hub import hf_hub_download

dir_to_add = os.path.dirname(os.path.relpath(__file__))
sys.path.append(os.path.join(dir_to_add, 'satclip'))
sys.path.append(os.path.join(dir_to_add, 'satclip', 'satclip'))

import satclip
from satclip.load import get_satclip

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from pytaxize import gn, Ids, itis

import utils

def get_record_species_embedding(species_list, embedding_dict):

    """
    A helper function to extract the mean embedding for each record. This is designed
    to be applied to a pandas series.

    Paramaters
    ----------
    species_list : list
        a list of species for the record
    embedding_dict : dictionary
        a dictionary with the embeddings for all species and higher level taxa

    Returns
    -------
    mean_embedding : numpy.array
        the mean embedding for all species represented in this record
    """

    all_embeddings = np.array([embedding_dict[s]['embedding'] for s in species_list])
    mean_embedding = np.mean(all_embeddings, axis = 0)

    return mean_embedding

def get_all_embeddings(ben_lop_data, pca = False, var_cutoff = 0.9, embeddings_to_use = None,
                       train_test_idxs = None, satclip_L = 40):

    """
    A function to get the requested deep learning embeddings for the dataset.

    Paramaters
    ----------
    ben_lop_data : pandas.DataFrame
        the Benitez-Lopez et al. (2019) dataset
    pca : boolean
        should we apply PCA to reduce the dimensionality of embeddings?
    var_cutoff : float
        the cutoff for variance explained in the PCA (dictates the number of
        components to keep)
    embeddings_to_use : list
        a list of embeddings to use (i.e., 'SatCLIP' and/or 'BioCLIP')
    train_test_idxs : dictionary
        a dictionary containing entries 'train' and 'test', with lists of indices
        for each
    satclip_L : integer
        the degree of the spherical harmonics in SatCLIP; either 10 or 40

    Returns
    -------
    all_emb : pandas.DataFrame
        a dataframe containing embeddings, each row corresponds to a row of the
        original dataframe (i.e., an observation) and each column is a dimension
        of the embedding (potentially projected onto the principal components)
    """

    assert train_test_idxs is not None, 'Please provide training and testing indices to facilitate standardization and PCA.'

    if embeddings_to_use is None:
        embeddings_to_use = ['SatCLIP', 'BioCLIP']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embeddings = []

    # Getting the SatCLIP location embedding
    if 'SatCLIP' in embeddings_to_use:
        #  this only loads location encoder by default
        with utils.HiddenPrints():
            model = get_satclip(hf_hub_download(f'microsoft/SatCLIP-ResNet50-L{satclip_L}',
                                                f'satclip-resnet50-l{satclip_L}.ckpt'),
                                device = device)
        model.eval()

        #  extracting coordinates - inputs are (longitude, latitude)
        coords = torch.from_numpy(ben_lop_data[['X', 'Y']].values).to(device)

        #  processing using the pre-trained location embedder from SatCLIP
        with torch.no_grad():
            coord_emb = model(coords).detach().cpu()
            coord_emb /= coord_emb.norm(dim = -1, keepdim = True) # vector norm to be consistent w/BioCLIP

        coord_emb = coord_emb.numpy()

        #  scaling the data (z-score normalization)
        scaler = StandardScaler()
        coord_emb_train = scaler.fit_transform(coord_emb[train_test_idxs['train']])
        coord_emb_test = scaler.transform(coord_emb[train_test_idxs['test']])

        #  optionally applying PCA to reduce dimensionality of the embedding
        if pca:
            pca = PCA()
            coord_emb_train = pca.fit_transform(coord_emb_train)
            coord_emb_test = pca.transform(coord_emb_test)

            #  getting enough components to explain > var_cutoff variance
            exp_var = pca.explained_variance_ratio_.cumsum()
            idx_cutoff = np.argmax(exp_var > var_cutoff) + 1
        else:
            idx_cutoff = coord_emb_train.shape[1] # just including all embeddings

        coord_emb = np.vstack((coord_emb_train[ : , : idx_cutoff], coord_emb_test[ : , : idx_cutoff]))

        #  putting into a dataframe
        cols = [f'satclip_{i}' for i in range(coord_emb.shape[1])]
        coord_emb_pd = pd.DataFrame(coord_emb, columns = cols)

        #  sorting rows to facilitate combination with the rest of the dataset
        coord_emb_pd.index = list(train_test_idxs['train']) + list(train_test_idxs['test'])
        coord_emb_pd = coord_emb_pd.sort_index()

        embeddings.append(coord_emb_pd)

    # Getting the mean BioCLIP embedding for each record in the dataset
    if 'BioCLIP' in embeddings_to_use:
        #  reading in the saved embeddings
        with open(os.path.join(dir_to_add, 'embeddings/bioclip_embeddings.json'), 'r') as f:
            bioclip_emb = json.load(f)

        #  getting embeddings for the dataset
        species = ben_lop_data['Species'].apply(multi_species_extraction)
        species_emb = species.apply(get_record_species_embedding, args = (bioclip_emb, )).values
        species_emb = np.stack(species_emb)

        #  scaling the data (z-score normalization)
        scaler = StandardScaler()
        species_emb_train = scaler.fit_transform(species_emb[train_test_idxs['train']])
        species_emb_test = scaler.transform(species_emb[train_test_idxs['test']])

        #  optionally applying PCA to reduce dimensionality of the embedding
        if pca:
            pca = PCA()
            species_emb_train = pca.fit_transform(species_emb_train)
            species_emb_test = pca.transform(species_emb_test)

            #  getting enough components to explain > var_cutoff variance
            exp_var = pca.explained_variance_ratio_.cumsum()
            idx_cutoff = np.argmax(exp_var > var_cutoff) + 1
        else:
            idx_cutoff = species_emb_train.shape[1] # just including all embeddings

        species_emb = np.vstack((species_emb_train[ : , : idx_cutoff], species_emb_test[ : , : idx_cutoff]))

        #  putting into a dataframe
        cols = [f'bioclip_{i}' for i in range(species_emb.shape[1])]
        species_emb_pd = pd.DataFrame(species_emb, columns = cols)

        #  sorting rows to facilitate combination with the rest of the dataset
        species_emb_pd.index = list(train_test_idxs['train']) + list(train_test_idxs['test'])
        species_emb_pd = species_emb_pd.sort_index()

        embeddings.append(species_emb_pd)

    # Combining embeddings into a unified dataframe
    all_emb = pd.concat(embeddings, axis = 1)

    return all_emb

def multi_species_extraction(species_names):

    """
    A function to handle the case where multiple species are present for a single observation
    or an observation is at a higher taxonomic level. This is meant for use with Benitez-Lopez
    et al. (2019).

    Paramaters
    ----------
    species_names : string
        a string containing one or more species

    Returns
    -------
    species_names : list
        a list with all species names found
    """

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

    """
    A helper function to read the Benitez-Lopez et al. (2019) dataset.

    Returns
    -------
    ben_lop2019 : pandas.DataFrame
        Benitez-Lopez et al. (2019) dataset
    """

    # Loading in general configuration
    with open(os.path.join(dir_to_add, 'config.json'), 'r') as f:
        config = json.load(f)

    # Getting filepaths
    gdrive_fp = config['gdrive_path']
    LIFE_fp = config['LIFE_folder']
    dataset_fp = config['datasets_path']

    # Grabbing Benitez-Lopez
    benitez_lopez2019 = config['indiv_data_paths']['benitez_lopez2019']
    ben_lop_path = os.path.join(gdrive_fp, LIFE_fp, dataset_fp, benitez_lopez2019)
    ben_lop2019 = utils.read_csv_non_utf(ben_lop_path)

    return ben_lop2019

def get_higher_level_dicts(higher_tax_level):

    """
    A specialized function to extract the full taxonomic hierarchy when only a level
    higher than species is indicated for a record.

    Paramaters
    ----------
    higher_tax_level : list
        a list of taxonomic names (e.g., Genus)

    Returns
    -------
    higher_tax_full : list
        a list containing the corresponding full taxonomic hierarchy
    """

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
        tax_names = utils.get_species_names(itis_id = sel_id, level = level_to_pass)
        higher_tax_full.append(tax_names)

    return higher_tax_full

def get_species_dicts(full_species, species_resolved):

    """
    A helper function to get the full taxonomic hierarchy when species names are
    present.

    Paramaters
    ----------
    full_species : list
        a list of the original scientific binomials from Benitez-Lopez et al. (2019)
    species_resolved : list
        a list of resolved scientific binomials

    Returns
    -------
    full_names : list
        a list containing the corresponding full taxonomic hierarchy
    """

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
        full_names.append(utils.get_species_names(species, itis_id))

    return full_names

def main():

    """
    A wrapper for obtaining species embeddings, to be run as a script. Results are
    saved to 'embeddings/bioclip_embeddings.json'.
    """

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
    species_embeddings = utils.get_species_embeddings(species_full, model, tokenizer, full_hierarchy = True,
                                                      common_name = True)
    higher_tax_embeddings = utils.get_species_embeddings(higher_tax_full, model, tokenizer, full_hierarchy = True,
                                                         common_name = True, names_to_use = higher_tax_level)

    #  merging the embedding dicts
    all_embeddings = copy.deepcopy(species_embeddings)
    for entry in higher_tax_embeddings.keys():
        all_embeddings[entry] = higher_tax_embeddings[entry]

    #  saving the unified embedding dict
    with open(os.path.join(dir_to_add, 'embeddings/bioclip_embeddings.json'), 'w') as f:
        json.dump(all_embeddings, f)

if __name__ == '__main__':
    main()
