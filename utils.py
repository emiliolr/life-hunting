from io import StringIO

import pandas as pd
import numpy as np
import torch

from pytaxize import Ids
from pytaxize import itis

from sklearn.metrics import recall_score

def read_csv_non_utf(filepath):

    """
    A wrapper function to handle cases where a CSV contains non-UTF-8 characters.

    Parameters
    ----------
    filepath : string
        the path to CSV file

    Returns
    -------
    dataset : pd.DataFrame
        the read dataframe with non-UTF-8 characters removed
    """

    # Removing the non-UTF-8 characters present in the CSV file
    data = ''
    with open(filepath, 'rb') as f:
        for line in f:
            line = line.decode('utf-8', 'ignore')
            data += line

    # Turning the string into a pandas dataframe
    dataset = pd.read_csv(StringIO(data))

    return dataset

def get_species_names(scientific_name):

    """
    A function to extract the relevant taxonomic information for a given species from ITIS;
    currently gets the full taxonomic hierarchy and any English-language common names.

    Paramaters
    ----------
    scientific_name : string
        the scientific name (binomial) for the species of interest

    Returns
    -------
    name_dict : dictionary
        a dictionary containing the scientific, full taxonomic, and common names
    """

    # Info to get from ITIS
    ranks_to_include = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']

    # Setting up data structure to hold obtained data
    name_dict = {}
    name_dict['scientific_name'] = scientific_name

    # Getting the species ID for ITIS
    tax_id = Ids(scientific_name)
    tax_id.itis(type = 'scientific')
    ids = tax_id.extract_ids()
    sel_id = int(ids[scientific_name][0]) # it seems like subspecies entries generally come after plain species...

    # Extracting the full taxonomic hierarchy for the species and adding to the dictionary
    tax_hier = itis.hierarchy_full(sel_id, as_dataframe = True)

    for rank in ranks_to_include:
        rank_value = tax_hier[tax_hier['rankName'] == rank]['taxonName'].values[0]
        if rank == 'Species':
            rank_value = rank_value.split(' ')[1]

        name_dict[rank] = rank_value

    # Get the species' common name (if it has one) and add to the dictionary
    common = itis.common_names(sel_id)

    if len(common) == 0:
        print(f'{scientific_names} has no common names recorded in ITIS')
        com_names = []
    else:
        com_names = [d['commonName'].lower() for d in common if d['language'] == 'English']
        com_names = list(set(com_names))

    name_dict['common_names'] = com_names

    return name_dict

def format_species_name_CLIP(name_dict, full_hierarchy = True, common_name = True):

    """
    A helper function to format species names in the several supported formats for BioCLIP.

    Paramaters
    ----------
    name_dict : dictionary
        a dictionary containing name information for a species (see get_species_names)
    full_hierarchy : boolean
        include the full taxonomic name in the name string?
    common_name : boolean
        include the common name/s in the name string?

    Returns
    -------
    name_strs : list
        a list of name strings for the species, one per common name
    """

    ranks_to_include = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    name_str = 'a photo of'

    if full_hierarchy or common_name:
        # Adding the full hierarchical name to the name string
        if full_hierarchy:
            hier_str = ' '.join([name_dict[r] for r in ranks_to_include])
            name_str += ' ' + hier_str

        # Adding the common name/s to the name string
        if common_name:
            com_names = name_dict['common_names']
            assert len(com_names) > 0, f'{name_dict["scientific_name"]} has no common names recorded in ITIS'

            if not full_hierarchy:
                name_strs = [name_str + ' ' + name_dict['scientific_name'] + ' with common name ' + name for name in com_names]
            else:
                name_strs = [name_str + ' with common name ' + name for name in com_names]

        # Esnuring that we always pass back a list
        if full_hierarchy and not common_name:
            name_strs = [name_str]
    else:
        # Simply returning the scientific (binomial) name
        name_strs = [name_str + ' ' + name_dict['scientific_name']]

    return name_strs


def get_species_embeddings(species_name_dicts, bioclip_model, tokenizer, full_hierarchy = True, common_name = True):

    """
    Get species embeddings based on a textual name (scientific, common, full taxonomic,
    or some combination). If multiple common names are used, the mean embedding is returned.

    Parameters
    ----------
    species_name_dicts : list
        a list of name dictionaries (see get_species_names)
    bioclip_model : open_clip.model.CLIP
        the pre-trained torch BioCLIP model to use
    tokenizer : open_clip.tokenizer
        the corresponding tokenizer to use for pre-processing inputs
    full_hierarchy : boolean
        include the full taxonomic name in the name string?
    common_name : boolean
        include the common name/s in the name string?

    Returns
    -------
    species_embeddings : dictionary
        a dictionary containing an entry for each species, holding their (mean) embedding
        and the name strings used to produce the embedding
    """

    species_embeddings = {}

    # For each species, get the relevant taxonomic name, process using BioCLIP, and add to the dictionary to return
    for name_dict in species_name_dicts:
        name_strs = format_species_name_CLIP(name_dict, full_hierarchy, common_name)
        text = tokenizer(name_strs)

        with torch.no_grad():
            text_features = bioclip_model.encode_text(text) # get embedding based on text representation of species
            text_features /= text_features.norm(dim = -1, keepdim = True) # taking a vector norm, as is standard with CLIP

            mean_embedding = text_features.mean(axis = 0) # get the mean embedding when there are multiple common names

        species_embeddings[name_dict['scientific_name']] = {}
        species_embeddings[name_dict['scientific_name']]['embedding'] = mean_embedding.numpy()
        species_embeddings[name_dict['scientific_name']]['names_used'] = name_strs

    return species_embeddings

def true_skill_statistic(y_pred, y_true, return_spec_sens = False):

    """
    Compute the true skill statistic (TSS) based on the definition given in Gallego-Zamorano
    et al. (2020). Designed for binary classication.

    Parameters
    ----------
    y_pred : iterable
        the predicted classifications for a given set of observations
    y_true : iterable
        the true labels for a given set of observations
    return_spec_sens : boolean
        should we return calculated specificity and sensitivity in addition to the TSS?

    Returns
    -------
    tss : float
        the calculated TSS
    sensitivity : float
        the calculated sensitivity
    specifcity : float
        the calculated specificity
    """

    sensitivity = recall_score(y_true, y_pred, pos_label = 1)
    specificity = recall_score(y_true, y_pred, pos_label = 0) # sensitivity is just recall for the negative class
    tss = sensitivity + specificity - 1

    if return_spec_sens:
        return tss, sensitivity, specificity

    return tss

def test_thresholds(y_pred, y_true, precision = 0.05):

    """
    Test all threshold values between 0 and 1 to find the best threshold based on the TSS.
    This mirrors the `findOptimum` function in the code from Gallego-Zamorano et al. (2020).

    Paramaters
    ----------
    y_pred : iterable
        the predicted classifications for a given set of observations
    y_true : iterable
        the true labels for a given set of observations
    precision : float
        the step size for thresholds tested between 0 and 1

    Returns
    -------
    opt_thresh : float
        the optimal threshold based on the TSS
    metrics : pandas.DataFrame
        all thresholds and the obtained TSS, sensitivity, and specificity
    """

    metrics = []

    thresholds = np.arange(0, 1, precision) # tresholds to test
    for thresh in thresholds:
        thresh = round(thresh, 4) # fixing weird precision thing with arange...
        y_hard = (y_pred >= thresh).astype(int) # values greater than the threshold are considered positive hard classifications

        tss, sens, spec = true_skill_statistic(y_hard, y_true, return_spec_sens = True) # compute metrics at this threshold
        metrics.append([thresh, tss, sens, spec])

    metrics = pd.DataFrame(metrics, columns = ['threshold', 'TSS', 'sensitivity', 'specificity']) # put results into dataframe for return
    opt_thresh = thresholds[np.argmax(metrics['TSS'])] # grab the optimal threshold based on the TSS

    return opt_thresh, metrics

def ratios_to_DI_cats(ratios, category_names = False):

    """
    Convert abundance ratios to defaunation categories, as defined in
    Benitez-Lopez et al. (2019).

    Paramaters
    ----------
    ratios : iterable
        the abundance ratio to be converted to defaunation categories
    category_names : boolean
        should we return category names rather than numeric values?

    Returns
    -------
    DI_categories : numpy.array
        the defaunation categories corresponding to the supplied abundance ratios
    """

    DI = 1 - ratios # go from abundance ratio to percentage lost due to hunting
    DI_categories = DI.copy()

    # Defaunation categories, following Benitez-Lopez et al. (2019)
    DI_categories[0.1 >= DI] = 0
    DI_categories[(0.7 > DI) & (DI > 0.1)] = 1
    DI_categories[DI >= 0.7] = 2

    # Change to category names rather than numeric values
    if category_names:
        categories = {0 : 'low', 1 : 'medium', 2 : 'high'}
        DI_categories = np.array([categories[cat] for cat in DI_categories])

    return DI_categories
