import sys
import os
from io import StringIO

import pandas as pd
import numpy as np
import torch

from pytaxize import Ids, itis

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from imblearn.over_sampling import SMOTENC

from custom_metrics import true_skill_statistic
import embeddings

def read_csv_non_utf(filepath, **kwargs):

    """
    A wrapper function to handle cases where a CSV contains non-UTF-8 characters.

    Parameters
    ----------
    filepath : string
        the path to CSV file
    **kwargs
        passed on to pandas.read_csv

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
    dataset = pd.read_csv(StringIO(data), **kwargs)

    return dataset

# Helper class from StackOverflow: https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
class HiddenPrints:

    """
    Helper class to suppress printing using `with`.
    """

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def get_species_names(scientific_name = None, itis_id = None, level = 'Species'):

    """
    A function to extract the relevant taxonomic information for a given species from ITIS;
    currently gets the full taxonomic hierarchy and any English-language common names.

    Paramaters
    ----------
    scientific_name : string
        the scientific name (binomial) for the species of interest
    itis_id : integer
        the ITIS ID for the species
    level : string
        the taxonomic level of the record, e.g., "species" or "genus"

    Returns
    -------
    name_dict : dictionary
        a dictionary containing the scientific, full taxonomic, and common names
    """

    assert (scientific_name is not None) or (itis_id is not None), 'Please supply a scientific name or ITIS ID.'

    # Info to get from ITIS
    ranks_to_include = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']

    # Setting up data structure to hold obtained data
    name_dict = {'scientific_name' : None}
    if scientific_name is not None:
        name_dict['scientific_name'] = scientific_name

    # Getting the species ID for ITIS
    if itis_id is None:
        tax_id = Ids(scientific_name)
        tax_id.itis(type = 'scientific')
        ids = tax_id.extract_ids()
        itis_id = int(ids[scientific_name][0]) # it seems like subspecies entries generally come after plain species...

    # Extracting the full taxonomic hierarchy for the species and adding to the dictionary
    tax_hier = itis.hierarchy_full(itis_id, as_dataframe = True)

    flag = True
    for rank in ranks_to_include:
        if flag:
            rank_value = tax_hier[tax_hier['rankName'] == rank]['taxonName']
            rank_value = rank_value.iloc[0]

            if rank == 'Species':
                if scientific_name is None:
                    name_dict['scientific_name'] = rank_value

                rank_value = rank_value.split(' ')[1]

            name_dict[rank] = rank_value
        else:
            name_dict[rank] = None

        #  flag to stop recording values since we've reached the level of the record
        if rank.lower() == level.lower():
            flag = False

    # Get the species' common name (if it has one) and add to the dictionary
    common = itis.common_names(itis_id)

    if len(common) == 0 or common[0] is None:
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
        include the common name/s in the name string? if no common names are present,
        nothing is added to the string

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
            hier_str = ' '.join([name_dict[r] for r in ranks_to_include if name_dict[r] is not None])
            name_str += ' ' + hier_str

        # Adding the common name/s to the name string
        com_names = name_dict['common_names']
        if common_name:
            if len(com_names) > 0:
                if not full_hierarchy:
                    name_strs = [name_str + ' ' + name_dict['scientific_name'] + ' with common name ' + name for name in com_names]
                else:
                    name_strs = [name_str + ' with common name ' + name for name in com_names]

        # Ensuring that we always pass back a list
        if full_hierarchy and (not common_name or len(com_names) == 0):
            name_strs = [name_str]
    else:
        # Simply returning the scientific (binomial) name
        name_strs = [name_str + ' ' + name_dict['scientific_name']]

    return name_strs

def get_species_embeddings(species_name_dicts, bioclip_model, tokenizer, full_hierarchy = True,
                           common_name = True, names_to_use = None):

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
    names_to_use : list
        the names to use instead of the scientific names in species_name_dicts

    Returns
    -------
    species_embeddings : dictionary
        a dictionary containing an entry for each species, holding their (mean) embedding
        and the name strings used to produce the embedding
    """

    species_embeddings = {}
    if names_to_use is None:
        names_to_use = [d['scientific_name'] for d in species_name_dicts]

    # For each species, get the relevant taxonomic name, process using BioCLIP, and add to the dictionary to return
    for name, name_dict in zip(names_to_use, species_name_dicts):
        name_strs = format_species_name_CLIP(name_dict, full_hierarchy, common_name)
        text = tokenizer(name_strs)

        with torch.no_grad():
            text_features = bioclip_model.encode_text(text) # get embedding based on text representation of species
            text_features /= text_features.norm(dim = -1, keepdim = True) # taking a vector norm, as is standard with CLIP

            mean_embedding = text_features.mean(axis = 0) # get the mean embedding when there are multiple common names

        species_embeddings[name] = {}
        species_embeddings[name]['embedding'] = mean_embedding.tolist()
        species_embeddings[name]['names_used'] = name_strs

    return species_embeddings

def test_thresholds(y_pred, y_true, precision = 0.05):

    """
    Test all threshold values between 0 and 1 to find the best threshold based on the TSS.
    This mirrors the `findOptimum` function in the code from Gallego-Zamorano et al. (2020).

    Paramaters
    ----------
    y_pred : iterable
        the predicted PROBABILITY classifications for a given set of observations
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

    thresholds = np.arange(0, 1, precision) # thresholds to test
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

# From: https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9
def count_parameters(model):

    """
    A helper function to get the number of parameters in a torch model.

    Parameters
    ----------
    model : torch.NN.module
        a PyTorch model

    Returns
    -------
    num_params : integer
        the number of parameters in the supplied torch model
    """

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return num_params

def get_zero_nonzero_datasets(pp_data, pred = True, outlier_cutoff = np.Inf, extirp_pos = True,
                              zero_columns = None, nonzero_columns = None, indicator_columns = None,
                              embeddings_to_use = None, dataset = 'mammals', rebalance_dataset = False):

    """
    A helper function to split out the datasets for the binary (zero) and continuous
    (nonzero) models of a hunting effects hurdle model.

    Paramaters
    ----------
    pp_data : pandas.DataFrame
        a dataframe containing preprocessed hunting effects data
    pred : boolean
        will these be used for prediction? if True, labels (y_zero, y_nonzero)
        are not returned
    outlier_cutoff : float
        a positive number that indicates the largest abundance ratio to keep
        in the datasets
    extirp_pos : boolean
        should we code a local extirpation event as the positive class, i.e.,
        extirpated = 1 and extant = 0?
    zero_columns : list
        a list of columns to extract for fitting the binary extirpation model
    nonzero_columns : list
        a list of columns to extract for fitting the continuous model
    indicator_columns : list
        a list of columns to use as indicator variables or random effects
    embeddings_to_use : list
        the name of the embeddings to use (i.e., 'SatCLIP' and/or 'BioCLIP')
    dataset : string
        the dataset being used, either 'mammals' (Benitez-Lopez et al., 2019),
        'birds' (Ferreiro-Arias et al., 2024), or one of 'both', 'mammals_extended', 
        'birds_extended', or 'mammals_recreated'
    rebalance_dataset : boolean
        should we oversample the minority class for the local extirpation model using
        the SMOTE algorithm?

    Returns
    -------
    X_zero : pandas.DataFrame
        a dataframe containing the predictors for the zero model
    X_nonzero : pandas.DataFrame
        a dataframe containing the predictors for the nonzero model
    y_zero : numpy.array
        an array containing the binary labels for the zero model
    y_nonzero : numpy.array
        an array containing the continuous response ratios for the zero model
    """

    # Default predictors
    if dataset in ['mammals', 'both']:
        if indicator_columns is None:
            indicator_columns = ['Country', 'Species']
            if dataset == 'mammals':
                indicator_columns += ['Study']
        if nonzero_columns is None:
            nonzero_columns = ['BM', 'DistKm', 'PopDens']
            if dataset == 'both':
                nonzero_columns += ['TravTime', 'Stunting', 'Reserve']
        if zero_columns is None:
            zero_columns = ['BM', 'DistKm', 'PopDens', 'Stunting', 'Reserve']
            if dataset == 'both':
                zero_columns += ['TravTime']
    elif dataset == 'mammals_recreated':
        all_cols = ['Body_Mass', 'Stunting_Pct', 'Literacy_Rate', 'Dist_Settlement_KM', 
                    'Travel_Time_Large', 'Livestock_Biomass', 'Population_Density', 
                    'Percent_Settlement_50km', 'Protected_Area', 'Corruption', 
                    'Government_Effectiveness', 'Political_Stability', 'Regulation', 
                    'Rule_of_Law', 'Accountability']
        if indicator_columns is None:
            indicator_columns = ['IUCN_Country_Region']
        if nonzero_columns is None:
            nonzero_columns = all_cols
        if zero_columns is None:
            zero_columns = all_cols
    elif dataset == 'birds':
        if indicator_columns is None:
            indicator_columns = ['Country', 'Species']
        if nonzero_columns is None:
            nonzero_columns = ['Dist_Hunters', 'TravDist', 'PopDens', 'Stunting',
                               'FoodBiomass', 'Forest_cover', 'NPP', 'Body_Mass',
                               'Reserve']
        if zero_columns is None:
            zero_columns = ['Dist_Hunters', 'TravDist', 'PopDens', 'Stunting',
                            'FoodBiomass', 'Forest_cover', 'NPP', 'Body_Mass',
                            'Reserve']
    elif dataset == 'birds_extended':
        if indicator_columns is None:
            indicator_columns = ['Country', 'Species']
        if nonzero_columns is None:
            nonzero_columns = ['Trophic_Niche', 'Dist_Hunters', 'TravDist', 'PopDens', 'Stunting', 
                               'FoodBiomass', 'Forest_Cover', 'NPP', 'Range_Size', 'Body_Mass', 
                               'Reserve', 'IUCN_Is_Hunted', 'IUCN_Is_Human_Food', 'IUCN_For_Pet_Trade', 
                               'IUCN_Is_Threatened', 'Habitat_Is_Dense']
        if zero_columns is None:
            zero_columns = ['Trophic_Niche', 'Dist_Hunters', 'TravDist', 'PopDens', 'Stunting', 
                            'FoodBiomass', 'Forest_Cover', 'NPP', 'Range_Size', 'Body_Mass', 
                            'Reserve', 'IUCN_Is_Hunted', 'IUCN_Is_Human_Food', 'IUCN_For_Pet_Trade', 
                            'IUCN_Is_Threatened', 'Habitat_Is_Dense']
    elif dataset == 'mammals_extended':
        all_cols = ['Diet', 'Body_Mass', 'Year', 'Dist_Access_Pts', 'Travel_Time', 
                    'Livestock_Biomass', 'Stunting', 'Population_Density', 'Literacy', 
                    'Forest_Cover', 'NPP', 'Reserve', 'Activity_Nocturnal', 
                    'Activity_Crepuscular', 'Activity_Diurnal', 'IUCN_Is_Hunted',
                    'IUCN_Is_Human_Food', 'IUCN_For_Handicrafts', 'IUCN_For_Medicine', 
                    'IUCN_For_Pet_Trade', 'IUCN_Is_Sport_Hunted', 'IUCN_For_Wearing_Apparel', 
                    'IUCN_Is_Threatened']

        if indicator_columns is None:
            indicator_columns = []
        if nonzero_columns is None:
            nonzero_columns = all_cols
        if zero_columns is None:
            zero_columns = all_cols
    else:
        raise ValueError('The only supported datasets are "mammals," "birds," "both", "birds_extended", "mammals_extended", or "mammals_recreated".')

    if embeddings_to_use is None:
        embeddings_to_use = []

    X_nonzero = pp_data[[]].copy(deep = True)
    X_zero = pp_data[[]].copy(deep = True)

    #  filling in like this ensures this works for polynomial features/indicator variables
    for col in nonzero_columns:
        X_nonzero = pd.concat((X_nonzero, pp_data.filter(like = col)), axis = 1)
    for col in zero_columns:
        X_zero = pd.concat((X_zero, pp_data.filter(like = col)), axis = 1)
    for col in indicator_columns + embeddings_to_use:
        col = col.lower() if col in embeddings_to_use else col
        X_nonzero = pd.concat((X_nonzero, pp_data.filter(like = col)), axis = 1)
        X_zero = pd.concat((X_zero, pp_data.filter(like = col)), axis = 1)

    #  making sure to toss any duplicate columns
    X_nonzero = X_nonzero.loc[ : , ~X_nonzero.columns.duplicated()].copy(deep = True)
    X_zero = X_zero.loc[ : , ~X_zero.columns.duplicated()].copy(deep = True)

    # Extracting the inputs/outputs for each of the models in the case where we have labels
    if not pred:
        resp_col = 'ratio' if dataset in ['mammals', 'both'] else 'RR'
        if dataset == 'mammals_extended':
            resp_col = 'Ratio'
        elif dataset == 'mammals_recreated':
            resp_col = 'Response_Ratio'

        ratio = pp_data[resp_col].values
        nonzero_mask = (ratio != 0)
        outlier_mask = ratio < outlier_cutoff # only keeping values smaller than cutoff - ratio is always positive!

        X_nonzero = X_nonzero[nonzero_mask & outlier_mask].copy(deep = True)
        X_zero = X_zero[outlier_mask].copy(deep = True)

        y_zero = (ratio == 0).astype(int) if extirp_pos else (ratio != 0).astype(int)
        y_zero = y_zero[outlier_mask]
        y_nonzero = np.log(ratio[nonzero_mask & outlier_mask].copy())

        #  optionally rebalancing classes using the categorical-friendly version of SMOTE
        if rebalance_dataset:
            assert dataset in ['mammals', 'birds', 'mammals_extended', 'birds_extended', 'mammals_recreated'], 'Oversampling of minority class only supported for mammals and birds datasets'

            #  defining all candidate columns which are categorical over the different datasets
            all_cat_cols = ['Country', 'Species', 'Study', 'Reserve', 'Diet']
            if 'extended' in dataset:
                all_cat_cols += ['Trophic_Niche', 'IUCN_Is_Hunted', 'IUCN_Is_Human_Food', 'IUCN_For_Pet_Trade', 
                                 'IUCN_Is_Threatened', 'Habitat_Is_Dense', 'Activity_Nocturnal', 'Activity_Crepuscular', 
                                 'Activity_Diurnal', 'IUCN_For_Handicrafts', 'IUCN_For_Medicine', 'IUCN_Is_Sport_Hunted', 
                                 'IUCN_For_Wearing_Apparel']
            elif dataset == 'mammals_recreated':
                all_cat_cols += ['Protected_Area', 'IUCN_Country_Region']

            #  getting the correct columns, recognizing that naming convention may have changed with
            #   addition of indicators variables
            smote_columns = []
            for c in X_zero.columns:
                for k in all_cat_cols:
                    if k in c:
                        smote_columns.append(c)
            
            smote_columns = list(set(smote_columns))

            #  applying the SMOTE algorithm to the data for synthetic oversampling
            smote = SMOTENC(categorical_features = smote_columns, random_state = 1693)
            X_zero, y_zero = smote.fit_resample(X_zero, y_zero)

        return X_zero, y_zero, X_nonzero, y_nonzero

    return X_zero, X_nonzero

def preprocess_data(data, include_indicators = False, include_categorical = False,
                    standardize = False, log_trans_cont = False, polynomial_features = 0,
                    embeddings_to_use = None, embeddings_args = None, train_test_idxs = None,
                    dataset = 'mammals'):

    """
    A helper function to preprocess the hunting effects dataset, including predictors.

    Paramaters
    ----------
    data : pandas.DataFrame
        a dataframe containing the hunting effects dataset (predictors and response)
    include_indicators : boolean
        should we create indicator columns for categorical predictors?
    include_categorical : boolean
        should we mantain categorical predictors?
    standardize : boolean
        should we standardize (center and scale) the continuous predictors?
    log_trans_cont : boolean
        should we log10 transform the continuous predictors?
    polynomial_features : integer
        the degree of the polynomial expansion to apply to continuous predictors
    embeddings_to_use : list
        the name of the embeddings to use (i.e., 'SatCLIP' and/or 'BioCLIP')
    embeddings_args : dictionary
        kwargs to pass to the get_all_embeddings function
    train_test_idxs : list
        a dictionary of training/testing indices to ensure preprocessing only uses
        information (e.g., statistics) from training data
    dataset : string
        the dataset being used, either 'mammals' (Benitez-Lopez et al., 2019) or
        'birds' (Ferreiro-Arias et al., 2024) for original datasets, or one of 'both',
        'birds_extended', 'mammals_extended', or 'mammals_recreated'

    Returns
    -------
    pp_data : pandas.DataFrame
        a dataframe containing the preprocessed dataset
    """

    assert not include_indicators or not include_categorical, 'Cannot include indicators and categorical variables at the same time.'

    # Setting mutable defaults
    if embeddings_args is None:
        embeddings_args = {}

    # Defining the variables needed
    if dataset in ['mammals', 'both']:
        indicator_columns = ['Country', 'Species', 'Family',
                             'Order'] + (['Diet', 'Study', 'Region'] if dataset == 'mammals' else [])
        continuous_columns = ['BM', 'DistKm', 'PopDens', 'Stunting', 'TravTime',
                              'LivestockBio'] + (['Literacy'] if dataset == 'mammals' else [])
        special_columns = ['Reserve']
        response_column = 'ratio'
    elif dataset == 'mammals_recreated':
        indicator_columns = ['IUCN_Country_Region']
        continuous_columns = ['Body_Mass', 'Stunting_Pct', 'Literacy_Rate', 'Dist_Settlement_KM', 
                              'Travel_Time_Large', 'Livestock_Biomass', 'Population_Density', 
                              'Percent_Settlement_50km', 'Corruption', 'Government_Effectiveness',
                              'Political_Stability', 'Regulation', 'Rule_of_Law', 'Accountability']
        special_columns = ['Protected_Area']
        response_column = 'Response_Ratio'
    elif dataset == 'birds':
        indicator_columns = ['Study', 'Dataset', 'Order', 'Family', 'Species',
                             'Traded', 'Realm', 'Country', 'Food', 'Hunted']
        continuous_columns = ['Dist_Hunters', 'TravDist', 'PopDens', 'Stunting',
                              'FoodBiomass', 'Forest_cover', 'NPP', 'Body_Mass']
        special_columns = ['Reserve']
        response_column = 'RR'
    elif dataset == 'birds_extended':
        indicator_columns = ['Trophic_Niche']
        continuous_columns = ['Dist_Hunters', 'TravDist', 'PopDens', 'Stunting', 'FoodBiomass', 
                              'Forest_Cover', 'NPP', 'Range_Size', 'Body_Mass']
        special_columns = ['Reserve', 'IUCN_Is_Hunted', 'IUCN_Is_Human_Food', 'IUCN_For_Pet_Trade', 
                           'IUCN_Is_Threatened', 'Habitat_Is_Dense']
        response_column = 'RR'
    elif dataset == 'mammals_extended':
        indicator_columns = ['Diet']
        continuous_columns = ['Body_Mass', 'Dist_Access_Pts', 'Travel_Time', 'Livestock_Biomass', 
                              'Stunting', 'Population_Density', 'Literacy', 'Forest_Cover', 'NPP']
        special_columns = ['Year', 'Reserve', 'Activity_Nocturnal', 'Activity_Crepuscular', 'Activity_Diurnal', 
                           'IUCN_Is_Hunted','IUCN_Is_Human_Food', 'IUCN_For_Handicrafts', 
                           'IUCN_For_Medicine', 'IUCN_For_Pet_Trade', 'IUCN_Is_Sport_Hunted', 
                           'IUCN_For_Wearing_Apparel', 'IUCN_Is_Threatened']
        response_column = 'Ratio'
    else:
        raise ValueError('The only supported datasets are "mammals," "birds," "both", "birds_extended", "mammals_extended", or "mammals_recreated".')

    # Grabbing just the continuous predictors
    pp_data = data[continuous_columns].copy(deep = True)

    # Optionally adding a polynomial basis expansion
    if polynomial_features > 1:
        poly = PolynomialFeatures(polynomial_features, include_bias = False)
        pp_data_poly = poly.fit_transform(pp_data)

        pp_data = pd.DataFrame(pp_data_poly, index = pp_data.index, columns = poly.get_feature_names_out())
        pp_data = pp_data.sort_index()

    # Optionally log10 transforming continuous predictors
    if log_trans_cont:
        for col in continuous_columns:
            pp_data.loc[pp_data[col] == 0, col] = 0.1 # ensuring we don't run into issues with log
            pp_data[col] = np.log10(pp_data[col].copy(deep = True))

    # Optionally standardizing continuous predictors
    if standardize:
        #  if we were supplied train indices, only using those stats for standardization
        if train_test_idxs is not None:
            scaler = StandardScaler()
            pp_train_scaled = scaler.fit_transform(pp_data.iloc[train_test_idxs['train']])
            pp_test_scaled = scaler.transform(pp_data.iloc[train_test_idxs['test']])

            pp_data_scaled = np.vstack((pp_train_scaled, pp_test_scaled))
        else:
            pp_data_scaled = StandardScaler().fit_transform(pp_data)

        idx = pp_data.index if train_test_idxs is None else np.concatenate((train_test_idxs['train'], train_test_idxs['test']))
        pp_data = pd.DataFrame(pp_data_scaled, index = idx, columns = pp_data.columns)
        pp_data = pp_data.sort_index()

    # Turning reserve into an indicator variable - not needed for bird dataset!
    if dataset in ['mammals', 'both', 'mammals_extended', 'mammals_recreated'] and 'Reserve' in special_columns:
        pp_data['Reserve'] = (data['Reserve'] == 'Yes').astype(int)

    # Adding other special columns as is
    for col in special_columns:
        if col not in pp_data.columns:
            pp_data[col] = data[col].copy(deep = True)

    # Optionally adding indicator (or straight categorical) variables for different groups present in data
    if include_indicators:
        pp_data = pd.concat((pp_data, data[indicator_columns].copy(deep = True).reset_index(drop = True)), axis = 1)
        pp_data = pd.get_dummies(pp_data, dtype = float, drop_first = True, columns = indicator_columns)
    elif include_categorical:
        pp_data = pd.concat((pp_data, data[indicator_columns].copy(deep = True).reset_index(drop = True)), axis = 1)

    # Optionally adding DL embeddings as predictors
    if embeddings_to_use is not None:
        all_embeddings = embeddings.get_all_embeddings(data, embeddings_to_use = embeddings_to_use,
                                                       train_test_idxs = train_test_idxs, **embeddings_args)
        pp_data = pd.concat((pp_data, all_embeddings), axis = 1) # this should be fine since both DFs are sorted by index

    # Add back in the response variable
    pp_data[response_column] = data[response_column].reset_index(drop = True)

    return pp_data

def match_to_closest_year(avail_years, study_years):

    """
    A helper function to match to study years to the closest available predictor
    years.

    Paramaters
    ----------
    avail_years : list
        the years for which we have datasets
    study_years : list
        the extracted study years, as integers

    Returns
    -------
    matched_years : numpy.array
        the closest match year amongst available years for each study
    """

    study_years = np.array(study_years).reshape(-1, 1)
    study_years = np.repeat(study_years, len(avail_years), axis = 1)

    avail_years = np.array(avail_years).reshape(1, -1)

    diff = np.abs(study_years - avail_years)
    closest_idx = np.argmin(diff, axis = 1)
    closest_idx = [[0 for i in range(len(closest_idx))], list(closest_idx)]

    matched_years = avail_years[closest_idx[0], closest_idx[1]]

    return matched_years

def extract_year(ref_str):

    """
    A helper function to extract the year value from the reference column.
    CURRENTLY ONLY TESTED ON BENITEZ-LOPEZ ET AL. (2019)!

    Paramaters
    ----------
    ref_str : string
        the reference string (e.g., 'Luz-Ricca et al., 2019')

    Returns
    -------
    year : integer
        the extracted study year
    """

    year = ref_str.split(' ')[-1].strip()
    year = ''.join([c for c in year if c.isnumeric()])
    year = int(year)

    return year

def get_train_test_split(len_dataset, train_size = 0.7):

    """
    A function to define a shared train/test split for all machine learning modelling.

    Paramaters
    ----------
    len_dataset : integer
        the length of the dataset to split
    train_size : float
        the proportion of data to use in the train set (between 0 and 1)

    Returns
    -------
    idxs : dictionary
        a dictionary with the train and test indices as entries
    """

    np.random.seed(1693)

    idxs = np.arange(0, len_dataset)
    np.random.shuffle(idxs)

    train_idxs = idxs[ : int(train_size * len(idxs))]
    test_idxs = idxs[int(train_size * len(idxs)) : ]

    idxs = {'train' : train_idxs, 'test' : test_idxs}

    return idxs

def direct_train_test(data, train_size = 0.7, task = 'classification', already_pp = False,
                      train_test_idxs = None, dataset = 'mammals'):

    """
    A helper function to get train/test data for direct regression or classification.

    Paramaters
    ----------
    data : pandas.DataFrame
        the dataset to split into train/test sets
    train_size : float
        the portion of data to assign to the train set
    task : string
        one of 'classification', 'regression', or 'joint'
    already_pp : boolean
        is the data already preprocessed?
    train_test_idxs : dictionary
        the training and testing indices, if already computed
    dataset : string
        the dataset being used, either 'mammals' (Benitez-Lopez et al., 2019) or
        'birds' (Ferreiro-Arias et al., 2024)

    Returns
    -------
    X_train : pandas.DataFrame
        the training predictors
    y_train : numpy.array
        the training response
    X_test : pandas.DataFrame
        the testing predictors
    y_test : numpy.array
        the testing response
    """

    # Getting the train/test split, if not supplied
    if train_test_idxs is None:
        train_test_idxs = get_train_test_split(len(data), train_size = train_size)

    # Pre-processing data, if not already preprocessed
    if not already_pp:
        pp_data = preprocess_data(data, include_indicators = False, standardize = True, log_trans_cont = False,
                                  polynomial_features = 0, train_test_idxs = train_test_idxs, dataset = dataset)
    else:
        pp_data = data.copy(deep = True)

    resp_col = 'ratio' if dataset in ['mammals', 'both'] else 'RR'
    if dataset == 'mammals_extended':
        resp_col = 'Ratio'
    elif dataset == 'mammals_recreated':
        resp_col = 'Response_Ratio'

    pp_data['DI_cat'] = ratios_to_DI_cats(pp_data[resp_col])
    pp_data['Local_Extirpation'] = (pp_data[resp_col] == 0).astype(int)

    # Splitting the dataset into train/test sets
    train_data, test_data = pp_data.iloc[train_test_idxs['train']], pp_data.iloc[train_test_idxs['test']]

    # Putting into the format that FLAML wants
    if task == 'regression':
        target_col = resp_col
    elif task == 'classification':
        target_col = 'DI_cat'
    elif task == 'joint':
        target_col = [resp_col, 'Local_Extirpation']

    drop_cols = [resp_col, 'DI_cat', 'Local_Extirpation']

    X_train, X_test = train_data.drop(columns = drop_cols), test_data.drop(columns = drop_cols)
    if task != 'joint':
        y_train, y_test = train_data[target_col].values, test_data[target_col].values
    else:
        y_train, y_test = train_data[target_col], test_data[target_col]

    return X_train, y_train, X_test, y_test

if __name__ == '__main__':
    ben_lop_rec_path = '/Users/emiliolr/Google Drive/My Drive/LIFE/datasets/derived_datasets/benitez_lopez2019_recreated/benitez_lopez2019_recreated_w_original.csv'
    data = pd.read_csv(ben_lop_rec_path)

    pp_data = preprocess_data(data, include_indicators = False, include_categorical = False,
                              standardize = True, log_trans_cont = False, polynomial_features = 0,
                              embeddings_to_use = None, embeddings_args = None, train_test_idxs = None,
                              dataset = 'mammals_recreated')
    
    X_zero, y_zero, X_nonzero, y_nonzero = get_zero_nonzero_datasets(pp_data, pred = False, outlier_cutoff = np.Inf, extirp_pos = False,
                                                                     zero_columns = None, nonzero_columns = None, indicator_columns = None,
                                                                     embeddings_to_use = None, dataset = 'mammals_recreated', 
                                                                     rebalance_dataset = True)

    print(X_zero.columns)
    print('\n\n\n')
    print(X_nonzero.columns)

    print()

    print(X_zero)
    print('\n\n\n')
    print(X_nonzero)