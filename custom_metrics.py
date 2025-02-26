import time

import numpy as np

from sklearn.metrics import recall_score, balanced_accuracy_score, mean_absolute_error, median_absolute_error
from scipy.stats import ks_2samp, wasserstein_distance

def mean_absolute_error_range(y_true, y_pred, lower_bound = 0, upper_bound = 1, return_pct_kept = True):

    """
    A function to compute the mean absolute error for a specified range of true
    values for the response variable.

    Paramaters
    ----------
    y_true : numpy.array
        an of true continuous labels
    y_pred : numpy.array
        an array of predicted continuous labels
    lower_bound : float
        the lower bound on indices to keep (based on labels)
    upper_bound : float
        the upper bound on indices to keep (based on labels)
    return_pct_kept : boolean
        whether to return the percent of data points used in the operation

    Returns
    -------
    mae_range : float
        the mean absolute error for the given range
    pct_kept : float
        the percent of data points kept after subsetting to the given range
    """

    bound_mask = (y_true >= lower_bound) & (y_true <= upper_bound)
    y_true_sub = y_true[bound_mask]
    y_pred_sub = y_pred[bound_mask]

    mae_range = mean_absolute_error(y_true_sub, y_pred_sub)
    
    if return_pct_kept:
        pct_kept = len(y_true_sub) / len(y_true)
        return mae_range, pct_kept
    return mae_range

def median_absolute_error_range(y_true, y_pred, lower_bound = 0, upper_bound = 1, return_pct_kept = True):

    """
    A function to compute the median absolute error for a specified range of true
    values for the response variable.

    Paramaters
    ----------
    y_true : numpy.array
        an of true continuous labels
    y_pred : numpy.array
        an array of predicted continuous labels
    lower_bound : float
        the lower bound on indices to keep (based on labels)
    upper_bound : float
        the upper bound on indices to keep (based on labels)
    return_pct_kept : boolean
        whether to return the percent of data points used in the operation

    Returns
    -------
    mae_range : float
        the median absolute error for the given range
    pct_kept : float
        the percent of data points kept after subsetting to the given range
    """

    bound_mask = (y_true >= lower_bound) & (y_true <= upper_bound)
    y_true_sub = y_true[bound_mask]
    y_pred_sub = y_pred[bound_mask]

    mae_range = median_absolute_error(y_true_sub, y_pred_sub)
    
    if return_pct_kept:
        pct_kept = len(y_true_sub) / len(y_true)
        return mae_range, pct_kept
    return mae_range

def root_median_squared_error(y_true, y_pred):

    """
    A function to compute root median squared error.

    Paramaters
    ----------
    y_true : numpy.array
        an array of true continuous labels
    y_pred : numpy.array
        an array of predicted continuous labels

    Returns
    -------
    r_med_se : float
        the computed root median squared error
    """

    r_med_se = np.sqrt(np.median(np.square(y_true - y_pred)))

    return r_med_se

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

def balanced_accuracy_FLAML(X_val, y_val, estimator, labels, X_train, y_train,
                            weight_val = None, weight_train = None, *args):

    """
    A wrapper function to port balanced accuracy to FLAML. See this page for details
    on parameters and returns: https://microsoft.github.io/FLAML/docs/Use-Cases/Task-Oriented-AutoML/#optimization-metric.
    """

    start = time.time()
    y_pred = estimator.predict(X_val)
    pred_time = (time.time() - start) / len(X_val)
    val_acc = 1.0 - balanced_accuracy_score(y_val, y_pred)

    return val_acc, {'val_acc' : val_acc, 'pred_time' : pred_time}

def median_absolute_error_FLAML(X_val, y_val, estimator, labels, X_train, y_train,
                                weight_val = None, weight_train = None, *args):

    """
    A wrapper function to port median absolute error to FLAML. See this page for details
    on parameters and returns: https://microsoft.github.io/FLAML/docs/Use-Cases/Task-Oriented-AutoML/#optimization-metric.
    """

    start = time.time()
    y_pred = estimator.predict(X_val)
    pred_time = (time.time() - start) / len(X_val)
    val_med_ae = median_absolute_error(y_val, y_pred)

    return val_med_ae, {'val_acc' : val_med_ae, 'pred_time' : pred_time}

def mean_absolute_percent_error_tau(y_true, y_pred, tau = 0, return_pct_kept = False, epsilon = 1e-4):

    """
    Compute a modified mean absolute percent error metric, which excludes observations with
    y_true > tau to prevent the metric from exploding.

    Parameters
    ----------
    y_true : numpy.array
        an array of true continuous labels
    y_pred : numpy.array
        an array of predicted continuous labels
    tau : float
        a value >0, below which observations with y_true < tau will be ignored in the calculation
    return_pct_kept : boolean
        should we return the percent of the original data used in the calculation?
    epsilon : float
        a (small) value >0, used as the denominator of the calculation to avoid dividing by zero

    Returns
    -------
    mape_tau : float
        the calculated mean absolute percent error using the given tau
    pct_kept : float
        the percent of data points kept after subsetting to the given range
    """
    
    # Converting to numpy arrays
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # Filtering out observations w/true RR smaller than tau
    bound_mask = (y_true >= tau)
    y_true_sub = y_true[bound_mask]
    y_pred_sub = y_pred[bound_mask]

    # Calculating the MAPE on this data subset
    mape_tau = np.mean(np.abs((y_pred_sub - y_true_sub)) / (np.maximum(y_true_sub, epsilon)))
    mape_tau = float(mape_tau)
    pct_kept = len(y_true_sub) / len(y_true)

    if return_pct_kept:
        return mape_tau, pct_kept
    
    return mape_tau

def get_DI_cats(ratios, neighborhood = 0, numeric = False):

    # Setting up an empty vector to hold DI category values
    DI_cat = np.zeros_like(ratios)
    if not numeric:
        DI_cat = DI_cat.astype(str)
    
    # Categorizing continuous input by defaunation intensity category
    DI_cat[ratios <= (0 + neighborhood)] = 0 if numeric else 'extirpated'
    DI_cat[(ratios > (0 + neighborhood)) & (ratios < (1 - neighborhood))] = 1 if numeric else 'decrease'
    DI_cat[(ratios >= (1 - neighborhood)) & (ratios <= (1 + neighborhood))] = 2 if numeric else 'no change'
    DI_cat[ratios > (1 + neighborhood)] = 3 if numeric else 'increase'

    return DI_cat

def balanced_accuracy_DI_cats(y_true, y_pred, neighborhood = 0.05):

    # Turning continuous RRs into DI categories
    y_true_cats = get_DI_cats(y_true, neighborhood = neighborhood, numeric = True)
    y_pred_cats = get_DI_cats(y_pred, neighborhood = neighborhood, numeric = True)

    # Calculating balanced accuracy on these categories
    ba = balanced_accuracy_score(y_true_cats, y_pred_cats)

    return ba

def wasserstein_distance_range(y_true, y_pred, lower_bound = 0, upper_bound = 2):

    # Pulling out just the range of values that we're interested in
    bound_mask = (y_true >= lower_bound) & (y_true <= upper_bound)
    y_true_sub = y_true[bound_mask]
    y_pred_sub = y_pred[bound_mask]

    # Compute the Wasserstein Distance on this subset of predictions
    wd_range = wasserstein_distance(y_true_sub, y_pred_sub)

    return wd_range

if __name__ == '__main__':
    import pandas as pd

    np.random.seed(123)

    RRs = np.abs(np.random.normal(0.75, 0.5, 100))
    print(RRs)

    DIs = pd.Series(get_DI_cats(RRs, neighborhood = 0.05))
    print(DIs.value_counts())