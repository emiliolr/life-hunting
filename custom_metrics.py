import numpy as np

from sklearn.metrics import recall_score

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
