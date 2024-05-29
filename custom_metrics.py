import time

import numpy as np

from sklearn.metrics import recall_score, balanced_accuracy_score, mean_absolute_error

def mean_absolute_error_range(y_true, y_pred, lower_bound = 0, upper_bound = 1):

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
    pct_kept = len(y_true_sub) / len(y_true)

    return mae_range, pct_kept

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
