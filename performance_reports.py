import os

import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, median_absolute_error, root_mean_squared_error, \
                            classification_report, balanced_accuracy_score

from custom_metrics import root_median_squared_error
from plotting_utils import plot_ratio_confusion_matrix, plot_ratio_scatterplot, \
                           plot_ratio_distribution_comparison

def get_classification_report(true_DI_cats, pred_DI_cats, save_dir = None, model_name = None,
                              printout = True):

    """
    Produce a performance report for classification into defaunation intensity (DI)
    categories with all relevant metrics and plots.

    Paramaters
    ----------
    true_DI_cats : numpy.array
        an array of true DI categories
    pred_DI_cats : numpy.array
        an array of predicted DI categories
    save_dir : string
        the top-level directory to save in (e.g., 'model_saves')
    model_name : string
        the model name, which will be used as the subdirectory for saving results
    printout : boolean
        should we print results and display plots?

    Returns
    -------
    None
    """

    # Setting up parameters for saving the classification results
    if save_dir is not None:
        if model_name is None:
            raise ValueError('If you want to save the report, please supply "model_name"')

        base_dir = os.path.join(save_dir, model_name)

    save_str = ''

    # Sklearn classification report, which includes precision, recall, F1-score,
    #  and accuracy
    cls_report = classification_report(true_DI_cats, pred_DI_cats, target_names = ['low', 'medium', 'high'])
    save_str += cls_report + '\n'
    if printout:
        print(cls_report)

    # Compute metrics of interest based on Benitez-Lopez et al. (2019)
    balanced_acc = balanced_accuracy_score(true_DI_cats, pred_DI_cats)
    balanced_acc_str = f'Balanced accuracy: {round(balanced_acc * 100, 2)}%'
    save_str += balanced_acc_str + '\n'

    if printout:
        print(balanced_acc_str)

    # Generate a confusion matrix
    fig, ax = plot_ratio_confusion_matrix(true_DI_cats, pred_DI_cats)

    if save_dir is not None:
        plt.savefig(os.path.join(base_dir, 'confusion_matrix.png'),
                    bbox_inches = 'tight', dpi = 300)

    if printout:
        plt.show()
    else:
        plt.close()

    # Optionally saving the metric report
    if save_dir is not None:
        with open(os.path.join(base_dir, 'classification_report.txt'), 'w') as f:
            f.write(save_str)

def get_regression_report(true_ratios, pred_ratios, save_dir = None, model_name = None,
                          printout = True, upper_thresh = 1):

    """
    Produce a performance report for regressing abundance ratios with all relevant
    metrics and plots.

    Paramaters
    ----------
    true_ratios : numpy.array
        an array of true abundance ratios
    pred_ratios : numpy.array
        an array of predicted abundance ratios
    save_dir : string
        the top-level directory to save in (e.g., 'model_saves')
    model_name : string
        the model name, which will be used as the subdirectory for saving results
    printout : boolean
        should we print results and display plots?
    upper_thresh : float
        the upper limit for ratios to plot in the histogram/scatterplot

    Returns
    -------
    None
    """

    # Setting up parameters for saving the regression results
    if save_dir is not None:
        if model_name is None:
            raise ValueError('If you want to save the report, please supply "model_name"')

        base_dir = os.path.join(save_dir, model_name)

    save_str = ''

    # Compute regression metrics of interest
    mae = mean_absolute_error(true_ratios, pred_ratios)
    mae_str = f'Mean absolute error: {round(mae, 3)}'
    save_str += mae_str + '\n'

    med_ae = median_absolute_error(true_ratios, pred_ratios)
    med_ae_str = f'Median absolute error: {round(med_ae, 3)}'
    save_str += med_ae_str + '\n'

    rmse = root_mean_squared_error(true_ratios, pred_ratios)
    rmse_str = f'Root mean squared error: {round(rmse, 3)}'
    save_str += rmse_str + '\n'

    r_med_se = root_median_squared_error(true_ratios, pred_ratios)
    r_med_se_str = f'Root median squared error: {round(r_med_se, 3)}'
    save_str += r_med_se_str + '\n'

    if printout:
        print(mae_str)
        print(med_ae_str)
        print(rmse_str)
        print(r_med_se_str)

    # Plotting the distribution of ratios for the 0-1 range
    fig, ax = plot_ratio_distribution_comparison(true_ratios, pred_ratios, upper_thresh = upper_thresh)

    if save_dir is not None:
        plt.savefig(os.path.join(base_dir, 'ratio_distribution_0-1.png'),
                    bbox_inches = 'tight', dpi = 300)

    if printout:
        plt.show()
    else:
        plt.close()

    # Plotting a scatterplot of ratios for the 0-1 range
    fig, ax = plot_ratio_scatterplot(true_ratios, pred_ratios, upper_thresh = upper_thresh)

    if save_dir is not None:
        plt.savefig(os.path.join(base_dir, 'ratio_scatter_0-1.png'),
                    bbox_inches = 'tight', dpi = 300)

    if printout:
        plt.show()
    else:
        plt.close()

    # Optionally saving the metric report
    if save_dir is not None:
        with open(os.path.join(base_dir, 'regression_report.txt'), 'w') as f:
            f.write(save_str)

if __name__ == '__main__':
    import numpy as np

    cls = [0, 1, 2]
    true_DI_cats = np.random.choice(cls, size = 100)
    pred_DI_cats = np.random.choice(cls, size = 100)

    get_classification_report(true_DI_cats, pred_DI_cats, save_dir = 'model_saves', model_name = 'test', printout = True)

    true_ratios = np.random.normal(loc = 0.5, scale = 0.25, size = 100)
    pred_ratios = np.random.normal(loc = 0.5, scale = 0.3, size = 100)

    get_regression_report(true_ratios, pred_ratios, save_dir = 'model_saves', model_name = 'test', printout = True)
