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
        the top-level directory to save in (e.g., "model_saves")
    model_name : string
        the model name, which will be used as the subdirectory for saving results
    printout : boolean
        should we print results?

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

    plt.close()

    # Optionally saving the metric report
    with open(os.path.join(base_dir, 'classification_report.txt'), 'w') as f:
        f.write(save_str)

def get_regression_report(true_ratios, pred_ratios):
    pass

if __name__ == '__main__':
    import numpy as np

    cls = [0, 1, 2]
    true_DI_cats = np.random.choice(cls, size = 100)
    pred_DI_cats = np.random.choice(cls, size = 100)

    get_classification_report(true_DI_cats, pred_DI_cats, save_dir = 'model_saves', model_name = 'test', printout = True)
