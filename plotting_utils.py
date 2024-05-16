import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import recall_score, balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay

def plot_ratio_confusion_matrix(true_classes, pred_classes):

    """
    A ratio-based confusion matrix to match Supplementary Figure 6c of Benitez-Lopez
    et al. (2019). This is intended for training/testing on the full dataset, but in
    principle should work for a train/test split.

    Paramaters
    ----------
    true_classes : list
        the true defaunation categories
    pred_classes : list
        the defaunation categories predicted by the model

    Returns
    -------
    fig : matplotlib.Figure
        the figure object
    ax : matplotlib.Axes
        the axes object
    """

    display_labels = ['low', 'medium', 'high']

    cm = confusion_matrix(true_classes, pred_classes, labels = [0, 1, 2]) # getting the confusion matrix
    cm = cm.T # transposing to match the paper figure
    cm = cm / cm.sum(axis = 0) # turning elements into the proportion of predicted label for each true label

    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['low', 'medium', 'high'])
    disp.plot()

    plt.xlabel('True', weight = 'bold')
    plt.ylabel('Predicted', weight = 'bold')

    fig, ax = plt.gcf(), plt.gca()

    return fig, ax

def plot_model_coefficients(feature_names, coefficient_values):

    """
    A helper function to visualize model coefficients for generalized linear models.

    Paramaters
    ----------
    feature_names : list-like
        a list of feature names
    coefficient_values : list-like
        a list of the fitted coefficient values

    Returns
    -------
    fig : matplotlib.Figure
        the figure object
    ax : matplotlib.Axes
        the axes object
    """

    plt.scatter(feature_names, coefficient_values, s = 100, edgecolor = 'black', color = 'maroon', zorder = 100) # scatter plot at coefficient values

    plt.axhline(0, color = 'black')
    plt.vlines(feature_names, 0, coefficient_values, color = 'black', zorder = 0) # adding sticks to plot

    plt.ylabel('Coefficient Estimate', weight = 'bold')

    fig, ax = plt.gcf(), plt.gca()

    return fig, ax

def plot_ratio_distribution_comparison(true_ratios, pred_ratios, upper_thresh = 1, n_bins = 20):

    """
    A function to plot a comparison of abundance ratio distributions between predicted
    and true, from zero to a chosen upper threshold.

    Paramaters
    ----------
    true_ratios : numpy.array
        an array of true abundance ratios
    pred_ratios : numpy.array
        an array of predicted abundance ratios
    upper_thresh : float
        the upper threshold for ratios to plot
    n_bins : int
        the number of bins to use in the histogram

    Returns
    -------
    fig : matplotlib.Figure
        the figure object
    ax : matplotlib.Axes
        the axes object
    """

    fig, ax = plt.subplots(2)

    pred_ratios_0_1 = pred_ratios[pred_ratios <= upper_thresh]
    ax[0].hist(pred_ratios_0_1, edgecolor = 'black', bins = n_bins)
    ax[0].set_title('Predicted Abundance Ratios', weight = 'bold')

    true_ratios_0_1 = true_ratios[true_ratios <= upper_thresh]
    ax[1].hist(true_ratios_0_1, edgecolor = 'black', bins = n_bins)
    ax[1].set_title('True Abundance Ratios', weight = 'bold')

    fig.tight_layout()

    return fig, ax

def plot_ratio_scatterplot(true_ratios, pred_ratios, true_upper_thresh = 1, alpha = 0.1):

    """
    A function to plot the true vs. predicted abundance ratios as a scatterplot,
    from zero to a chosen upper threshold.

    Paramaters
    ----------
    true_ratios : numpy.array
        an array of true abundance ratios
    pred_ratios : numpy.array
        an array of predicted abundance ratios
    true_upper_thresh : float
        the upper threshold for (true) ratios to plot
    alpha : float
        the alpha value for points in the scatterplot

    Returns
    -------
    fig : matplotlib.Figure
        the figure object
    ax : matplotlib.Axes
        the axes object
    """

    threshold_mask = (true_ratios <= true_upper_thresh)

    plt.scatter(pred_ratios[threshold_mask], true_ratios[threshold_mask], alpha = alpha)

    plt.xlabel('Predicted Abundance Ratio', weight = 'bold')
    plt.ylabel('True Abundance Ratio', weight = 'bold')

    fig, ax = plt.gcf(), plt.gca()

    return fig, ax
