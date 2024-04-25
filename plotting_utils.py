import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import recall_score, balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay

def plot_ratio_confusion_matrix(true_classes, pred_classes, display_labels = None):

    """
    ADD DESCRIPTION.

    Paramaters
    ----------

    Returns
    -------
    """

    if display_labels is None:
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

def plot_metrics_barplot(true_classes, pred_classes):

    """
    ADD DESCRIPTION + SUPPORT CROSS-VALIDATION!

    Paramaters
    ----------

    Returns
    -------
    """

    classes = [0, 1, 2]
    class_names = ['low', 'medium', 'high']
    metrics = []

    for c in classes:
        #  binarizing the true/pred labels
        true = (true_classes == c).astype(int)
        pred = (pred_classes == c).astype(int)

        #  calculating metrics
        sens = recall_score(true, pred, pos_label = 1)
        spec = recall_score(true, pred, pos_label = 0)
        balanced_acc = balanced_accuracy_score(true, pred)

        metrics.append([sens, spec, balanced_acc])

    #  adding in the positive class and putting into a dataframe for plotting
    metrics = pd.DataFrame(metrics, columns = ['sensitivity', 'specificity', 'balanced_accuracy'], index = class_names)
    metrics = metrics[['balanced_accuracy', 'sensitivity', 'specificity']]

    #  making the barplot
    colors = {'balanced_accuracy' : 'grey', 'sensitivity' : 'blue', 'specificity' : 'red'}
    metrics.plot.bar(rot = 0, color = colors, edgecolor = 'black');

    plt.ylim((0, 1))
    plt.xlabel('Defaunation Category', weight = 'bold')

    fig, ax = plt.gcf(), plt.gca()

    return fig, ax
