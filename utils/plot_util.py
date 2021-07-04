from sklearn.metrics import *
from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
from numpy import interp
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
from utils import evaluations


def plot_confusion_matrix(y_true, y_pred, classes, save_path,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    np.set_printoptions(precision=2)
    plt.show()
    fig.savefig(save_path)
    return ax


def precision_recall_graph(Y_test, y_score, classes, method='cnn'):

    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i, _ in enumerate(classes):
        precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],
                                                            y_score[:, i])
        average_precision[i] = precision_score(Y_test[:, i], np.round(y_score[:, i]))

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
                                                                    y_score.ravel())
    average_precision["micro"] = precision_score(Y_test, np.round(y_score),
                                                         average="micro")

    auc_score = auc(recall["micro"], precision["micro"] )
    print('PR AUC: %.3f' % auc_score)

    # A "micro-average": quantifying score on all classes jointly
    # precision["macro"], recall["macro"], _ = precision_recall_curve(Y_test.ravel(),
    #                                                                 y_score.ravel())
    # average_precision["macro"] = average_precision_score(Y_test, y_score,
    #                                                      average="macro")

    print('Average precision score, micro-averaged over all classes: {0:0.2f}'
          .format(average_precision["micro"]))

    plt.figure()
    plt.step(recall['micro'], precision['micro'], where='post', label=f'micro-averaged precision score'
                                                                      f'(area= {auc_score:0.2f})')
    # plt.step(recall['macro'], precision['macro'], where='post',
    #          label=f'Average precision score, macro-averaged over all classes '
    #                f'(area = {average_precision["macro"]:0.2f})')

    for i, target in enumerate(classes):
        plt.step(recall[i], precision[i], where='post', label=f'class {target} precision '
                                                              f'{average_precision[i]:0.2f}')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.legend(fontsize='x-small')
    plt.title(f'Precision Recall graph {method}')
    plt.show()


def calc_and_plot_roc(y_test, y_score, save_path, model_name, classes):
    # y_test = test_path
    # y_score = pred_path
    # y_test = pd.read_csv(test_path, index_col=0)
    # y_score = pd.read_csv(pred_path, index_col=0)

    evaluations.plot_confusion_matrix(y_test, y_score, classes=classes)
    # Binarize the output
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_score = lb.transform(y_score)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = len(classes)

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    ##############################################################################
    # Plot ROC curves for the multiclass problem

    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw = 2
    fig = plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(model_name + ' Average ROC curve')
    plt.legend(loc="lower right")
    # plt.show()
    fig.savefig(f'{save_path}/average_ROC_{model_name}.png')

    colors = cycle(['b', 'g', 'r', 'c', 'm', 'y', 'navy'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw)
                # label='ROC curve of class {0} (area = {1:0.2f})'
                #       ''.format(classes_mapping.map.get(i), roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(model_name + ' ROC curves per class')
    plt.legend(loc="lower right")
    # plt.show()

    plt.savefig(f'{save_path}/ROC_per_class_{model_name}.png')


def plot_data_dist(value_count, section_size):
    """
    plot data distribution by the given value count
    :param value_count: series of the label and count
    :param section_size: number of seconds for the data
    :return: void
    """
    (value_count / (section_size * 10)).sort_index().plot.barh()
    plt.xlabel('Number of time series samples')
    plt.ylabel('Driver ID')
    plt.show()
