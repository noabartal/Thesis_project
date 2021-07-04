import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
from utils.config import CLASSIFIERS, FEATURES, ITERS, SELECTION
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold, StratifiedKFold
import os
import pandas as pd
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import average_precision_score, accuracy_score
from scipy.stats import wilcoxon
from scipy.stats import friedmanchisquare
import operator
import math
import networkx

from utils.model_utils import create_target


def form_cliques(p_values, nnames):
    """
    This method forms the cliques
    """
    # first form the numpy matrix data
    m = len(nnames)
    g_data = np.zeros((m, m), dtype=np.int64)
    for p in p_values:
        if p[3] == False:
            i = np.where(nnames == p[0])[0][0]
            j = np.where(nnames == p[1])[0][0]
            min_i = min(i, j)
            max_j = max(i, j)
            g_data[min_i, max_j] = 1

    g = networkx.Graph(g_data)
    return networkx.find_cliques(g)


def graph_ranks(avranks, names, p_values, cd=None, cdmethod=None, lowv=None, highv=None,
                width=6, textspace=1, reverse=False, filename=None, labels=False, **kwargs):
    """
    Draws a CD graph, which is used to display  the differences in methods'
    performance. See Janez D                        if NORMALIZE:
                            params += 'norm'emsar, Statistical Comparisons of Classifiers over
    Multiple Data Sets, 7(Jan):1--30, 2006.
    Needs matplotlib to work.
    The image is ploted on `plt` imported using
    `import matplotlib.pyplot as plt`.
    Args:
        avranks (list of float): average ranks of methods.
        names (list of str): names of methods.
        cd (float): Critical difference used for statistically significance of
            difference between methods.
        cdmethod (int, optional): the method that is compared with other methods
            If omitted, show pairwise comparison of methods
        lowv (int, optional): the lowest shown rank
        highv (int, optional): the highest shown rank
        width (int, optional): default width in inches (default: 6)
        textspace (int, optional): space on figure sides (in inches) for the
            method names (default: 1)
        reverse (bool, optional):  if set to `True`, the lowest rank is on the
            right (default: `False`)
        filename (str, optional): output file name (with extension). If not
            given, the function does not write a file.
        labels (bool, optional): if set to `True`, the calculated avg rank
        values will be displayed
    """
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
    except ImportError:
        raise ImportError("Function graph_ranks requires matplotlib.")

    width = float(width)
    textspace = float(textspace)

    def nth(l, n):
        """
        Returns only nth elemnt in a list.
        """
        n = lloc(l, n)
        return [a[n] for a in l]

    def lloc(l, n):
        """
        List location in list of list structure.
        Enable the use of negative locations:
        -1 is the last element, -2 second last...
        """
        if n < 0:
            return len(l[0]) + n
        else:
            return n

    def mxrange(lr):
        """
        Multiple xranges. Can be used to traverse matrices.
        This function is very slow due to unknown number of
        parameters.
        >>> mxrange([3,5])
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        >>> mxrange([[3,5,1],[9,0,-3]])
        [(3, 9), (3, 6), (3, 3), (4, 9), (4, 6), (4, 3)]
        """
        if not len(lr):
            yield ()
        else:
            # it can work with single numbers
            index = lr[0]
            if isinstance(index, int):
                index = [index]
            for a in range(*index):
                for b in mxrange(lr[1:]):
                    yield tuple([a] + list(b))

    def print_figure(fig, *args, **kwargs):
        canvas = FigureCanvasAgg(fig)
        canvas.print_figure(*args, **kwargs)

    sums = avranks

    nnames = names
    ssums = sums

    if lowv is None:
        lowv = min(1, int(math.floor(min(ssums))))
    if highv is None:
        highv = max(len(avranks), int(math.ceil(max(ssums))))

    cline = 0.4

    k = len(sums)

    lines = None

    linesblank = 0
    scalewidth = width - 2 * textspace

    def rankpos(rank):
        if not reverse:
            a = rank - lowv
        else:
            a = highv - rank
        return textspace + scalewidth / (highv - lowv) * a

    distanceh = 0.25

    cline += distanceh

    # calculate height needed height of an image
    minnotsignificant = max(2 * 0.2, linesblank)
    height = cline + ((k + 1) / 2) * 0.2 + minnotsignificant

    fig = plt.figure(0, figsize=(width, height))
    fig.set_facecolor('white')
    ax = fig.add_axes([0, 0, 1, 1])  # reverse y axis
    ax.set_axis_off()

    hf = 1. / height  # height factor
    wf = 1. / width

    def hfl(l):
        return [a * hf for a in l]

    def wfl(l):
        return [a * wf for a in l]

    # Upper left corner is (0,0).
    ax.plot([0, 1], [0, 1], c="w")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)

    def line(l, color='k', **kwargs):
        """
        Input is a list of pairs of points.
        """
        ax.plot(wfl(nth(l, 0)), hfl(nth(l, 1)), color=color, **kwargs)

    def text(x, y, s, *args, **kwargs):
        ax.text(wf * x, hf * y, s, *args, **kwargs)

    line([(textspace, cline), (width - textspace, cline)], linewidth=2)

    bigtick = 0.3
    smalltick = 0.15
    linewidth = 2.0
    linewidth_sign = 4.0

    tick = None
    for a in list(np.arange(lowv, highv, 0.5)) + [highv]:
        tick = smalltick
        if a == int(a):
            tick = bigtick
        line([(rankpos(a), cline - tick / 2),
              (rankpos(a), cline)],
             linewidth=2)

    for a in range(lowv, highv + 1):
        text(rankpos(a), cline - tick / 2 - 0.05, str(a),
             ha="center", va="bottom", size=16)

    k = len(ssums)

    def filter_names(name):
        return name

    space_between_names = 0.24

    for i in range(math.ceil(k / 2)):
        chei = cline + minnotsignificant + i * space_between_names
        line([(rankpos(ssums[i]), cline),
              (rankpos(ssums[i]), chei),
              (textspace - 0.1, chei)],
             linewidth=linewidth)
        if labels:
            text(textspace + 0.3, chei - 0.075, format(ssums[i], '.4f'), ha="right", va="center", size=10)
        text(textspace - 0.2, chei, filter_names(nnames[i]), ha="right", va="center", size=16)

    for i in range(math.ceil(k / 2), k):
        chei = cline + minnotsignificant + (k - i - 1) * space_between_names
        line([(rankpos(ssums[i]), cline),
              (rankpos(ssums[i]), chei),
              (textspace + scalewidth + 0.1, chei)],
             linewidth=linewidth)
        if labels:
            text(textspace + scalewidth - 0.3, chei - 0.075, format(ssums[i], '.4f'), ha="left", va="center", size=10)
        text(textspace + scalewidth + 0.2, chei, filter_names(nnames[i]),
             ha="left", va="center", size=16)

    # no-significance lines
    def draw_lines(lines, side=0.05, height=0.1):
        start = cline + 0.2

        for l, r in lines:
            line([(rankpos(ssums[l]) - side, start),
                  (rankpos(ssums[r]) + side, start)],
                 linewidth=linewidth_sign)
            start += height
            print('drawing: ', l, r)

    # draw_lines(lines)
    start = cline + 0.2
    side = -0.02
    height = 0.1

    # draw no significant lines
    # get the cliques
    cliques = form_cliques(p_values, nnames)
    i = 1
    achieved_half = False
    print(nnames)
    for clq in cliques:
        if len(clq) == 1:
            continue
        print(clq)
        min_idx = np.array(clq).min()
        max_idx = np.array(clq).max()
        if min_idx >= len(nnames) / 2 and achieved_half == False:
            start = cline + 0.25
            achieved_half = True
        line([(rankpos(ssums[min_idx]) - side, start),
              (rankpos(ssums[max_idx]) + side, start)],
             linewidth=linewidth_sign)
        start += height


def predict_on_given_set(y_true, y_pred, classes, multilabel):
    tp_amount = []
    for idx, target in enumerate(classes):
        y_true_i = y_true[:, idx]
        y_pred_i = y_pred[:, idx]

        # y_pred_i[y_pred_i >= 0.5] = 1
        # y_pred_i[y_pred_i < 0.5] = 0

        tp = np.sum(np.clip(y_true_i * np.round(y_pred_i), 0, 1))
        # check what happens when it is negative
        fp = np.sum(np.clip(np.round(y_pred_i) - y_true_i, 0, 1))
        # fn = np.sum(np.round(np.clip(y_true_i - y_pred_i, 0, 1)), axis=1)
        # calculate precision
        p = tp / (tp + fp)
        # print(f'precision of {target}:{p}')
        tp_amount += [tp]
    if multilabel:
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0

        print(f'average_precision_score micro:{precision_score(y_true, y_pred, average="micro")}')
        print(f'average_precision_score macro:{precision_score(y_true, y_pred, average="macro")}')
        print(f'recall_score mean micro:{recall_score(y_true, y_pred, average="micro")}')
        print(f'recall_score mean macro:{recall_score(y_true, y_pred, average="macro")}')
        print(tp_amount)
    else:
        y_pred = np.argmax(y_pred, axis=-1)
        y_test = np.argmax(y_true, axis=-1)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'accuracy score:{accuracy}')

        plot_confusion_matrix_sns(y_test, y_pred, classes=np.unique(y_test),
                                              partial=8, title='Baseline 1 confusion matrix')

        print("Accuracy: %.2f%%" % (accuracy * 100.0))
        return accuracy

    return 0


def get_majority_class_for_baselines(baseline, complete_test_df, file_id, y, true_baseline, pred_baseline):
    baselines = complete_test_df.groupby(file_id)

    for index, (baseline_name, baseline_df) in enumerate(baselines):
        # choose majority
        common_class_true = y[baseline_df.index].mean(axis=0)
        max_element = np.bincount(baseline_df[list(range(y.shape[1]))].values.argmax(axis=1)).argmax()
        common_class_predicted = np.zeros(shape=30)
        common_class_predicted[max_element] = 1
        true_baseline[baseline * baselines.ngroups + index] = common_class_true
        pred_baseline[baseline * baselines.ngroups + index] = common_class_predicted
    return true_baseline, pred_baseline


def get_distribution_summation_class_for_baselines(baseline, complete_test_df, file_id,
                                                   y, true_baseline, pred_baseline):
    baselines = complete_test_df.groupby(file_id)

    for index, (baseline_name, baseline_df) in enumerate(baselines):
        # choose majority
        class_true = y[baseline_df.index].mean(axis=0)
        common_class_predicted = baseline_df[list(range(y.shape[1]))].values.mean(axis=0)
        true_baseline[baseline * baselines.ngroups + index] = class_true
        pred_baseline[baseline * baselines.ngroups + index] = common_class_predicted
    return pred_baseline, true_baseline


def evaluate(model, x_test, y_test, unique_test, record, label, output_path, multilabel=False):

    test_pred, test_true = model.predict(x_test), y_test
    if test_pred.shape[1] == 1 or test_pred.shape[0] == 1:
        test_pred, test_true, _, classes = create_target(test_true, test_pred, test_true)

    else:
        np.save(os.path.join(output_path, 'x_test_split.npy'), x_test[0])
        np.save(os.path.join(output_path, 'x_test_agg_split.npy'), x_test[1])

    true_baseline = np.zeros((unique_test.drop_duplicates().shape[0], test_pred.shape[1]))
    pred_baseline = np.zeros((unique_test.drop_duplicates().shape[0], test_pred.shape[1]))

    y_pred_baseline, y_true_baseline = get_distribution_summation_class_for_baselines(0, pd.concat(
        [unique_test.reset_index(drop=True), pd.DataFrame(
            test_pred)], axis=1), [record, label], test_true, true_baseline, pred_baseline)

    true_baseline = np.zeros((unique_test.drop_duplicates().shape[0], test_pred.shape[1]))
    pred_baseline = np.zeros((unique_test.drop_duplicates().shape[0], test_pred.shape[1]))

    y_pred_maj, y_true_maj = get_majority_class_for_baselines(0, pd.concat(
        [unique_test.reset_index(drop=True), pd.DataFrame(
            test_pred)], axis=1), [record, label], test_true, true_baseline, pred_baseline)

    result_dict = {}
    # compute distribution summation accuracy
    result_dict['distribution_summation_ACCURACY'] = [predict_on_given_set(y_true_baseline, y_pred_baseline.copy(),
                                                                           classes, multilabel)]

    # compute majority vote accuracy
    result_dict['majority_vote_ACCURACY'] = [predict_on_given_set(y_true_maj, y_pred_maj.copy(),
                                                                  classes, multilabel)]

    # compute each baseline accuracy
    result_dict['baseline_ACCURACY'] = [predict_on_given_set(test_true, test_pred, classes, multilabel)]

    results = pd.DataFrame.from_dict(result_dict, orient='columns')
    results.to_csv(os.path.join(output_path, 'ACCURACY.csv'))

    return test_pred

def evaluate_from_file(model, y_test, folder_path):
    x_test = np.load(os.path.join(folder_path, 'x_test_split.npy'))
    x_test_agg = np.load(os.path.join(folder_path, 'x_test_agg_split.npy'))
    # model.summary()
    test_pred = model.predict([x_test, x_test_agg])

    test_pred_real = test_pred[np.arange(len(test_pred)), y_test]

    return test_pred_real


def create_results_file(results_path, tf, object_combined):
    res = pd.DataFrame(data=np.zeros((0, 6), dtype=np.float), index=[],
                       columns=['classifier_name', 'distribution_summation_ACCURACY', 'majority_vote_ACCURACY',
                                'baseline_ACCURACY', 'feature_selection_method', 'features'])
    res_per_sample = pd.DataFrame()
    _, _, unique_test = object_combined.split_and_scale_data(0, 'ufs')

    for classifier_name in CLASSIFIERS:
        results_folder = classifier_name

        print('classifier_name', classifier_name)

        for iter in range(0, ITERS):
            tmp_0_folder_path = results_folder
            tmp_0_folder_path += f'_iter_{iter}'
            for feature in FEATURES:
                tmp_folder_path = tmp_0_folder_path
                if feature != 0:
                    if 'extension' not in classifier_name:
                        continue
                    params = '_f_' + str(feature)
                    tmp_folder_path += params
                elif 'extension' in classifier_name:
                    continue

                for i, met in enumerate(SELECTION):
                    tmp_2_folder_path = tmp_folder_path
                    if 'extension' not in classifier_name:
                        if i > 0:
                            continue
                    else:
                        tmp_2_folder_path += '_' + met

                    results_folder_path = f"{tmp_2_folder_path}_{tf}"
                    results_folder_path = os.path.join(results_path, results_folder_path)
                    if not os.path.exists(f'{results_folder_path}/ACCURACY.csv'):
                        continue
                    df_metrics = pd.read_csv(f'{results_folder_path}/ACCURACY.csv')
                    df_metrics['classifier_name'] = classifier_name
                    df_metrics['feature_selection_method'] = met
                    df_metrics['features'] = feature

                    best_model = object_combined.load_model(results_folder_path)
                    probabilities = evaluate_from_file(best_model, np.argmax(object_combined.y_test, axis=1), results_folder_path)
                    df_prob = pd.DataFrame(probabilities, columns=['prob'])
                    df_prob['data_index'] = range(df_prob.shape[0])
                    df_prob['classifier_name'] = classifier_name
                    df_prob['feature_selection_method'] = met
                    df_prob['features'] = feature

                    res_per_sample = pd.concat((res_per_sample, df_prob), axis=0, sort=False)
                    res = pd.concat((res, df_metrics), axis=0, sort=False)

    res.to_csv("Results/results_raw.csv", index=False)
    # aggreagte the accuracy for iterations on same dataset
    # res = pd.DataFrame({
    #     'accuracy': res.groupby(
    #         ['classifier_name', 'archive_name', 'dataset_name'])['accuracy'].mean()
    # }).reset_index()
    res_per_sample_agg = res_per_sample.groupby(
        ['classifier_name', 'feature_selection_method', 'features', 'data_index'])['prob'].agg(
        ['mean'])
    res_per_sample_agg.reset_index().to_csv(f'Results/results_per_sample_agg_{"".join(CLASSIFIERS)}.csv',
                                 index=False)
    res_agg = res.groupby(
        ['classifier_name', 'feature_selection_method', 'features'])['distribution_summation_ACCURACY'].agg(
        ['mean', 'median', 'std', 'max'])
    res_agg['res'] = (res_agg['mean'] * 100).round(1).astype(str) + '(' + \
                     (res_agg['std'] * 100).round(1).astype(str) + ')'
    res_agg.reset_index().to_csv(f'Results/results_agg_{"".join(CLASSIFIERS)}.csv',
                                 index=False)
    return res


def wilcoxon_holm(alpha=0.05, df_perf=None, test_measure='max'):
    """
    Applies the wilcoxon signed rank test between each pair of algorithm and then use Holm
    to reject the null's hypothesis
    """
    # df_perf = df_perf[df_perf['classifier_name'].str.contains('resnet')]

    print(pd.unique(df_perf['classifier_name']))
    # count the number of tested datasets per classifier
    df_counts = pd.DataFrame({'count': df_perf.groupby(
        ['classifier_name']).size()}).reset_index()

    # get the maximum number of tested datasets
    # resnet = df_perf[df_perf['classifier_name'] == 'my_model_resnet']
    max_nb_datasets = df_counts['count'].max()
    df_perf = df_perf.groupby(
        ['classifier_name']).filter(lambda x: len(x) == max_nb_datasets)
    # get the list of classifiers who have been tested on nb_max_datasets
    classifiers = list(df_counts.loc[df_counts['count'] == max_nb_datasets]
                       ['classifier_name'])
    # test the null hypothesis using friedman before doing a post-hoc analysis
    if len(classifiers) > 2:
        friedman_p_value = friedmanchisquare(*(
            np.array(df_perf.loc[df_perf['classifier_name'] == c][test_measure])
            for c in classifiers))[1]
        if friedman_p_value >= alpha:
            # then the null hypothesis over the entire classifiers cannot be rejected
            print(friedman_p_value)
            print('the null hypothesis over the entire classifiers cannot be rejected')
            # exit()
    # get the number of classifiers
    m = len(classifiers)

    # init array that contains the p-values calculated by the Wilcoxon signed rank test
    p_values = []
    # loop through the algorithms to compare pairwise
    for i in range(m - 1):
        # get the name of classifier one
        classifier_1 = classifiers[i]
        # get the performance of classifier one
        perf_1 = np.array(df_perf.loc[df_perf['classifier_name'] == classifier_1][test_measure]
                          , dtype=np.float64)
        for j in range(i + 1, m):
            # get the name of the second classifier
            classifier_2 = classifiers[j]
            # get the performance of classifier one
            perf_2 = np.array(df_perf.loc[df_perf['classifier_name'] == classifier_2]
                              [test_measure], dtype=np.float64)
            # calculate the p_value
            p_value = wilcoxon(perf_1, perf_2, zero_method='pratt')[1]
            # append to the list
            p_values.append((classifier_1, classifier_2, p_value, False))
    # get the number of hypothesis
    k = len(p_values)
    # sort the list in acsending manner of p-value
    p_values.sort(key=operator.itemgetter(2))

    # loop through the hypothesis
    for i in range(k):
        # correct alpha with holm
        new_alpha = float(alpha / (k - i))
        # test if significant after holm's correction of alpha
        if p_values[i][2] <= new_alpha:
            p_values[i] = (p_values[i][0], p_values[i][1], p_values[i][2], True)
        else:
            # stop
            break
    # compute the average ranks to be returned (useful for drawing the cd diagram)
    # sort the dataframe of performances
    sorted_df_perf = df_perf.loc[df_perf['classifier_name'].isin(classifiers)]. \
        sort_values(['classifier_name', 'data_index'])
    # get the rank data
    rank_data = np.array(sorted_df_perf[test_measure]).reshape(m, max_nb_datasets)

    # create the data frame containing the accuracies
    df_ranks = pd.DataFrame(data=rank_data, index=np.sort(classifiers),
                            columns=np.unique(sorted_df_perf['data_index']))

    # number of wins
    dfff = df_ranks.rank(ascending=False)
    print(dfff[dfff == 1.0].sum(axis=1))
    for index_of_df in range(1, df_ranks.shape[0]):
        current_dfff = df_ranks.iloc[[0, index_of_df], :].rank(ascending=False)
        print(current_dfff[current_dfff == 1.0].sum(axis=1))
    # average the ranks
    average_ranks = df_ranks.rank(ascending=False).mean(axis=1).sort_values(ascending=False)

    # draw_dots_diagram(df_ranks, classifiers)
    # return the p-values and the average ranks
    return p_values, average_ranks, max_nb_datasets


def draw_cd_diagram(df_perf=None, alpha=0.05, title=None, labels=False, test_measure='max'):
    """
    Draws the critical difference diagram given the list of pairwise classifiers that are
    significant or not
    """
    p_values, average_ranks, _ = wilcoxon_holm(df_perf=df_perf, alpha=alpha, test_measure=test_measure)

    print(average_ranks)

    for p in p_values:
        print(p)

    graph_ranks(average_ranks.values, average_ranks.keys(), p_values,
                cd=None, reverse=True, width=9, textspace=1.5, labels=labels)

    font = {'family': 'sans-serif',
            'color': 'black',
            'weight': 'normal',
            'size': 22,
            }
    if title:
        plt.title(title, fontdict=font, y=0.9, x=0.5)
    plt.savefig(f'Results/cd-diagram_{"".join(CLASSIFIERS)}.png', bbox_inches='tight')


def significance_tests():
    res_per_sample_agg = pd.read_csv(f'Results/results_per_sample_agg_{"".join(CLASSIFIERS)}.csv')
    res_per_sample_agg['classifier_name'] = res_per_sample_agg['classifier_name'] + '_' + res_per_sample_agg['features'].astype(str) + '_' + res_per_sample_agg['feature_selection_method']
    draw_cd_diagram(df_perf=res_per_sample_agg, test_measure='mean')


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=True,
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
    return ax


def plot_confusion_matrix_sns(y_true, y_pred, classes,
                              partial=None,
                              normalize=True,
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

    plt.figure(figsize=(20, 20))

    # plot heat map
    # if partial:
    #     g = sns.heatmap(cm[:partial, :partial], annot=True, cmap="Blues", annot_kws={"fontsize": 24})
    # else:
    #     g = sns.heatmap(cm, annot=True, cmap="Blues")

    loc, labelsx = plt.xticks()
    loc, labelsy = plt.yticks()
    plt.title(title, fontdict={'fontsize': 20})

    # g.set_xticklabels(classes, rotation=90, fontsize=18)
    # g.set_yticklabels(classes, rotation=0, fontsize=18)

    # plt.show()

    # return g