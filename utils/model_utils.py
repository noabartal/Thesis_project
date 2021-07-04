import numpy as np
import pandas as pd
import matplotlib

from utils.preprocessing_utils import feature_selection

matplotlib.use('agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Arial'
import os

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


def scale_train_test(self, df_train, df_test):
    for column in df_test.columns:
        # scale
        mean_s = np.mean(df_test.loc[:, column])
        sd_s = np.std(df_test.loc[:, column])
        df_test.loc[:, column] = (df_test.loc[:, column] - mean_s) / (sd_s + 1 * np.exp(-6))
        df_train.loc[:, column] = (df_train.loc[:, column] - mean_s) / (sd_s + 1 * np.exp(-6))


def scale_raw_data(x_train, x_val, x_test):

    # znorm
    std_ = np.nanstd(x_train, axis=1, keepdims=True)
    std_[std_ == 0] = 1.0
    x_train = (x_train - np.nanmean(x_train, axis=1, keepdims=True)) / std_

    std_ = np.nanstd(x_val, axis=1, keepdims=True)
    std_[std_ == 0] = 1.0
    x_val = (x_val - np.nanmean(x_val, axis=1, keepdims=True)) / std_

    std_ = np.nanstd(x_test, axis=1, keepdims=True)
    std_[std_ == 0] = 1.0
    x_test = (x_test - np.nanmean(x_test, axis=1, keepdims=True)) / std_

    x_train = np.nan_to_num(x_train, nan=-1)
    x_val = np.nan_to_num(x_val, nan=-1)
    x_test = np.nan_to_num(x_test, nan=-1)

    return x_train, x_val, x_test


def scale_agg_data(x_train_agg, x_val_agg, x_test_agg):

    scaler = MinMaxScaler(feature_range=(0, 1))
    x_train_agg.loc[:, x_train_agg.columns] = scaler.fit_transform(
        x_train_agg.values)
    x_val_agg.loc[:, x_train_agg.columns] = scaler.transform(
        x_val_agg.values)
    x_test_agg.loc[:, x_train_agg.columns] = scaler.transform(
        x_test_agg.values)
    return x_train_agg, x_val_agg, x_test_agg


def train_test_split_agg(df, event_column='eventID', label='DriverID'):
    x_train_idx, x_val_idx, x_test_idx = no_leakage_train_test_split_idx(
        df, event_column, label, test_percent=0.15, val_percent=0.15)

    x_train = df[df[event_column].isin(x_train_idx)]
    x_val = df[df[event_column].isin(x_val_idx)]
    x_test = df[df[event_column].isin(x_test_idx)]
    y_train = x_train[label]
    y_val = x_val[label]
    y_test = x_test[label]

    return x_train.drop(['series_num', event_column, label], axis=1), \
           x_val.drop(['series_num', event_column, label], axis=1), \
           x_test.drop(['series_num', event_column, label], axis=1), \
           y_train, y_val, y_test, x_test[[event_column, label]]


def create_target(y_train, y_val, y_test):

    binary_encoder = OneHotEncoder(sparse=False)
    y_train = binary_encoder.fit_transform(y_train.reshape(-1, 1))
    y_val = binary_encoder.transform(y_val.reshape(-1, 1))  # .reshape(-1, 1)
    y_test = binary_encoder.transform(y_test.reshape(-1, 1))
    classes = binary_encoder.categories_[0]
    return y_train, y_val, y_test, classes


def create_multi_target(y_train, y_val, y_test, data_path):
    targets = pd.read_csv(f'{data_path}/code_products/patients_labels.csv', index_col=0)
    targets_test = pd.read_csv(f'{data_path}/code_products/patients_labels_test.csv', index_col=0)

    classes = targets.columns[-6:]

    y_train = pd.DataFrame(y_train).merge(targets[classes],
                                                    how='left', right_index=True,
                                                    left_on="taskID").drop("taskID", axis=1).fillna(0)
    y_val = pd.DataFrame(y_val).merge(targets[classes],
                                                how='left', right_index=True,
                                                left_on="taskID").drop("taskID", axis=1).fillna(0)

    y_test = pd.DataFrame(y_test).merge(targets_test[classes],
                                                  how='inner', right_index=True,
                                                  left_on="taskID").drop("taskID", axis=1).fillna(0)

    return y_train, y_val, y_test


def LOO_Split(raw_df, agg_df, record, label, seq_length, column_list, idx=-1, raw_df_test=None, agg_df_test=None):
    """
    split test and train by label (ie different patient at test)
    """

    x_record_id = raw_df[[label]].drop_duplicates().values.flatten()
    x_train_idx, x_test_idx = np.delete(x_record_id, idx), x_record_id[[idx]]

    x_train = raw_df[raw_df[label].isin(x_train_idx)]
    x_train_agg = agg_df[agg_df[label].isin(x_train_idx)].reset_index(drop=True)

    x_val = raw_df[raw_df[label].isin(x_test_idx)]
    x_val_agg = agg_df[agg_df[label].isin(x_test_idx)].reset_index(drop=True)

    if raw_df_test:
        x_test = raw_df_test
        x_test_agg = agg_df_test.reset_index(drop=True)
    else:
        x_test = raw_df[raw_df[label].isin(x_test_idx)]
        x_test_agg = agg_df[agg_df[label].isin(x_test_idx)].reset_index(drop=True)

    x_train = x_train[column_list].values.reshape((int(x_train.shape[0] / seq_length),
                                                             seq_length, len(column_list)))

    x_val = x_val[column_list].values.reshape((int(x_val.shape[0] / seq_length), seq_length,
                                                        len(column_list)))

    x_test = x_test[column_list].values.reshape((int(x_test.shape[0] / seq_length), seq_length,
                                                          len(column_list)))

    y_train = x_train_agg[record]
    y_val = x_val_agg[record]
    y_test = x_test_agg[record]

    x_train_agg.drop(['series_num', record, label], axis=1, inplace=True)
    x_val_agg.drop(['series_num', record, label], axis=1, inplace=True)
    x_test_agg.drop(['series_num', record, label], axis=1, inplace=True)

    return x_train, x_train_agg, x_val, x_val_agg, x_test, x_test_agg, y_train, y_val, y_test


def no_leakage_train_test_split_idx(full_df, record, label, test_percent=0.15, val_percent=0.15):

    x_record_id = full_df[[record, label]].drop_duplicates()
    x, x_test_idx, y, y_test_idx = train_test_split(
        x_record_id[record], x_record_id[label], stratify=x_record_id[label], test_size=test_percent,
        shuffle=True)

    x_train_idx, x_val_idx, _, _ = train_test_split(x, y, stratify=y,
                                                                      test_size=val_percent / (1 - test_percent),
                                                                      shuffle=True, random_state=42)
    return x_train_idx, x_val_idx, x_test_idx


def split_raw_data(raw_df, x_train_idx, x_val_idx, x_test_idx, record, column_list, seq_length):
    x_train = raw_df[raw_df[record].isin(x_train_idx)]
    x_val = raw_df[raw_df[record].isin(x_val_idx)]
    x_test = raw_df[raw_df[record].isin(x_test_idx)]

    x_train = x_train[column_list].values.reshape((int(x_train.shape[0] / seq_length), seq_length,
                                                            len(column_list)))

    x_val = x_val[column_list].values.reshape((int(x_val.shape[0] / seq_length), seq_length,
                                                        len(column_list)))

    x_test = x_test[column_list].values.reshape((int(x_test.shape[0] / seq_length), seq_length,
                                                          len(column_list)))

    return x_train, x_val, x_test


def split_agg_data(agg_df, record, label, x_train_idx, x_val_idx, x_test_idx):

    x_train_agg = agg_df[agg_df[record].isin(x_train_idx)]

    x_val_agg = agg_df[agg_df[record].isin(x_val_idx)]

    x_test_agg = agg_df[agg_df[record].isin(x_test_idx)]

    y_train = x_train_agg[label].values
    y_val = x_val_agg[label].values
    y_test = x_test_agg[label].values

    return x_train_agg.drop(['series_num', record, label], axis=1), x_val_agg.drop(['series_num', record, label], axis=1), \
           x_test_agg.drop(['series_num', record, label], axis=1), y_train, y_val, y_test


def split_and_scale_data(raw_df, agg_df, record, label, seq_length, column_list, data_path, multilabel=False,
                         idx=0, raw_df_test=None,
                         agg_df_test=None, return_agg=True):

    if multilabel:
        x_train, x_train_agg, x_val, x_val_agg, x_test, x_test_agg, \
        y_train, y_val, y_test = LOO_Split(raw_df, agg_df, record, label,
                                           seq_length, column_list, idx, raw_df_test=raw_df_test,
                  agg_df_test=agg_df_test)
        y_train, y_val, y_test = create_multi_target(y_train, y_val, y_test, data_path)

    else:
        x_train_idx, x_val_idx, x_test_idx = no_leakage_train_test_split_idx(raw_df, record, label)

        x_train, x_val, x_test = split_raw_data(raw_df, x_train_idx, x_val_idx, x_test_idx, record, column_list, seq_length)
        x_train_agg, x_val_agg, x_test_agg, y_train, y_val, y_test = split_agg_data(
            agg_df, record, label, x_train_idx, x_val_idx, x_test_idx)

        y_train, y_val, y_test, _ = create_target(y_train, y_val, y_test)

    x_train, x_val, x_test = scale_raw_data(x_train, x_val, x_test)
    x_train_agg, x_val_agg, x_test_agg = scale_agg_data(x_train_agg, x_val_agg, x_test_agg)

    if not return_agg:
        return x_train, x_val, x_test, y_train, y_val, y_test
    return [x_train, x_train_agg], [x_val, x_val_agg], [x_test, x_test_agg], y_train, y_val, y_test


def select_agg_features(x_train_agg, x_val_agg, x_test_agg, y_train, method, percent_of_features, model_description, data_path):
    columns_selected = feature_selection(x_train_agg,
                                         y_train, f'{data_path}/',
                                         description=model_description, features=percent_of_features,
                                         method=method)

    x_train_agg = x_train_agg[columns_selected]
    x_val_agg = x_val_agg[columns_selected]
    x_test_agg = x_test_agg[columns_selected]

    return x_train_agg, x_val_agg, x_test_agg


def calculate_metrics(y_true, y_pred, duration, y_true_val=None, y_pred_val=None):
    res = pd.DataFrame(data=np.zeros((1, 4), dtype=np.float), index=[0],
                       columns=['precision', 'accuracy', 'recall', 'duration'])

    res['precision'] = precision_score(y_true, y_pred, average='macro')
    res['accuracy'] = accuracy_score(y_true, y_pred)

    res['recall'] = recall_score(y_true, y_pred, average='macro')
    res['duration'] = duration
    return res


def save_test_duration(file_name, test_duration):
    res = pd.DataFrame(data=np.zeros((1, 1), dtype=np.float), index=[0],
                       columns=['test_duration'])
    res['test_duration'] = test_duration
    res.to_csv(file_name, index=False)


def plot_epochs_metric(hist, file_name, metric='loss'):
    plt.figure()
    plt.plot(hist.history[metric])
    plt.plot(hist.history['val_' + metric])
    plt.title('model ' + metric)
    plt.ylabel(metric, fontsize='large')
    plt.xlabel('epoch', fontsize='large')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()


def save_logs(output_directory, hist, duration, lr=True, y_true_val=None, y_pred_val=None):
    hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv(os.path.join(output_directory, 'history.csv'), index=False)

    index_best_model = hist_df['loss'].idxmin()
    row_best_model = hist_df.loc[index_best_model]

    df_best_model = pd.DataFrame(data=np.zeros((1, 6), dtype=np.float), index=[0],
                                 columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_train_acc',
                                          'best_model_val_acc', 'best_model_learning_rate', 'best_model_nb_epoch'])

    df_best_model['best_model_train_loss'] = row_best_model['loss']
    df_best_model['best_model_val_loss'] = row_best_model['val_loss']
    df_best_model['best_model_train_acc'] = row_best_model['accuracy']
    df_best_model['best_model_val_acc'] = row_best_model['val_accuracy']
    if lr == True:
        df_best_model['best_model_learning_rate'] = row_best_model['lr']
    df_best_model['best_model_nb_epoch'] = index_best_model

    df_best_model.to_csv(os.path.join(output_directory,'df_best_model.csv'), index=False)

    # plot losses
    plot_epochs_metric(hist, output_directory + 'epochs_loss.png')




