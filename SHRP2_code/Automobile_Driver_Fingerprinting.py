from utils import evaluations, preprocessing_utils as pre
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import os
from utils.config import *
from utils.model_utils import train_test_split_agg, scale_agg_data
from utils.evaluations import evaluate


class AutomobileDriverFingerprint:
    def __init__(self, preprocess_path, continues_columns, discrete_columns, label, event, step_size, seq_length, run_all=False):

        self.label = label
        self.event = event
        self.seq_length = seq_length
        self.column_list = continues_columns + discrete_columns
        if not os.path.exists(f'{path_to_data}/{preprocess_path}.csv') or run_all:
            final_df, _ = pre.preprocess(continues_columns=continues_columns, discrete_columns=discrete_columns,
                                         label=label, handle_nan=True)
            final_df.to_csv(f'{path_to_data}/{preprocess_path}.csv', index=False)

        path_for_sliding_win = f'{preprocess_path}_step_{step_size}_seq_{seq_length}'

        if not os.path.exists(f'{path_to_data}/{path_for_sliding_win}.csv') or run_all:
            pre.window_seq(f'{path_to_data}/{preprocess_path}', f'{path_to_data}/{path_for_sliding_win}', seq_length,
                       step_size)

        self.raw_df = pd.read_csv(f'{path_to_data}/{path_for_sliding_win}.csv')
        self.agg_df = pd.DataFrame()

    def create_statistical_features(self):
        for col in self.column_list:
            self.agg_df[col + '_min'] = self.raw_df.groupby(['series_num'])[col].min()
            self.agg_df[col + '_max'] = self.raw_df.groupby(['series_num'])[col].max()
            self.agg_df[col + '_mean'] = self.raw_df.groupby(['series_num'])[col].mean()
            self.agg_df[col + '_q1'] = self.raw_df.groupby(['series_num'])[col].quantile(.25)
            self.agg_df[col + '_q3'] = self.raw_df.groupby(['series_num'])[col].quantile(.75)
            self.agg_df[col + '_std'] = self.raw_df.groupby(['series_num'])[col].std()
            self.agg_df[col + '_autocorr'] = self.raw_df.groupby(['series_num'])[col].agg(pd.Series.autocorr)
            self.agg_df[col + '_skewness'] = self.raw_df.groupby(['series_num'])[col].skew()
            self.agg_df[col + '_kurtosis'] = self.raw_df.groupby(['series_num'])[col].agg(pd.Series.kurtosis)

        self.agg_df[self.event] = self.raw_df[self.event].values.reshape((int(self.raw_df.shape[0] / self.seq_length),
                                                                          self.seq_length))[:, 0]
        self.agg_df[self.label] = self.raw_df[self.label].values.reshape((int(self.raw_df.shape[0] / self.seq_length),
                                                                          self.seq_length))[:, 0]

    def create_descriptive_features(self):

        for col in self.column_list:
            paa = 10  # number of outputs in the output vector
            res = []
            for name, series_df in self.raw_df.groupby(['series_num']):
                if series_df.shape[0] % paa == 0:  # always true
                    section_arr = np.array_split(series_df[col].values, paa)
                    res.append([item.mean() for item in section_arr])

            res = np.array(res)
            for i in range(paa):
                self.agg_df[f'{col}_paa_{i}'] = res[:, i]


if __name__ == '__main__':
    preprocess_file_name = f'time_series_df_CANBUS_{seq_length}_automobile'

    baseline_model = AutomobileDriverFingerprint(preprocess_file_name, continues_features, discrete_features, label, event,
                                                 step_size, seq_length)

    baseline_model.create_statistical_features()
    baseline_model.create_descriptive_features()

    x_train, x_val, x_test, y_train, y_val, y_test, event_test = train_test_split_agg(baseline_model.agg_df, event, label)

    x_train, x_val, x_test = scale_agg_data(x_train, x_val, x_test)

    # Instantiate model with 1000 decision trees
    rf = RandomForestClassifier(n_estimators=1000, random_state=42)

    x_train = pd.concat([x_train, x_val], axis=0, ignore_index=True)
    y_train = pd.concat([y_train, y_val], axis=0, ignore_index=True)
    # Train the model on training data
    rf.fit(x_train, y_train)

    # Predict and evaluate
    y_pred = evaluate(rf, x_test, y_test, event_test, event, label, path_to_data)

    evaluations.plot_confusion_matrix_sns(y_test, y_pred, classes=np.unique(y_train),
                                          partial=8, title='Baseline 1 confusion matrix')

