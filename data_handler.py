import tensorflow as tf

from datetime import datetime
from utils import preprocessing_utils as pre

import pandas as pd
from utils.preprocessing_utils import window_seq
import numpy as np
import os
import random
from tsfresh import extract_features, select_features
from tsfresh.feature_extraction import EfficientFCParameters
from pathlib import Path


class PathsConfig:
    def __init__(self, folder_path, dataset, sequence_length, step_size):
        subfolder_path = os.path.join(folder_path, dataset)
        Path(subfolder_path).mkdir(parents=True, exist_ok=True)

        self.folder_and_file_path = f'{subfolder_path}/seq_{sequence_length}_step_{step_size}'

        self.preprocessing_path = self.create_prepcosessing_path()
        self.window_path = self.create_sliding_window_path()
        self.agg_data_path = self.create_agg_data_path()

    def create_prepcosessing_path(self):
        return f'{self.folder_and_file_path}_preprocessing.csv'

    def create_sliding_window_path(self):
        return f'{self.folder_and_file_path}_sliding_window.csv'

    def create_agg_data_path(self):
        return f'{self.folder_and_file_path}_agg_data.csv'


class DataHandler:

    def __init__(self, continues_features, discrete_features, label, event='eventID', path_to_data='Data',
                 data_kind='SHRP2', time_column='vtti.timestamp', preprocess=False, sliding_window=False, agg_features=False,
                 seq_length=7, frequency_in_ms=10, step_size=1, controls='', handle_nan=True):

        self.seq_length = seq_length
        self.seq_size = int(seq_length / frequency_in_ms)
        self.step_size = step_size
        self.record = event

        self.paths_config = PathsConfig(path_to_data, data_kind, seq_length, step_size)

        self.column_list = continues_features + discrete_features
        self.label = label

        self.raw_data_preprocessing(preprocess, sliding_window, continues_features, discrete_features,
                                    handle_nan, data_kind, controls, time_column)

        self.agg_data_preprocessing(agg_features, time_column)

        self.agg_df = pd.read_csv(self.paths_config.agg_data_path)

        self.deep_df = pd.read_csv(self.paths_config.window_path)

    def raw_data_preprocessing(self, preprocess, sliding_window, continues_features, discrete_features, handle_nan,
                               data_kind, controls, time_column):

        if preprocess:
            df_dl, _ = pre.preprocess(continues_columns=continues_features,
                                      discrete_columns=discrete_features, label=self.label, handle_nan=handle_nan,
                                      model=data_kind, event=self.record, window_size=self.seq_length,
                                      with_controls=(controls!=''), time_tsfresh=[time_column])
            df_dl.to_csv(self.paths_config.preprocessing_path, index=False)

        if sliding_window:
            window_seq(labeled_data_path=self.paths_config.preprocessing_path, results_path=self.paths_config.window_path,
                       seq_length=self.seq_length, step=self.step_size)

    def agg_data_preprocessing(self, agg_features, time_column):
        if agg_features:
            df_stat = pd.read_csv(self.paths_config.window_path)
            df_stat.loc[:, self.column_list] = df_stat.loc[:, self.column_list].fillna(-1)

            agg_df = self.tsfresh_features(df_stat, time_column)

            agg_df.to_csv(self.paths_config.agg_data_path)

    def tsfresh_features(self, data, time_column, fc_param=EfficientFCParameters):

        features_tsfresh = extract_features(
            data.drop([self.label, self.record], axis=1),
            column_id='series_num', column_sort=time_column, default_fc_parameters=fc_param(), n_jobs=0)

        y = data[self.label].values.reshape((int(
            data.shape[0] / self.seq_length), self.seq_length))[:, 0]
        #
        features_tsfresh_filtered = select_features(features_tsfresh.dropna(axis=1, how='any'), y,
                        ml_task='classification', n_jobs=0)

        features_tsfresh = features_tsfresh_filtered
        features_tsfresh[self.record] = data[self.record].values.reshape((int(
            data.shape[0] / self.seq_length), self.seq_length))[:, 0]
        features_tsfresh[self.label] = data[self.label].values.reshape((int(
            data.shape[0] / self.seq_length), self.seq_length))[:, 0]

        features_tsfresh.index.rename('series_num', inplace=True)
        return features_tsfresh





