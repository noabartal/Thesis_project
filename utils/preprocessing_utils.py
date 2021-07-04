import os
import re
import numpy as np
import pandas as pd
import pywt
from ec_feature_selection import ECFS
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder

SEC_SIZE = 10
import matplotlib.pyplot as plt
from scipy import signal, interpolate


# plot a histogram of each variable in the dataset
def plot_variable_distributions(trainX, column_list):
    """
    plot all sensors dist
    :param trainX: ndarray of values
    :param column_list: column name list
    """
    for column in range(trainX.shape[1]):
        plt.figure()
        plt.hist(trainX[:, column], bins=100, range=(np.nanmin(trainX[:, column]), np.nanmax(trainX[:, column])))
        plt.title(column_list[column])
        plt.show()


def compute_velocity(x, y, z):
    """
    usually calculating using wrist signal
    :param x: x time series signal
    :param y: y time series signal
    :param z: z time series signal
    :return: velocity time series vector with the same length as input
    """

    b, a = signal.butter(1, 20 / 120)
    del_t = 1 / 120

    x_filtered = signal.filtfilt(b, a, x)
    y_filtered = signal.filtfilt(b, a, y)
    z_filtered = signal.filtfilt(b, a, z)

    x_filtered = np.abs(np.gradient(x_filtered, del_t))
    y_filtered = np.abs(np.gradient(y_filtered, del_t))
    z_filtered = np.abs(np.gradient(z_filtered, del_t))

    velocity = np.sqrt(np.square(x_filtered) + np.square(y_filtered) + np.square(z_filtered))

    return velocity


def read_matlab_time_cut(time_series_folder, path_to_data, file_name="/matlab_features_full.xlsx"):
    """
    Just for not implementing th1 - th4. 
    :return: indices to cut each frame
    """

    # create unique row key based on subject id and king of experiment (full / empty cup, height of shelf)
    agg_matlab = pd.read_excel(f'{path_to_data}/{time_series_folder}/{file_name}')
    agg_matlab['height'] = agg_matlab['height'].replace({1: 'H', 2: 'M', 3: 'L'})
    agg_matlab['empty'] = agg_matlab['empty'].replace({1: 'E', 2: 'F'})
    agg_matlab['key'] = agg_matlab['Subjects'] + '_' + agg_matlab['height'] + agg_matlab["moveNum"].astype(
        str) + agg_matlab['empty']

    time_features = ['th1F', 'th4F']
    new_time_features = []

    for feature in time_features:
        agg_matlab[feature[:-1] + 'V'] = ((agg_matlab[feature] - 1) * (1.2)).astype(int)
        new_time_features += [feature[:-1] + 'V']
    return agg_matlab[['key'] + new_time_features]


def read_stroke_data(time_series_folder='Soroka/raw_data/Patients/splitted_data', event_column='taskName',
                     data_folder_path='',
                     with_controls=False, window_size=10):
    """
    create df of all the time series data and annotations
    :param window_size: size of fixed data size
    :param with_controls: default use patient data only (starting with p) else use control group (c)
    :param event_column: unique series index column
    :param data_folder_path: data folder path
    :param time_series_folder: folder with raw data
    :param annotations_path: file with summed trip data
    :return: list of df for each event
    """
    index_trip = 0
    subjects_folders = os.listdir(f'{data_folder_path}/{time_series_folder}')
    subjects = [folder for folder in subjects_folders if 'P' in folder]
    if with_controls:
        subjects += [folder for folder in subjects_folders if 'C' in folder]
    # filter time series between th1 to th4
    # start_end_time_map = read_matlab_time_cut(time_series_folder, path_to_data)

    dic = {}
    for subject in subjects:
        path_raw_data = f'{data_folder_path}/{time_series_folder}/{subject}'
        subject_files = [i for i in os.listdir(path_raw_data) if os.path.isfile(os.path.join(path_raw_data, i))]
        for file in subject_files:
            data = pd.read_csv(f'{path_raw_data}/{file}')
            data = data.apply(pd.to_numeric, errors='coerce')
            columns = data.columns[1:]

            data[event_column] = subject + '_' + ''.join([x[0] if not x.isdigit() else str(int(x))
                                                          for x in re.findall(f'{subject}_(.+).csv',
                                                                              file)[0].split('_')]).upper()

            # # filter time series between th1 to th4
            # start_end_by_key = start_end_time_map.loc[start_end_time_map['key'] == data.loc[0, event_column]]
            # data = data.iloc[start_end_by_key['th1V'].values[0]:start_end_by_key['th4V'].values[0]]

            # fix sequence size by interpolation
            data_inter = pd.DataFrame([])
            x = np.arange(0, data.shape[0])
            xnew = np.arange(0, data.shape[0] - 1, (data.shape[0] - 1) / window_size)
            data_inter['Time'] = xnew
            for column in columns:
                y = data[column]
                f = interpolate.interp1d(x, y, kind='cubic')
                data_inter[column] = f(xnew)  # use interpolation function returned by `interp1d`

            data_inter[event_column] = data.iloc[0][event_column]
            data_inter['index_trip'] = index_trip
            data_inter['PatientID'] = subject
            dic[index_trip] = data_inter
            index_trip += 1

    data_list = [values for values in dic.values()]
    return data_list, []


def read_shrp2_data(time_series_folder, annotations_path, trip_id='vtti.file_id', event_column='eventID',
                    path_to_data='', original_label="anonymousparticipantid", label='DriverID'):
    """
    create df of all the time series data and annotations
    :param path_to_data: data folder path
    :param event_column: trip event number (each trip contains more then one event)
    :param trip_id: trip id
    :param time_series_folder: folder with raw data
    :param annotations_path: file with summed trip data
    :param original_label: name of original target column
    :param label: name of wanted target column

    :return: list of df for each event
    """
    data_list = []
    index_trip = 0
    for folder in os.walk(f'{path_to_data}/{time_series_folder}'):
        for file in folder[2]:
            data = pd.read_csv(f'{folder[0]}/{file}')
            data = data.apply(pd.to_numeric, errors='coerce')
            data[event_column] = re.findall('Index_(.+).csv', file)[0]
            data['index_trip'] = index_trip
            data_list.append(data)
            index_trip += 1

    trip_summary = pd.read_csv(f'{path_to_data}/{annotations_path}', index_col=0)

    for i in range(len(data_list)):
        data_list[i][label] = trip_summary.loc[data_list[i][trip_id][0], original_label]

    return data_list, [trip_id]


def split_by_sc(df, time_column, sec):
    """
    add section column to the data to determine the section
    :param df: dataframe without section
    :param time_column: column with time stamp
    :param sec: split by this value
    :return: dataframe with section column
    """
    ms = sec * 1000
    starting_point = df.loc[0, time_column]
    df['section'] = df[time_column].apply(lambda x: int((x - starting_point) / ms))
    return df


def data_cleaining(df, trip_id='vtti.file_id'):
    """
    clean data unnecessary row / columns
    :param df: dataframe containing one train sample (time series)
    :param trip_id: column that must be "full", if there are nan values, the records should be dropped

    :return: dataframe df after cleaning
    """

    # remove nan values from trip_id (corrupted records)
    if trip_id:
        df.dropna(inplace=True, subset=[trip_id])

    # remove irrelevant features
    drop_list = list(df.filter(regex='^TRACK'))
    df.drop(drop_list, inplace=True, axis=1)

    # drop std zero columns (irrelevant )
    # drop_list = df.std()[df.std() == 0].index[1:]
    # drop_list = list(set(columns) & set(drop_list))
    # df.drop(columns=drop_list, inplace=True)

    return df


def missing_values(df, linear_filling, padding_filling):
    """
    fill missing values, in the record level (not between different time series records)
    :param df: dataframe of one recorde (time series)
    :param linear_filling: columns to fill linearily (continues)
    :param padding_filling: columns to fill with padding (discrete)
    :return: dataframe with filled missing values.
    """
    linear_filling = [filling for filling in linear_filling if filling in df.columns]
    # filling continues ( straight line between two point)
    df[linear_filling] = df[linear_filling].interpolate(method='linear', limit_direction='both', axis=0)

    padding_filling = [filling for filling in padding_filling if filling in df.columns]
    # fill with same number as before
    df[padding_filling] = df[padding_filling].fillna(method='pad')

    return df


def fill_nan_records(df, continues_columns, discrete_columns):
    """
    fill records where all the values are nan based on other records.
    :param df: dataframe of all data
    :param continues_columns: columns that should be filled and has continues values
    :param discrete_columns: columns that should be filled and has discrete values
    :return: dataframe with filled missing values
    """
    replace_cont = list(set(continues_columns) & set(df.columns))
    replace_disc = list(set(discrete_columns) & set(df.columns))

    if len(replace_disc) > 0:
        df[replace_disc] = df[replace_disc].apply(lambda x: x.fillna(x.value_counts().index[0]))
    if len(replace_cont) > 0:
        df[replace_cont] = df[replace_cont].apply(lambda x: x.fillna(x.mean()))  # fill nan with mean val

    return df


def rm_outliers(df, column_list):
    """
    Remove outliers from the data based on iqr statistical method
    :param df: dataframe with outliers
    :param column_list: column_list to find outliers in
    :return: dataframe without outliers
    """
    for column in column_list:
        q3 = np.nanpercentile(df.loc[:, column], 75)
        q1 = np.nanpercentile(df.loc[:, column], 25)

        max = q3 + (q3 - q1) * 1.5
        min = q1 - (q3 - q1) * 1.5

        df.loc[df.loc[:, column] > max, column] = max
        df.loc[df.loc[:, column] < min, column] = min
    return df


def stat_wavelet(df, columns):
    """
    smooth the data by stationary wavelet transformation, might be instead of removing outliers
    :return: transformed df
    """
    for idx, index in enumerate(df['index_trip'].unique()):
        for column in columns:

            if df.loc[df['index_trip'] == index].shape[0] % 2 == 1:
                drop_index = df.loc[df['index_trip'] == index].index[-1]
                df.drop(drop_index, axis=0, inplace=True)

            df.loc[df['index_trip'] == index, column] = pywt.swt(data=df.loc[df['index_trip'] == index, column],
                                                                 wavelet='haar')[0][0]

    return df


def disc_wavelet(df, columns):
    """
    smooth the data by discrete wavelet transformation, might be instead of removing outliers
    :return: transformed df
    """
    for index in df['index_trip'].unique():
        for column in columns:
            df.loc[df['index_trip'] == index, column] = pywt.swt(data=df.loc[df['index_trip'] == index, column],
                                                                 wavelet='haar')[0][0]
    return df


def preprocess(continues_columns, discrete_columns, label, event='eventID', handle_nan=False,
               model='SHRP2', path_to_data='Data', time_series_folder='Time-series', trip_id='vtti.file_id',
               annotations_path='Clean_Data/TripSummary_timeseries.csv',
               window_size=320, with_controls=False, time_tsfresh=[]):
    file_name = 'df_after_reading_all_trips'
    if handle_nan:
        file_name = 'df_after_reading_all_trips_handle_nan'

    if not os.path.exists(f'{path_to_data}/{model}/{file_name}.csv'):
        final_df = pd.DataFrame()
        if model == 'SOROKA':
            x_train, extra_columns = read_stroke_data(data_folder_path=path_to_data,
                                                      time_series_folder=time_series_folder,
                                                      event_column=event, window_size=window_size,
                                                      with_controls=with_controls)
        elif model == 'SHRP2':
            x_train, extra_columns = read_shrp2_data(path_to_data=path_to_data,
                                                     time_series_folder=time_series_folder, event_column=event,
                                                     trip_id=trip_id, label=label,
                                                     annotations_path=annotations_path)
        else:
            return

        for i in range(len(x_train)):
            x_train[i] = data_cleaining(x_train[i], columns=continues_columns + discrete_columns)
            # x_train[i] = split_by_sc(x_train[i], SEC_SIZE)
            if handle_nan:
                x_train[i] = missing_values(x_train[i], continues_columns, discrete_columns)
            final_df = final_df.append(x_train[i], ignore_index=True)

        final_df.to_csv(f'{path_to_data}/{model}/{file_name}.csv')

    final_df = pd.read_csv(f'{path_to_data}/{model}/{file_name}.csv', index_col=0)
    final_df = final_df[continues_columns + discrete_columns + [event, label, 'index_trip'] + time_tsfresh]

    column_list = continues_columns + discrete_columns

    if handle_nan:

        # final_df = rm_outliers(final_df, padding_filling)
        # final_df = stat_wavelet(final_df, linear_filling)
        final_df.loc[:, column_list] = fill_nan_records(final_df.loc[:, column_list], continues_columns,
                                                        discrete_columns)
        final_df.loc[:, column_list] = final_df.loc[:, column_list].fillna(-1)

    final_df.dropna(axis=1, how='all', inplace=True)
    return final_df[column_list + [label, event, 'index_trip'] + time_tsfresh], final_df[label].unique()


def window_seq(labeled_data_path, results_path, seq_length, step):

    data = pd.read_csv(labeled_data_path)

    print("original data shape {}".format(data.shape))

    # creating trips
    data_as_sequences = pd.DataFrame()
    dic = {}

    seq_next_idx = 0
    for idx in data['index_trip'].unique():
        group_df = data.loc[data['index_trip'] == idx, :]

        for i in range(0, group_df.shape[0] - seq_length + 1, step):
            seq_to_add = group_df.iloc[i:seq_length + i]
            seq_to_add.insert(0, 'series_num', seq_next_idx)
            # data_as_sequences = data_as_sequences.append(seq_to_add)
            dic[seq_next_idx] = seq_to_add

            seq_next_idx += 1
    list_of_values = [values for values in dic.values()]
    data_as_sequences = data_as_sequences.append(list_of_values)
    data_as_sequences = data_as_sequences.reset_index(drop=True)
    data_as_sequences.drop('index_trip', axis=1, inplace=True)
    data_as_sequences.to_csv(results_path, index=False)


def feature_selection_ecfs(x_train, y_train, input_path, features, name):

    if not os.path.exists(input_path + f'ecfs_filtered_train_{name}.csv'):
        percent = int(x_train.shape[1] * (features / 100.0))
        print(percent)
        ecfs = ECFS(n_features=percent)
        ecfs.fit(X=x_train, y=y_train, alpha=0.5, positive_class=1, negative_class=0)
        summary = pd.DataFrame({'Feature': x_train.columns, 'Ranking': ecfs.ranking, 'MI': ecfs.mutual_information,
                                'Fisher Score': ecfs.fisher_score})

        summary = summary.sort_values(by='Ranking')

        summary.to_csv(input_path + f'ecfs_summary.csv')
        x_train_filtered = x_train.loc[:, summary.iloc[:percent]['Feature']]
        print(x_train_filtered.shape)

        x_train_filtered.to_csv(input_path + f'ecfs_filtered_train_{name}.csv')

    else:
        x_train_filtered = pd.read_csv(input_path + f'ecfs_filtered_train_{name}.csv', index_col=0)

    return x_train_filtered.columns


def feature_selection_ufs(x_train, y_train, input_path, features, name):
    percent = int(x_train.shape[1] * (features / 100.0))

    if not os.path.exists(input_path + f'k_best{name}.csv'):
        x_train = x_train.dropna(axis='columns')
        x_train = x_train.loc[:, (x_train != x_train.iloc[0]).any()]
        X_best = SelectKBest(f_classif, k=percent).fit(x_train, y_train)
        mask = X_best.get_support()  # list of booleans for selected features
        new_feat = []
        for bool, feature in zip(mask, x_train.columns):
            if bool:
                new_feat.append(feature)
        print(new_feat)
        k_best = pd.DataFrame(new_feat, columns=['column'])
        k_best.to_csv(input_path + f'k_best{name}.csv')

    else:
        k_best = pd.read_csv(input_path + f'k_best{name}.csv', index_col=0)

    x_train_filtered = x_train[k_best['column']]

    return x_train_filtered.columns


def feature_selection_cfs(x_train, y_train, input_path, features, name):
    th = 0
    percent = int(x_train.shape[1] * (features / 100.0))

    if not os.path.exists(input_path + f'correlation_dict{name}.csv'):

        x_train = x_train.dropna(axis='columns')
        x_train = x_train.loc[:, (x_train != x_train.iloc[0]).any()]
        print(x_train.shape)

        correlation_dict, remove_history = fcbf(x_train, pd.Series(y_train), threshold=th, base=2, is_debug=False)
        # x_train.iloc[:, idx], x_test.iloc[:, idx]
        correlation_dict = pd.DataFrame(correlation_dict)
        correlation_dict.to_csv(input_path + f'correlation_dict{name}.csv')

    correlation_dict = pd.read_csv(input_path + f'correlation_dict{name}.csv', index_col=0)

    correlation_dict = correlation_dict.iloc[:percent, :]
    x_train_filtered = x_train.loc[:, correlation_dict['0']]

    return x_train_filtered.columns


def feature_selection_weka(x_train, y_train, input_path, features, name):
    percent = int(x_train.shape[1] * (features / 100.0))

    if not os.path.exists(input_path + f'selected_features_weka_{features}_{name}.csv'):
        import weka.core.jvm as jvm
        jvm.start(system_cp=True, packages=True, system_info=True)

        x_train = x_train.loc[:, (x_train != x_train.iloc[0]).any()]
        sava_data = x_train.copy()
        sava_data.columns = range(sava_data.shape[1])

        label_encoder = LabelEncoder()
        label_encoder.fit(np.unique(y_train))
        sava_data['target'] = label_encoder.transform(y_train)

        sava_data.to_csv('WEKA/train_weka_format.csv', index=False)

        from weka.attribute_selection import ASEvaluation, AttributeSelection,ASSearch
        from weka.core.converters import Loader
        from weka.filters import Filter
        loader = Loader(classname="weka.core.converters.CSVLoader")
        data = loader.load_file('WEKA/train_weka_format.csv', class_index='last')
        remove = Filter(classname="weka.filters.unsupervised.attribute.RemoveType", options=["-T", "string"])

        if features == 'all':
            search = ASSearch(classname="weka.attributeSelection.BestFirst", options=["-D", "1", "-N", "5"])
            evaluator = ASEvaluation(classname="weka.attributeSelection.CfsSubsetEval", options=["-P", "1", "-E", "1", "-M"])
            attsel = AttributeSelection()
            attsel.search(search)
            attsel.evaluator(evaluator)
            remove.inputformat(data)
            filtered = remove.filter(data)
            attsel.select_attributes(filtered)
            set_of_features = attsel.selected_attributes[:-1]
            print(f"selected attributes set size {len(set_of_features)}")
            # print("# BestFirst: " + str(attsel.number_attributes_selected))
            # print("result string:\n" + attsel.results_string)
        elif features == 'small':
            # search = ASSearch(classname="weka.attributeSelection.BestFirst", options=["-D", "1", "-N", "5"])
            # evaluator = ASEvaluation(classname="weka.attributeSelection.CfsSubsetEval",
            #                          options=["-P", "1", "-E", "1", "-M"])
            # attsel = AttributeSelection()
            # attsel.search(search)
            # attsel.evaluator(evaluator)
            # remove.inputformat(data)
            # filtered = remove.filter(data)
            # attsel.select_attributes(filtered)
            # set_of_features = attsel.selected_attributes[:-1]
            #
            # if len(set_of_features) > 16:
            search = ASSearch(classname="weka.attributeSelection.GreedyStepwise", options=["-C", "-R", "-N", "16"])
            evaluator = ASEvaluation(classname="weka.attributeSelection.CfsSubsetEval",
                                     options=["-P", "1", "-E", "1", "-L"])
            attsel = AttributeSelection()
            attsel.search(search)
            attsel.evaluator(evaluator)
            remove.inputformat(data)
            filtered = remove.filter(data)
            attsel.select_attributes(filtered)
            ranked_attributes = pd.DataFrame(attsel.ranked_attributes, columns=['Feature', 'Rank'])
            ranked_attributes['Feature'] = ranked_attributes['Feature'].astype(int)
            set_of_features = ranked_attributes.loc[:15, 'Feature']
        else:
            search = ASSearch(classname="weka.attributeSelection.GreedyStepwise", options=["-C", "-R", "-N", f"{percent}"])
            evaluator = ASEvaluation(classname="weka.attributeSelection.CfsSubsetEval",
                                     options=["-P", "1", "-E", "1", "-L"])
            attsel = AttributeSelection()
            attsel.search(search)
            attsel.evaluator(evaluator)
            attsel.select_attributes(data)
            ranked_attributes = pd.DataFrame(attsel.ranked_attributes, columns=['Feature', 'Rank'])
            ranked_attributes['Feature'] = ranked_attributes['Feature'].astype(int)
            set_of_features = ranked_attributes.loc[:percent - 1, 'Feature']

        x_train.iloc[:, set_of_features].to_csv(input_path + f'selected_features_weka_{features}_{name}.csv')
        selected_features = x_train.iloc[:, set_of_features].columns
        jvm.stop()
    else:
        selected_features = pd.read_csv(input_path + f'selected_features_weka_{features}_{name}.csv', index_col=0).columns

    x_train_filtered = x_train.loc[:, selected_features]

    return x_train_filtered.columns


def feature_selection(x_train, y_train, input_path, features, method, description):
    if method == 'cfs':
        return feature_selection_cfs(x_train, y_train, input_path, features, f'_{features}_{description}')
    elif method == 'ufs':
        return feature_selection_ufs(x_train, y_train, input_path, features, f'_{features}_{description}')
    elif method == 'weka':
        return feature_selection_weka(x_train, y_train, input_path, features, f'_{features}_{description}')
    elif method == 'ecfs':
        return feature_selection_ecfs(x_train, y_train, input_path, features, f'_{features}_{description}')
    else:
        print('method not exist')
        return []