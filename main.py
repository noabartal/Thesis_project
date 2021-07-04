from data_handler import *
from utils.config import CLASSIFIERS, path_to_results
from utils.config import FEATURES
from utils.config import SELECTION, ITERS, DENSE
from utils.evaluations import evaluate, create_results_file, significance_tests
from utils.config import dataset, label, event, time_column, running_porpose
from utils.config import seq_length, step_size, ephocs
from utils.config import continues_features, discrete_features, path_to_data
from nn_model import NNModelRun
from utils.model_utils import split_and_scale_data, select_agg_features


def create_result_folder_name(folder_path, classifier_name, iter, dense='', fs_method=''):
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(folder_path, classifier_name)).mkdir(parents=True, exist_ok=True)

    sub_folder_name = f'iter_{iter}_dense_{dense}_fs_{fs_method}'
    full_folder_path = os.path.join(folder_path, classifier_name, sub_folder_name)
    Path(full_folder_path).mkdir(parents=True, exist_ok=True)

    return full_folder_path


def extended_run(classifier_name):

    for iter in range(ITERS):
        for i, dense in enumerate(DENSE):
            for feature in FEATURES:
                for i, fs_method in enumerate(SELECTION):
                    results_folder_path = create_result_folder_name(
                        path_to_results, classifier_name, iter, dense, fs_method)

                    x_train, x_val, x_test, y_train, y_val, y_test = split_and_scale_data(data_handler.deep_df,
                                                                                          data_handler.agg_df, event, label,
                                                                                          seq_length, discrete_features +
                                                                                          continues_features, path_to_data)

                    x_train[1], x_val[1], x_test[1] = select_agg_features(x_train[1], x_val[1], x_test[1],
                                                                          y_train, fs_method, feature,
                                                                          f'_window_{seq_length}_{step_size}',
                                                                          path_to_data)

                    nn_model = NNModelRun(classifier_name, x_train[0].shape[1:], len(data_handler.agg_df[label].unique()),
                                       output_directory=results_folder_path, file_date='today', verbose=0,
                                       input_agg=x_train[1].shape[1], build=True)

                    nn_model.fit_model(ephocs, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)
                    df_metrics = nn_model.evaluate(x_test, y_test)


def slim_run(classifier_name):
    for iter in range(0, ITERS):
        results_folder_path = create_result_folder_name(
            path_to_results, classifier_name, iter)

        x_train, x_val, x_test, y_train, y_val, y_test = split_and_scale_data(data_handler.deep_df,
                                                                              data_handler.agg_df, event,
                                                                              label,
                                                                              seq_length,
                                                                              discrete_features +
                                                                              continues_features,
                                                                              path_to_data,
                                                                              return_agg=False)

        model = NNModelRun(classifier_name, x_train.shape[1:], len(data_handler.agg_df[label].unique()),
                           output_directory=results_folder_path, file_date='today', verbose=0,
                           input_agg=x_train[1].shape[1], build=True)

        model.fit_model(ephocs, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)
        df_metrics = model.evaluate(x_test, y_test)


if __name__ == '__main__':
    data_handler = DataHandler(continues_features, discrete_features, label, event=event, path_to_data=path_to_data,
                               preprocess=False,
                               sliding_window=True, agg_features=True, time_column=time_column, step_size=step_size,
                               seq_length=seq_length, data_kind=dataset)

    if running_porpose == "train_model":
        for classifier in CLASSIFIERS:
            if 'extension' in classifier:  # run over several configurations
                extended_run(classifier)
            else:
                slim_run(classifier)

    elif running_porpose == "generate_results":
        create_results_file(path_to_results, data_handler.tf, data_handler)

    else:
        significance_tests()



