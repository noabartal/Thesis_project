from sklearn.metrics import roc_auc_score, accuracy_score
from xgboost import XGBClassifier

from data_handler import *
from utils.config import dataset, label, event, time_column
from utils.config import seq_length, step_size
from utils.config import continues_features, discrete_features, path_to_data
from utils.model_utils import train_test_split_agg

if __name__ == '__main__':

    data = DataHandler(continues_features, discrete_features, label, event=event, path_to_data=path_to_data,
                       preprocess=False, sliding_window=False, agg_features=False, time_column=time_column,
                       step_size=step_size, seq_length=seq_length, data_kind=dataset, use_tsfresh=True, test=None)

    x_train, x_val, x_test, y_train, y_val, y_test, _ = train_test_split_agg(data.agg_df, event, label)

    eval_set = [(x_val, y_val)]

    model = XGBClassifier()
    model.fit(x_train, y_train, eval_set=eval_set)

    y_pred = model.predict(x_test)

    # evaluate predictions
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    # evaluate predictions
    roc = roc_auc_score(y_test, y_pred)
