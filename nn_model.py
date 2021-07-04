from datetime import date
from pathlib import Path

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model

from utils import evaluations

from utils.evaluations import *

from sklearn.metrics import accuracy_score
from classifiers.fcn import *
from classifiers.fcn_extension import *
from classifiers.inception import *
from classifiers.inception_extension import *
from classifiers.resnet import *
from classifiers.resnet_extension import *


models_dict = {'fcn': Classifier_FCN, 'resnet': Classifier_RESNET, 'inception': Classifier_INCEPTION,
               'fcn_extension': Classifier_FCN_TSFRESH, 'resnet_extension': Classifier_RESNET_TSFRESH,
               'inception_extension': Classifier_INCEPTION_TSFRESH}


class NNModelRun:
    """
    neural network model class. given a model name - build new model, fit the model / load model, predict and evaluate
    """
    def __init__(self, model_str, input_shape, nb_classes, output_directory, file_date='today', verbose=0,
                 input_agg=10, build=True, multilabel=False):
        model = models_dict.get(model_str)

        if file_date == 'today':
            file_date = date.today().strftime('%Y-%m-%d')

        Path(f'{output_directory}/{file_date}').mkdir(parents=True, exist_ok=True)

        self.output_path = f'{output_directory}/{file_date}'
        self.verbose = verbose
        self.classes = nb_classes
        self.multilabel = multilabel
        if model is None:
            print("incorrect model name")
            return

        if 'extension' in model_str:
            self.model_obj = model(self.output_path, input_shape, nb_classes, verbose, input_agg=input_agg, build=build)

        else:
            self.model_obj = model(self.output_path, input_shape, nb_classes, verbose, build=build)

    def fit_model(self, epochs, x_train, x_val, y_train, y_val, batch_size=16):

        cp = ModelCheckpoint(
            filepath=os.path.join(self.output_path, f"best_model.h5"),
            save_best_only=True, monitor='val_loss')

        es = EarlyStopping(monitor='val_loss', patience=100)

        rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, min_lr=0.00001)

        # self.model.summary()
        start_time = time.time()

        hist = self.model_obj.model.fit(x=x_train, y=y_train, batch_size=batch_size, shuffle=True, epochs=epochs,
                                        validation_data=(x_val, y_val), verbose=self.verbose,
                                        callbacks=[cp, rlrop, es])

        duration = time.time() - start_time

        self.model_obj.model.save(os.path.join(self.output_path + 'last_model.h5'))

        save_logs(self.output_path, hist, duration)

    def load_model(self, model_type='best'):
        if model_type == 'best':
            model = "best_model.h5"
        elif model_type == 'last':
            model = 'last_model.h5'
        else:  # full path
            model = model_type

        self.model_obj.model = load_model(os.path.join(self.output_path, model))

    def predict(self, x_test):
        self.load_model()
        y_pred = self.model_obj.model.predict(x_test)
        return y_pred

    def evaluate(self, x_test, y_test):
        start_time = time.time()
        y_pred = self.predict(x_test)
        duration = time.time() - start_time

        for idx in range(self.classes):
            y_true_i = y_test.values[:, idx]
            y_pred_i = y_pred[:, idx]
            tp = np.sum(np.round(np.clip(y_true_i * y_pred_i, 0, 1)))
            fp = np.sum(np.round(np.clip(y_pred_i - y_true_i, 0, 1)))

            # calculate precision
            p = tp / (tp + fp)
            print(f'accuracy of {idx} {p}')

        if self.multilabel:
            y_pred[y_pred >= 0.5] = 1
            y_pred[y_pred < 0.5] = 0
            print(f'precision_score mean micro {precision_score(y_test, y_pred, average="micro")}')
            print(f'precision_score mean macro {precision_score(y_test, y_pred, average="macro")}')
            print(f'recall_score mean micro:{recall_score(y_test, y_pred, average="micro")}')
            print(f'recall_score mean macro:{recall_score(y_test, y_pred, average="macro")}')

        else:
            y_pred = np.argmax(y_pred, axis=-1)
            y_test = np.argmax(y_test, axis=-1)
            accuracy = accuracy_score(y_test, y_pred)
            evaluations.plot_confusion_matrix_sns(y_test, y_pred, classes=np.unique(y_test),
                                                  partial=8, title='Baseline 1 confusion matrix')

            print("Accuracy: %.2f%%" % (accuracy * 100.0))

            df_metrics = calculate_metrics(y_test, y_pred, duration)
            return df_metrics

