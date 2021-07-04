# Deep Learning for Time Series Classification with aggregated features
This repository contains the models and experiments for the paper titled "Deep learning for time series classification with aggregated features" which is currently under review.
This paper is part of my MSc thesis.
![architecture](https://github.com/noabartal/Thesis_project/tree/master/images/diagram.png)

## Data 
The code uses two unique datasets:
* The [SHRP2 dataset](https://insight.shrp2nds.us./), which contains multivariate vehicle sensors data that used for driver identification task.
* The SOROKA after stroke patients dataset, which contains multivariate sensors data with patients after stroke and control group. The task is compensation detection, which is a multilabel task.

## Code 
* The [main.py](https://github.com/noabartal/Thesis_project/tree/master/main.py) python file contains the necessary code to run an experiement.
* The [nn_model.py](https://github.com/noabartal/Thesis_project/tree/master/nn_model.py) python file contains NNModelRun class that is responsible on training and evaluating the neural network models.
* The [data_handler.py](https://github.com/noabartal/Thesis_project/tree/master/data_handler.py) python file contains DataHandler class that is responsible on all the data preparation phases (raw data preprocessing, sliding window, aggregated feature creation).
* The [tsfresh_xgboost_baseline.py](https://github.com/noabartal/Thesis_project/tree/master/tsfresh_xgboost_baseline.py) python file contains generic strong baseline for our method.
* The [utils](https://github.com/noabartal/Thesis_project/tree/master/utils) folder contains the [configuration file](https://github.com/noabartal/Thesis_project/tree/master/utils/config.py) necessary functions to read the datasets, perform preprocessing, run models, evaluate the models and visualize them.
* The [CFSmethod](https://github.com/noabartal/Thesis_project/tree/master/CFSmethod) contains the Correlation based Feature Selection method based on [ZixiaoShen git repository](https://github.com/ZixiaoShen/Correlation-based-Feature-Selection/tree/master/CFSmethod) with minor changes for non-binary task.
* The [classifiers](https://github.com/noabartal/Thesis_project/tree/master/classifiers) folder contains six python files, three for the deep neural network architectures without our extension 
and three for the deep neural network architectures with our extension.
* The [SOROKA_code](https://github.com/noabartal/Thesis_project/tree/master/SOROKA_code) folder contains unique code for SOROKA dataset (label creation)
* The [SHRP2_code](https://github.com/noabartal/Thesis_project/tree/master/SHRP2_code) folder contains unique code for SHRP2 dataset (driver identification baseline implementation)


## Acknowledgements
TBA



