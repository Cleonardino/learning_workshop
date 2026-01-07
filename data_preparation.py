import pandas as pd
import numpy as np
DATA_PATH = "./data/"
from feature_extraction import add_features
columns = ["visibility",  "temperature",  "humidity",  "humidex",  "windchill",  "wind",  "pressure"]


def time_step_to_minutes(time_step):
    return int(pd.Timestamp(time_step).timestamp()/60)

def nb_consecutive_nan(dataset : pd.DataFrame, column : str):
    max = 0
    count = 0
    for i in range(dataset.shape[0]):
        if np.isnan(dataset[column][i]):
            count += 1
        else:
            if count > max:
                max = count
            count = 0
    if count > max:
        max = count
    return max

# "2013-04-18T00:01:00.0"
def data_preparation(dataset_train, dataset_labels):
    dataset_train = dataset_train.drop(columns="Unnamed: 9")
    dataset_train["minutes_since_Epoch"] = dataset_train["time_step"].apply(time_step_to_minutes)
    dataset_labels["minutes_since_Epoch"] = dataset_labels["time_step"].apply(time_step_to_minutes)
    for column in columns:
        dataset_train[column] = dataset_train[column].interpolate()
        dataset_train[column] = dataset_train[column].bfill()
    dataset_train = add_features(dataset_train)
