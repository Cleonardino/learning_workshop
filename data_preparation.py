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

def import_datasets():
    train_data = pd.read_csv(DATA_PATH + "X_train.csv")
    train_labels = pd.read_csv(DATA_PATH + "Y_train.csv")
    test_data = pd.read_csv(DATA_PATH + "X_test.csv")
    return train_data, train_labels, test_data

# "2013-04-18T00:01:00.0"
def prepare_data(dataset):
    dataset = dataset.drop(columns="Unnamed: 9")
    dataset["minutes_since_Epoch"] = dataset["time_step"].apply(time_step_to_minutes)
    for column in columns:
        dataset[column] = dataset[column].interpolate()
        dataset[column] = dataset[column].bfill()
        dataset[column] = dataset[column].ffill()
    dataset = add_features(dataset)

def prepare_label(dataset_labels):
    dataset_labels["minutes_since_Epoch"] = dataset_labels["time_step"].apply(time_step_to_minutes)