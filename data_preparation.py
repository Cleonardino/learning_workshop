import pandas as pd
import numpy as np
DATA_PATH = "./data/"
from feature_extraction import data_preparation

def time_step_to_minutes(time_step):
    return int(pd.Timestamp(time_step).timestamp()/60)

def nb_consecutive_nan(dataset : pd.DataFrame, column : str):
    max = 0
    count = 0
    for i in range(train_data.shape[0]):
        if np.isnan(train_data[column][i]):
            count += 1
        else:
            if count > max:
                max = count
            count = 0
    if count > max:
        max = count
    return max

train_data = pd.read_csv(DATA_PATH + "X_train.csv")
train_data = train_data.drop(columns="Unnamed: 9")
train_labels = pd.read_csv(DATA_PATH + "Y_train.csv")

# "2013-04-18T00:01:00.0"
train_data = data_preparation(train_data)
train_data["time_step"] = train_data["time_step"].apply(time_step_to_minutes)
train_labels["time_step"] = train_labels["time_step"].apply(time_step_to_minutes)
columns = ["visibility",  "temperature",  "humidity",  "humidex",  "windchill",  "wind",  "pressure"]
for column in columns:
    train_data[column] = train_data[column].interpolate()
    train_data[column] = train_data[column].bfill()
