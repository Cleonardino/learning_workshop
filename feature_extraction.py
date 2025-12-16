import pandas as pd
DATA_PATH = "./data/"

train_data = pd.read_csv(DATA_PATH + "X_train.csv")
train_data = train_data.drop(columns="Unnamed: 9")

def get_day_of_week(time_step):
    return pd.Timestamp(time_step).dayofweek

def is_weekend(time_step):
    return pd.Timestamp(time_step).dayofweek >= 5

train_data["dayofweek"] = train_data["time_step"].apply(get_day_of_week)
train_data["isweekend"] = train_data["time_step"].apply(is_weekend)
print(train_data["dayofweek"].unique())
print(train_data["isweekend"].unique())
print(train_data)