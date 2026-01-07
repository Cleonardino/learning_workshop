import pandas as pd
import datetime as dt
DATA_PATH = "./data/"
from jours_feries_france import JoursFeries

def get_day_of_week(time_step):
    return pd.Timestamp(time_step).dayofweek

def is_weekend(time_step):
    return pd.Timestamp(time_step).dayofweek >= 5

def get_season(time_step):
    month = pd.Timestamp(time_step).month
    day = pd.Timestamp(time_step).day
    if (month < 3) or (month == 3 and day < 20):
        return 0 # 0 pour l'hiver
    elif (month < 6) or (month == 6 and day < 20):
        return 1 # 1 pour le printemps
    elif (month < 9) or (month == 9 and day < 22):
        return 2 # 2 pour l'été
    elif (month < 12) or (month == 3 and day < 20):
        return 3 # 3 pour l'automne
    else:
        return 0

def is_public_holiday(time_step, public_holidays):
    return dt.date(pd.Timestamp(time_step).year, pd.Timestamp(time_step).month, pd.Timestamp(time_step).day) in public_holidays[str(pd.Timestamp(time_step).year)]

def is_buisness_hour(time_step, public_holidays):
    if is_weekend(time_step) or is_public_holiday(time_step, public_holidays):
        return False
    return pd.Timestamp(time_step).hour >= 8 and pd.Timestamp(time_step).hour <= 18

def data_preparation(dataset):
    years = dataset["time_step"].apply(lambda time_step : pd.Timestamp(time_step).year).unique()
    public_holidays = {}
    for year in years:
        public_holidays[str(year)] = list(JoursFeries.for_year(int(year)).values())
    dataset["dayofweek"] = dataset["time_step"].apply(get_day_of_week)
    dataset["isweekend"] = dataset["time_step"].apply(is_weekend)
    dataset["saison"] = dataset["time_step"].apply(get_season)
    dataset["ispublicholiday"] = dataset["time_step"].apply((lambda x : is_public_holiday(x, public_holidays)))
    dataset["isbuisnesshour"] = dataset["time_step"].apply((lambda x : is_buisness_hour(x, public_holidays)))
    return dataset