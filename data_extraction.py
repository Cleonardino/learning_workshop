import pandas as pd
DATA_PATH = "./data/"

def time_step_to_minutes(time_step):
    return int(pd.Timestamp(time_step).timestamp()/60)


train_data = pd.read_csv(DATA_PATH + "X_train.csv")
train_data = train_data.drop(columns="Unnamed: 9")

# "2013-04-18T00:01:00.0"
train_data["time_step"] = train_data["time_step"].apply(time_step_to_minutes)
columns = ["visibility",  "temperature",  "humidity",  "humidex",  "windchill",  "wind",  "pressure"]
for column in columns:
    train_data[column] = train_data[column].interpolate()
    train_data[column] = train_data[column].bfill()
import matplotlib.pyplot as plt

df["consumption"].plot(figsize=(14, 5))
plt.title("Consommation dans le temps")
plt.ylabel("kW / MW")
plt.show()