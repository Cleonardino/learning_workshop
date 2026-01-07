from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from sklearn.model_selection import train_test_split
train_labels[['washing_machine','fridge_freezer','TV','kettle']] = \
    train_labels[['washing_machine','fridge_freezer','TV','kettle']].interpolate().fillna(0)

data_preparation(train_data)
df = pd.merge(train_data, train_labels, on="time_step", how="left")


print(df.columns)


features = [
    "consumption",       
    "visibility",
    "temperature",
    "humidity",
    "humidex",
    "windchill",
    "wind",
    "pressure",
    "isweekend",
    "saison",
    "isbuisnesshour"
]



X = df[features]


Y = df[["washing_machine", "fridge_freezer", "TV", "kettle"]]


X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, Y_train)

Y_pred = model.predict(X_val)
#TESTPUSH
devices = ['washing_machine', 'fridge_freezer', 'TV', 'kettle']

for device in devices:
    mae = mean_absolute_error(Y_val[device], Y_pred[:, devices.index(device)])
    rmse = np.sqrt(mean_squared_error(Y_val[device], Y_pred[:, devices.index(device)]))
    print(f"{device} - MAE: {mae:.2f} W, RMSE: {rmse:.2f} W")
