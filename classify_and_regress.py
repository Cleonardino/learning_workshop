import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from data_preparation import import_datasets, prepare_data, prepare_label

# --------------------------------------------------
# Load & prepare data
# --------------------------------------------------

X_train, Y_train, X_test = import_datasets()

prepare_data(X_train)
prepare_data(X_test)
prepare_label(Y_train)

# Align indexes
X_train = X_train.sort_values("minutes_since_Epoch").reset_index(drop=True)
Y_train = Y_train.sort_values("minutes_since_Epoch").reset_index(drop=True)
X_test = X_test.sort_values("minutes_since_Epoch").reset_index(drop=True)

# --------------------------------------------------
# Identify device columns
# --------------------------------------------------

device_columns = [
    col for col in Y_train.columns
    if col not in ["time_step", "minutes_since_Epoch"]
]

# OFF consumption rule
OFF_VALUES = {device: 0 for device in device_columns}
for device in device_columns:
    if "tv" in device.lower():
        OFF_VALUES[device] = 7

# --------------------------------------------------
# Feature set
# --------------------------------------------------

feature_cols = [
    col for col in X_train.columns
    if col not in ["time_step"]
]

# --------------------------------------------------
# Train models
# --------------------------------------------------

classifiers = {}
regressors = {}

for device in device_columns:

    off_value = OFF_VALUES[device]

    # Binary target: ON / OFF
    y_binary = (Y_train[device] > off_value).astype(int)

    # -------- Classifier --------
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train[feature_cols], y_binary)
    classifiers[device] = clf

    # -------- Regressor (ON only) --------
    on_mask = y_binary == 1

    reg = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LinearRegression())
    ])

    reg.fit(
        X_train.loc[on_mask, feature_cols],
        Y_train.loc[on_mask, device]
    )

    regressors[device] = reg

# --------------------------------------------------
# Prediction on X_test
# --------------------------------------------------

predictions = pd.DataFrame(
    index=X_test.index,
    columns=device_columns
)

for device in device_columns:

    off_value = OFF_VALUES[device]

    # Predict ON / OFF
    on_pred = classifiers[device].predict(X_test[feature_cols])

    # Default OFF value
    predictions[device] = off_value

    # Predict consumption where ON
    on_indices = np.where(on_pred == 1)[0]

    if len(on_indices) > 0:
        predictions.loc[on_indices, device] = regressors[device].predict(
            X_test.loc[on_indices, feature_cols]
        )

# --------------------------------------------------
# Output
# --------------------------------------------------

predictions["time_step"] = X_test["time_step"]
predictions = predictions[["time_step"] + device_columns]

predictions.to_csv("Y_test_pred.csv", index=False)

print("Prediction completed → Y_test_pred.csv")
