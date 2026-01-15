import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from data_preparation import import_datasets, prepare_data, prepare_label

x_train, y_train, x_test = import_datasets()

x_train = prepare_data(x_train)
x_test = prepare_data(x_test)
y_train = prepare_label(y_train)

# Align indexes
x_train = x_train.sort_values("minutes_since_Epoch").reset_index(drop=True)
y_train = y_train.sort_values("minutes_since_Epoch").reset_index(drop=True)
x_test = x_test.sort_values("minutes_since_Epoch").reset_index(drop=True)

# --------------------------------------------------
# Train/Validation split (80/20)
# --------------------------------------------------

split_idx = int(len(x_train) * 0.8)

x_tr = x_train.iloc[:split_idx].reset_index(drop=True)
y_tr = y_train.iloc[:split_idx].reset_index(drop=True)

x_val = x_train.iloc[split_idx:].reset_index(drop=True)
y_val = y_train.iloc[split_idx:].reset_index(drop=True)

print(f"Train size: {len(x_tr)}, Validation size: {len(x_val)}")

# --------------------------------------------------
# Identify device columns
# --------------------------------------------------

device_columns = [
    col for col in y_train.columns
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
    col for col in x_train.columns
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
    y_binary = (y_tr[device] > off_value).astype(int)

    # -------- Classifier --------
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=12,
        n_jobs=-1
    )
    clf.fit(x_tr[feature_cols], y_binary)
    classifiers[device] = clf

    # -------- Regressor (ON only) --------
    on_mask = y_binary == 1

    reg = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LinearRegression())
    ])

    reg.fit(
        x_tr.loc[on_mask, feature_cols],
        y_tr.loc[on_mask, device]
    )

    regressors[device] = reg

# --------------------------------------------------
# Validation evaluation
# --------------------------------------------------

print("\n" + "="*60)
print("VALIDATION PERFORMANCE")
print("="*60)

val_predictions = pd.DataFrame(
    index=x_val.index,
    columns=device_columns
)

for device in device_columns:

    off_value = OFF_VALUES[device]

    # Ground truth binary
    y_val_binary = (y_val[device] > off_value).astype(int)

    # Predict ON / OFF
    on_pred = classifiers[device].predict(x_val[feature_cols])

    # Classification metrics
    acc = accuracy_score(y_val_binary, on_pred)
    f1 = f1_score(y_val_binary, on_pred, zero_division=0)

    # Default OFF value
    val_predictions[device] = off_value

    # Predict consumption where ON
    on_indices = np.where(on_pred == 1)[0]

    if len(on_indices) > 0:
        val_predictions.loc[on_indices, device] = regressors[device].predict(
            x_val.loc[on_indices, feature_cols]
        )

    # Regression metrics (only on actual ON samples)
    actual_on_mask = y_val_binary == 1
    if actual_on_mask.sum() > 0:
        mae = mean_absolute_error(
            y_val.loc[actual_on_mask, device],
            val_predictions.loc[actual_on_mask, device]
        )
        rmse = np.sqrt(mean_squared_error(
            y_val.loc[actual_on_mask, device],
            val_predictions.loc[actual_on_mask, device]
        ))
    else:
        mae = rmse = np.nan

    print(f"\n{device}:")
    print(f"  Classification - Accuracy: {acc:.4f}, F1: {f1:.4f}")
    print(f"  Regression (ON only) - MAE: {mae:.2f}, RMSE: {rmse:.2f}")

# Overall MAE across all devices and time steps
overall_mae = mean_absolute_error(
    y_val[device_columns].values.flatten(),
    val_predictions[device_columns].values.flatten()
)
print(f"\nOverall Validation MAE: {overall_mae:.2f}")
print("="*60 + "\n")

# --------------------------------------------------
# Prediction on x_test
# --------------------------------------------------

predictions = pd.DataFrame(
    index=x_test.index,
    columns=device_columns
)

for device in device_columns:

    off_value = OFF_VALUES[device]

    # Predict ON / OFF
    on_pred = classifiers[device].predict(x_test[feature_cols])

    # Default OFF value
    predictions[device] = off_value

    # Predict consumption where ON
    on_indices = np.where(on_pred == 1)[0]

    if len(on_indices) > 0:
        predictions.loc[on_indices, device] = regressors[device].predict(
            x_test.loc[on_indices, feature_cols]
        )

# --------------------------------------------------
# Output
# --------------------------------------------------

predictions["time_step"] = x_test["time_step"]
predictions = predictions[["time_step"] + device_columns]

predictions.to_csv("Y_test_pred.csv", index=False)

print("Prediction completed → Y_test_pred.csv")