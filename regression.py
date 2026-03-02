# Exemple complet de mise en place d’un modèle de prédiction multi-sorties
# avec gestion robuste des NaN capteurs

import pandas as pd
import numpy as np
from data_preparation import import_datasets, prepare_data, prepare_label, remove_nan_consumption
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# -------------------------------------------------------------------
# 1. Chargement des données
# -------------------------------------------------------------------
train_data, train_labels, test_data = import_datasets()

train_data = prepare_data(train_data)
train_labels = prepare_label(train_labels)
test_data = prepare_data(test_data)
train_data = remove_nan_consumption(train_data)
test_data = remove_nan_consumption(test_data)
X_cols = [
    "minutes_since_Epoch",
    "consumption",
    "visibility",
    "temperature",
    "humidity",
    "humidex",
    "windchill",
    "wind",
    "pressure",
    "dayofweek",
    "isweekend",
    "saison",
    "ispublicholiday",
    "isbusinesshour"
]

y_cols = [
    "washing_machine",
    "fridge_freezer",
    "TV",
    "kettle"
]

X = train_data[X_cols]
y = train_labels[y_cols]

# -------------------------------------------------------------------
# 2. MASQUE DE VALIDITÉ (Y sans NaN)
# -------------------------------------------------------------------

valid_y_mask = ~y.isna().any(axis=1)

X_valid = X.loc[valid_y_mask]
y_valid = y.loc[valid_y_mask]

# -------------------------------------------------------------------
# 3. Split temporel (sans shuffle)
# -------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_valid, y_valid, test_size=0.2, shuffle=False
)

# -------------------------------------------------------------------
# 4. Normalisation
# -------------------------------------------------------------------

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------------------------------------------
# 5. Modèle multi-sorties
# -------------------------------------------------------------------

base_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

model = MultiOutputRegressor(base_model)
model.fit(X_train_scaled, y_train)

# -------------------------------------------------------------------
# 6. Évaluation (uniquement sur données valides)
# -------------------------------------------------------------------

y_pred = model.predict(X_test_scaled)

for i, col in enumerate(y_cols):
    mae = mean_absolute_error(y_test[col], y_pred[:, i])
    print(f"MAE {col}: {mae:.3f}")

# -------------------------------------------------------------------
# 7. FONCTION DE PRÉDICTION AVEC GESTION DES NaN
# -------------------------------------------------------------------

def predict_with_nan_handling(model, scaler, X, y_cols):
    """
    Prédit uniquement sur les lignes sans NaN dans X.
    Retourne NaN sinon.
    """
    predictions = pd.DataFrame(
        index=X.index,
        columns=y_cols,
        dtype=float
    )

    valid_X_mask = ~X.isna().any(axis=1)

    if valid_X_mask.any():
        X_valid = scaler.transform(X.loc[valid_X_mask])
        y_pred = model.predict(X_valid)
        predictions.loc[valid_X_mask, y_cols] = y_pred

    return predictions

# -------------------------------------------------------------------
# 8. Prédictions sur le jeu de test (ou production)
# -------------------------------------------------------------------

test_predictions = predict_with_nan_handling(
    model,
    scaler,
    test_data[X_cols],
    y_cols
)

print(test_predictions.head())

# -------------------------------------------------------------------
# 9. Prédiction sur une nouvelle observation (safe)
# -------------------------------------------------------------------

new_sample = pd.DataFrame([{
    "minutes_since_Epoch": 22534080,
    "consumption": 586.9,
    "visibility": 35,
    "temperature": 8.9,
    "humidity": 86,
    "humidex": 8.9,
    "windchill": 6,
    "wind": 19,
    "pressure": 1017.3,
    "dayofweek": 6,
    "isweekend": 1,
    "saison": 1,
    "ispublicholiday": 0,
    "isbusinesshour": 0
}])

new_prediction = predict_with_nan_handling(
    model,
    scaler,
    new_sample,
    y_cols
)

print(new_prediction)
