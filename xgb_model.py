
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    print("OK XGBoost disponible")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("ERREUR XGBoost non disponible")

def evaluate_model(y_true, y_pred, dataset_name):
    devices = ['washing_machine', 'fridge_freezer', 'TV', 'kettle']
    print(f"\nXGBoost - {dataset_name}:")
    print("-" * 60)
    total_mae = 0
    total_rmse = 0
    total_r2 = 0
    results = {}
    for i, device in enumerate(devices):
        mae = mean_absolute_error(y_true[device], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_true[device], y_pred[:, i]))
        r2 = r2_score(y_true[device], y_pred[:, i])
        results[device] = {'mae': mae, 'rmse': rmse, 'r2': r2}
        total_mae += mae
        total_rmse += rmse
        total_r2 += r2
        print(f"{device:15} - MAE: {mae:6.2f} W | RMSE: {rmse:6.2f} W | R2: {r2:6.3f}")
    avg_mae = total_mae / len(devices)
    avg_rmse = total_rmse / len(devices)
    avg_r2 = total_r2 / len(devices)
    results['average'] = {'mae': avg_mae, 'rmse': avg_rmse, 'r2': avg_r2}
    print(f"{'MOYENNE':15} - MAE: {avg_mae:6.2f} W | RMSE: {avg_rmse:6.2f} W | R2: {avg_r2:6.3f}")
    return results

def main():
    print("="*80)
    print("XGBOOST - TOUTES LES DONNEES + FEATURES TEMPORELLES")
    print("="*80)
    
    if not XGBOOST_AVAILABLE:
        print("\nERREUR XGBoost n'est pas installe")
        return None
    
    print("\n[1] Chargement des donnees brutes...")
    train_data_raw = pd.read_csv("./data/X_train.csv")
    train_data_raw = train_data_raw.drop(columns="Unnamed: 9")
    train_labels_raw = pd.read_csv("./data/Y_train.csv")
    print(f"   OK {len(train_data_raw):,} lignes chargees")
    
    print("\n[2] Extraction des features temporelles (optimise)...")
    
    # Convertir time_step en datetime (vectorise)
    train_data_raw["time_step_dt"] = pd.to_datetime(train_data_raw["time_step"])
    train_labels_raw["time_step_dt"] = pd.to_datetime(train_labels_raw["time_step"])
    
    # Extraction vectorisee des features temporelles
    train_data_raw["dayofweek"] = train_data_raw["time_step_dt"].dt.dayofweek
    train_data_raw["isweekend"] = train_data_raw["dayofweek"] >= 5
    train_data_raw["hour"] = train_data_raw["time_step_dt"].dt.hour
    train_data_raw["month"] = train_data_raw["time_step_dt"].dt.month
    train_data_raw["day"] = train_data_raw["time_step_dt"].dt.day
    
    # Saison (vectorise)
    def get_season_vectorized(month, day):
        season = np.zeros(len(month), dtype=int)
        season[(month < 3) | ((month == 3) & (day < 20))] = 0  # hiver
        season[((month > 3) | ((month == 3) & (day >= 20))) & ((month < 6) | ((month == 6) & (day < 20)))] = 1  # printemps
        season[((month > 6) | ((month == 6) & (day >= 20))) & ((month < 9) | ((month == 9) & (day < 22)))] = 2  # ete
        season[((month > 9) | ((month == 9) & (day >= 22))) & ((month < 12) | ((month == 12) & (day < 20)))] = 3  # automne
        return season
    
    train_data_raw["saison"] = get_season_vectorized(train_data_raw["month"].values, train_data_raw["day"].values)
    
    # Jours feries francais (vectorise)
    from jours_feries_france import JoursFeries
    years = train_data_raw["time_step_dt"].dt.year.unique()
    all_holidays = set()
    for year in years:
        holidays = JoursFeries.for_year(int(year)).values()
        all_holidays.update(holidays)
    
    train_data_raw["date_only"] = train_data_raw["time_step_dt"].dt.date
    train_data_raw["ispublicholiday"] = train_data_raw["date_only"].isin(all_holidays)
    
    # Heures de travail (vectorise)
    train_data_raw["isbuisnesshour"] = (
        ~train_data_raw["isweekend"] & 
        ~train_data_raw["ispublicholiday"] & 
        (train_data_raw["hour"] > 7) & 
        (train_data_raw["hour"] < 17)
    )
    
    # Convertir time_step en minutes pour la fusion (vectorise)
    train_data_raw["time_step_minutes"] = (train_data_raw["time_step_dt"].astype('int64') // 10**9 // 60).astype(int)
    train_labels_raw["time_step_minutes"] = (train_labels_raw["time_step_dt"].astype('int64') // 10**9 // 60).astype(int)
    
    train_data_with_features = train_data_raw
    train_labels_with_features = train_labels_raw
    
    # Nettoyage des colonnes météo
    weather_columns = ["visibility", "temperature", "humidity", "humidex", "windchill", "wind", "pressure", "consumption"]
    for column in weather_columns:
        if column in train_data_with_features.columns:
            train_data_with_features[column] = train_data_with_features[column].interpolate().fillna(0)
    
    # Nettoyage des labels
    train_labels_with_features[['washing_machine','fridge_freezer','TV','kettle']] = \
        train_labels_with_features[['washing_machine','fridge_freezer','TV','kettle']].interpolate().fillna(0)
    
    # Fusion sur time_step_minutes
    df = pd.merge(train_data_with_features, train_labels_with_features, on="time_step_minutes", how="left", suffixes=('', '_y'))
    df = df.drop(columns=[col for col in df.columns if col.endswith('_y')])
    
    print("   OK Features temporelles creees:")
    print("      - dayofweek (jour de la semaine)")
    print("      - isweekend (weekend)")
    print("      - saison (saison)")
    print("      - ispublicholiday (jours feries)")
    print("      - isbuisnesshour (heures de travail)")
    
    print("\n[3] Selection des features...")
    features = [
        "consumption", "visibility", "temperature", "humidity",
        "humidex", "windchill", "wind", "pressure",
        "dayofweek", "isweekend", "saison", "ispublicholiday", "isbuisnesshour"
    ]
    target_columns = ['washing_machine', 'fridge_freezer', 'TV', 'kettle']
    
    available_features = [f for f in features if f in df.columns]
    print(f"   OK {len(available_features)} features selectionnees:")
    
    # Séparer par type
    weather_features = [f for f in available_features if f in ['consumption', 'visibility', 'temperature', 'humidity', 'humidex', 'windchill', 'wind', 'pressure']]
    temporal_features = [f for f in available_features if f in ['dayofweek', 'isweekend', 'saison', 'ispublicholiday', 'isbuisnesshour']]
    
    print(f"      * Features meteo/consommation: {len(weather_features)}")
    for f in weather_features:
        print(f"        - {f}")
    print(f"      * Features temporelles: {len(temporal_features)}")
    for f in temporal_features:
        print(f"        - {f}")
    
    print(f"\n[4] Preparation X et Y...")
    X = df[available_features].fillna(0)
    Y = df[target_columns].fillna(0)
    print(f"   OK X: {X.shape}, Y: {Y.shape}")
    
    print("\n[5] Split train/test (80/20)...")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True)
    print(f"   Entrainement: {len(X_train):,} echantillons ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   Test: {len(X_test):,} echantillons ({len(X_test)/len(X)*100:.1f}%)")
    
    print("\n[6] Configuration XGBoost...")
    xgb_params = {
        'objective': 'reg:squarederror',
        'n_estimators': 5000,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0,
        'early_stopping_rounds': 20
    }
    print("   OK Parametres:")
    print(f"      * Arbres: {xgb_params['n_estimators']}")
    print(f"      * Profondeur max: {xgb_params['max_depth']}")
    print(f"      * Learning rate: {xgb_params['learning_rate']}")
    
    print(f"\n[7] Entrainement XGBoost...")
    print("   ATTENTION: Entrainement d'un modele par appareil (4 modeles)")
    print("   ATTENTION: Cela peut prendre plusieurs minutes...")
    models = {}
    train_pred = np.zeros((len(X_train), 4))
    test_pred = np.zeros((len(X_test), 4))
    feature_importances = []
    devices = ['washing_machine', 'fridge_freezer', 'TV', 'kettle']
    
    for i, device in enumerate(devices):
        print(f"\n   Modele {device} ({i+1}/4)...")
        model = xgb.XGBRegressor(**xgb_params)
        model.fit(X_train, Y_train[device], eval_set=[(X_test, Y_test[device])], verbose=False)
        models[device] = model
        train_pred[:, i] = model.predict(X_train)
        test_pred[:, i] = model.predict(X_test)
        feature_importances.append(model.feature_importances_)
        print(f"   OK Modele {device} entraine (best iteration: {model.best_iteration})")
    
    print("\n   OK Tous les modeles entraines avec succes")
    
    print("\n[8] Evaluation des performances...")
    train_results = evaluate_model(Y_train, train_pred, "ENTRAINEMENT")
    test_results = evaluate_model(Y_test, test_pred, "TEST")
    
    print(f"\n[9] IMPORTANCE DES FEATURES:")
    print("-" * 60)
    avg_importance = np.mean(feature_importances, axis=0)
    importance_df = pd.DataFrame({
        'feature': available_features,
        'importance': avg_importance
    }).sort_values('importance', ascending=False)
    
    print("Top 15 features les plus importantes:")
    for i, (_, row) in enumerate(importance_df.head(15).iterrows()):
        print(f"{i+1:2d}. {row['feature']:25}: {row['importance']:.4f}")
    
    # Importance des nouvelles features temporelles
    if temporal_features:
        temporal_importance = importance_df[importance_df['feature'].isin(temporal_features)]
        if not temporal_importance.empty:
            print(f"\nIMPORTANCE DES NOUVELLES FEATURES TEMPORELLES:")
            for _, row in temporal_importance.iterrows():
                rank = list(importance_df['feature']).index(row['feature']) + 1
                print(f"   {rank:2d}. {row['feature']:20}: {row['importance']:.4f}")
    
    print(f"\n[10] ANALYSE DE L'OVERFITTING:")
    print("-" * 40)
    train_mae = train_results['average']['mae']
    test_mae = test_results['average']['mae']
    train_r2 = train_results['average']['r2']
    test_r2 = test_results['average']['r2']
    overfitting_mae = test_mae - train_mae
    
    print(f"MAE - Train: {train_mae:.2f}W, Test: {test_mae:.2f}W")
    print(f"Difference: {overfitting_mae:+.2f}W")
    print(f"R2 - Train: {train_r2:.3f}, Test: {test_r2:.3f}")
    
    if overfitting_mae < 2:
        print("OK Bon equilibre")
    elif overfitting_mae < 5:
        print("ATTENTION Leger overfitting")
    else:
        print("ERREUR Overfitting important")
    
    print(f"\n{'='*25} RESUME XGBOOST {'='*25}")
    print(f"Donnees: {len(df):,} lignes (TOUTES)")
    print(f"Features: {len(available_features)}")
    print(f"   * Features meteo/consommation: {len(weather_features)}")
    print(f"   * Features temporelles: {len(temporal_features)}")
    print(f"Modeles: 4 (un par appareil)")
    print(f"Entrainement: {len(X_train):,} echantillons")
    print(f"Test: {len(X_test):,} echantillons")
    print(f"Performance Test:")
    print(f"   * MAE: {test_mae:.2f}W")
    print(f"   * RMSE: {test_results['average']['rmse']:.2f}W")
    print(f"   * R2: {test_r2:.3f}")
    print(f"\nXGBoost termine avec succes!")
    
    return {
        'models': models,
        'train_results': train_results,
        'test_results': test_results,
        'feature_importance': importance_df,
        'temporal_features': temporal_features
    }

if __name__ == "__main__":
    results = main()
