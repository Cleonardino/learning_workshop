import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, classification_report
from data_preparation import import_datasets, prepare_data, prepare_label
from classify_and_regress import ClassifyAndRegressModel


def train_val_split(x_train, y_train, val_ratio=0.2):
    """
    Split training data into train and validation sets chronologically.
    
    Parameters:
    -----------
    x_train : pd.DataFrame
        Features
    y_train : pd.DataFrame
        Labels
    val_ratio : float
        Proportion of data to use for validation
        
    Returns:
    --------
    tuple
        (x_tr, y_tr, x_val, y_val)
    """
    split_idx = int(len(x_train) * (1 - val_ratio))
    
    x_tr = x_train.iloc[:split_idx].reset_index(drop=True)
    y_tr = y_train.iloc[:split_idx].reset_index(drop=True)
    
    x_val = x_train.iloc[split_idx:].reset_index(drop=True)
    y_val = y_train.iloc[split_idx:].reset_index(drop=True)
    
    return x_tr, y_tr, x_val, y_val


def evaluate_model(model, x_val, y_val, device_columns, off_values):
    """
    Evaluate the model on validation set.
    
    Parameters:
    -----------
    model : DeviceConsumptionModel
        Trained model
    x_val : pd.DataFrame
        Validation features
    y_val : pd.DataFrame
        Validation labels
    device_columns : list
        List of device names
    off_values : dict
        Dictionary of OFF values for each device
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    print("\n" + "="*70)
    print("VALIDATION PERFORMANCE")
    print("="*70)
    
    # Get predictions
    val_predictions = model.predict(x_val)
    
    metrics = {}
    
    for device in device_columns:
        off_value = off_values[device]
        
        # Ground truth binary
        y_val_binary = (y_val[device] > off_value).astype(int)
        
        # Predicted binary
        pred_binary = (val_predictions[device] > off_value).astype(int)
        
        # Classification metrics
        acc = accuracy_score(y_val_binary, pred_binary)
        f1 = f1_score(y_val_binary, pred_binary, zero_division=0)
        
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
        
        # Store metrics
        metrics[device] = {
            'accuracy': acc,
            'f1_score': f1,
            'mae_on_samples': mae,
            'rmse_on_samples': rmse,
            'num_actual_on': actual_on_mask.sum(),
            'num_predicted_on': pred_binary.sum()
        }
        
        # Print device metrics
        print(f"\n{device}:")
        print(f"  Classification:")
        print(f"    Accuracy: {acc:.4f}")
        print(f"    F1 Score: {f1:.4f}")
        print(f"    Actual ON samples: {actual_on_mask.sum()} / {len(y_val_binary)}")
        print(f"    Predicted ON samples: {pred_binary.sum()} / {len(y_val_binary)}")
        print(f"  Regression (ON samples only):")
        print(f"    MAE:  {mae:.2f}")
        print(f"    RMSE: {rmse:.2f}")
    
    # Overall metrics
    y_val_flat = y_val[device_columns].values.flatten()
    pred_val_flat = val_predictions[device_columns].values.flatten()
    
    # Remove any NaN values
    mask = ~(np.isnan(y_val_flat) | np.isnan(pred_val_flat))
    overall_mae = mean_absolute_error(y_val_flat[mask], pred_val_flat[mask])
    overall_rmse = np.sqrt(mean_squared_error(y_val_flat[mask], pred_val_flat[mask]))
    
    metrics['overall'] = {
        'mae': overall_mae,
        'rmse': overall_rmse
    }
    
    print(f"\n" + "-"*70)
    print(f"OVERALL METRICS (all devices, all time steps):")
    print(f"  MAE:  {overall_mae:.2f}")
    print(f"  RMSE: {overall_rmse:.2f}")
    print("="*70 + "\n")
    
    return metrics


def main():
    """
    Main evaluation script.
    """
    # Load and prepare data
    print("Loading datasets...")
    x_train, y_train, x_test = import_datasets()
    
    print("Preparing data...")
    x_train = prepare_data(x_train)
    x_test = prepare_data(x_test)
    y_train = prepare_label(y_train)
    
    # Align indexes
    x_train = x_train.sort_values("minutes_since_Epoch").reset_index(drop=True)
    y_train = y_train.sort_values("minutes_since_Epoch").reset_index(drop=True)
    x_test = x_test.sort_values("minutes_since_Epoch").reset_index(drop=True)
    
    # Identify device columns
    device_columns = [
        col for col in y_train.columns
        if col not in ["time_step", "minutes_since_Epoch"]
    ]
    
    print(f"Devices to predict: {device_columns}")
    
    # Train/Validation split
    print("\nSplitting train/validation...")
    x_tr, y_tr, x_val, y_val = train_val_split(x_train, y_train, val_ratio=0.2)
    print(f"Train size: {len(x_tr)}, Validation size: {len(x_val)}")
    
    # Initialize and train model
    print("\nInitializing model...")
    model = ClassifyAndRegressModel(
        device_columns=device_columns,
        n_estimators=200,
        max_depth=10,
        random_state=12
    )
    
    print("\nTraining model...")
    model.fit(x_tr, y_tr)
    
    # Evaluate on validation set
    metrics = evaluate_model(
        model=model,
        x_val=x_val,
        y_val=y_val,
        device_columns=device_columns,
        off_values=model.off_values
    )
    
    # Generate predictions on test set
    print("Generating predictions on test set...")
    predictions = model.predict(x_test)
    
    # Add time_step column and reorder
    predictions["time_step"] = x_test["time_step"]
    predictions = predictions[["time_step"] + device_columns]
    
    # Save predictions
    output_file = "Y_test_pred.csv"
    predictions.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")
    
    return model, metrics


if __name__ == "__main__":
    model, metrics = main()