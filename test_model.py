import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, classification_report
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

def plot_predictions(y_val, val_predictions, device_columns, save_path="predictions_plot.png"):
    """
    Plot predicted vs actual values for all devices over time.
    
    Parameters:
    -----------
    y_val : pd.DataFrame
        Actual validation labels
    val_predictions : pd.DataFrame
        Predicted validation labels
    device_columns : list
        List of device names
    save_path : str
        Path to save the plot
    """
    n_devices = len(device_columns)
    n_cols = 2
    n_rows = (n_devices + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    fig.suptitle('Predicted vs Actual Device Consumption Over Time', fontsize=16, y=1.00)

    if n_devices > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    time_steps = np.arange(len(y_val))
    
    for idx, device in enumerate(device_columns):
        ax = axes[idx]

        ax.plot(time_steps, y_val[device].values, 
                label='Actual', color='blue', alpha=0.7, linewidth=1.5)

        ax.plot(time_steps, val_predictions[device].values, 
                label='Predicted', color='red', alpha=0.7, linewidth=1.5, linestyle='--')
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Consumption')
        ax.set_title(f'{device}')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    for idx in range(n_devices, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPrediction plot saved to {save_path}")
    plt.close()

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

    val_predictions = model.predict(x_val)
    
    metrics = {}
    
    for device in device_columns:
        off_value = off_values[device]

        y_val_binary = (y_val[device] > off_value).astype(int)

        pred_binary = (val_predictions[device] > off_value).astype(int)

        acc = accuracy_score(y_val_binary, pred_binary)
        f1 = f1_score(y_val_binary, pred_binary, zero_division=0)

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
            r2 = r2_score(
                y_val.loc[actual_on_mask, device],
                val_predictions.loc[actual_on_mask, device]
            )
        else:
            mae = rmse = r2 = np.nan

        metrics[device] = {
            'accuracy': acc,
            'f1_score': f1,
            'mae_on_samples': mae,
            'rmse_on_samples': rmse,
            'r2_on_samples': r2,
            'num_actual_on': actual_on_mask.sum(),
            'num_predicted_on': pred_binary.sum()
        }

    y_val_flat = y_val[device_columns].values.flatten()
    pred_val_flat = val_predictions[device_columns].values.flatten()

    mask = ~(np.isnan(y_val_flat) | np.isnan(pred_val_flat))
    overall_mae = mean_absolute_error(y_val_flat[mask], pred_val_flat[mask])
    overall_rmse = np.sqrt(mean_squared_error(y_val_flat[mask], pred_val_flat[mask]))
    overall_r2 = r2_score(y_val_flat[mask], pred_val_flat[mask])
    
    metrics['overall'] = {
        'mae': overall_mae,
        'rmse': overall_rmse,
        'r2': overall_r2
    }
    
    print(f"\n" + "-"*70)
    print(f"OVERALL METRICS (all devices, all time steps):")
    print(f"  MAE: {overall_mae:.2f}")
    print(f"  RMSE: {overall_rmse:.2f}")
    print(f"  R²: {overall_r2:.4f}")
    print("="*70 + "\n")

    plot_predictions(y_val, val_predictions, device_columns)
    
    return metrics

def print_metrics_table(metrics, device_columns):
    """
    Print metrics in a formatted table.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary containing evaluation metrics
    device_columns : list
        List of device names
    """
    print("\n" + "="*70)
    print("METRICS SUMMARY")
    print("="*70)
    print(f"{'Devices':<25} {'MAE':>12} {'RMSE':>12} {'R²':>12}")
    print("-"*70)
    
    for device in device_columns:
        mae = metrics[device]['mae_on_samples']
        rmse = metrics[device]['rmse_on_samples']
        r2 = metrics[device]['r2_on_samples']
        
        print(f"{device:<25} {mae:>12.3f} {rmse:>12.3f} {r2:>12.3f}")
    
    print("="*70 + "\n")

def main():
    """
    Main evaluation script.
    """
    print("Loading datasets...")
    x_train, y_train, x_test = import_datasets()
    
    print("Preparing data...")
    x_train = prepare_data(x_train)
    x_test = prepare_data(x_test)
    y_train = prepare_label(y_train)

    x_train = x_train.sort_values("minutes_since_Epoch").reset_index(drop=True)
    y_train = y_train.sort_values("minutes_since_Epoch").reset_index(drop=True)
    x_test = x_test.sort_values("minutes_since_Epoch").reset_index(drop=True)

    device_columns = [
        col for col in y_train.columns 
        if col not in ["time_step", "minutes_since_Epoch"]
    ]
    print(f"Devices to predict: {device_columns}")

    print("\nSplitting train/validation...")
    x_tr, y_tr, x_val, y_val = train_val_split(x_train, y_train, val_ratio=0.2)
    print(f"Train size: {len(x_tr)}, Validation size: {len(x_val)}")

    print("\nInitializing model...")
    model = ClassifyAndRegressModel(
        device_columns=device_columns,
        n_estimators=200,
        max_depth=10,
        random_state=12
    )
    
    print("\nTraining model...")
    model.fit(x_tr, y_tr)
    
    print("Train results")

    metrics = evaluate_model(
        model=model,
        x_val=x_val,
        y_val=y_val,
        device_columns=device_columns,
        off_values=model.off_values
    )

    print_metrics_table(metrics, device_columns)
    
    print("Test results")

    metrics = evaluate_model(
        model=model,
        x_val=x_val,
        y_val=y_val,
        device_columns=device_columns,
        off_values=model.off_values
    )

    print_metrics_table(metrics, device_columns)

    print("Generating predictions on test set...")
    predictions = model.predict(x_test)

    predictions["time_step"] = x_test["time_step"]
    predictions = predictions[["time_step"] + device_columns]

    output_file = "Y_test_pred.csv"
    predictions.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")
    
    return model, metrics

if __name__ == "__main__":
    model, metrics = main()