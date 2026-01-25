import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class ClassifyAndRegressModel:
    """
    Two-stage model for predicting device consumption:
    1. Classification: Predict if device is ON or OFF
    2. Regression: Predict consumption level when device is ON
    """
    
    def __init__(self, device_columns, off_values=None, n_estimators=200, max_depth=10, random_state=12):
        """
        Initialize the model.
        
        Parameters:
        -----------
        device_columns : list
            List of device column names to predict
        off_values : dict, optional
            Dictionary mapping device names to their OFF consumption values
            Default is 0 for all devices except TVs (7)
        n_estimators : int
            Number of trees in RandomForest
        max_depth : int
            Maximum depth of RandomForest trees
        random_state : int
            Random state for reproducibility
        """
        self.device_columns = device_columns
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        
        # Set default OFF values
        if off_values is None:
            self.off_values = {device: 0 for device in device_columns}
            for device in device_columns:
                if "tv" in device.lower():
                    self.off_values[device] = 7
        else:
            self.off_values = off_values
        
        # Initialize model containers
        self.classifiers = {}
        self.regressors = {}
        self.feature_cols = None
        
    def add_temporal_features(self, df):
        """
        Add time-based features to the dataframe.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with time_step column
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with added temporal features
        """
        df = df.copy()
        
        # Convert time_step to numeric if it's not already
        time_step_numeric = pd.to_numeric(df['time_step'], errors='coerce')
        
        # Cyclical time encoding (assuming time_step is in seconds)
        # Hour of day cycle (24 hours = 86400 seconds)
        df['hour_sin'] = np.sin(2 * np.pi * time_step_numeric / 86400)
        df['hour_cos'] = np.cos(2 * np.pi * time_step_numeric / 86400)
        
        # Day of week cycle (7 days = 604800 seconds)
        df['day_sin'] = np.sin(2 * np.pi * time_step_numeric / 604800)
        df['day_cos'] = np.cos(2 * np.pi * time_step_numeric / 604800)
        
        return df
    
    def handle_missing_values(self, df):
        """
        Handle missing values using forward-fill then back-fill.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with NaN values filled
        """
        df = df.copy()
        
        # Forward fill then backward fill
        df = df.ffill().bfill()
        
        # If any NaN still remain (shouldn't happen), fill with 0
        df = df.fillna(0)
        
        return df
        
    def fit(self, x_train, y_train):
        """
        Train the models for all devices.
        
        Parameters:
        -----------
        x_train : pd.DataFrame
            Training features
        y_train : pd.DataFrame
            Training labels (device consumptions)
        """
        # Add temporal features
        x_train = self.add_temporal_features(x_train)
        
        # Handle missing values
        x_train = self.handle_missing_values(x_train)
        y_train = self.handle_missing_values(y_train)
        
        # Identify feature columns (exclude time_step)
        self.feature_cols = [
            col for col in x_train.columns
            if col not in ["time_step"]
        ]
        
        for device in self.device_columns:
            off_value = self.off_values[device]
            
            # Check if device is a fridge/freezer (always-on device)
            is_fridge = "fridge" in device.lower() or "freezer" in device.lower()
            
            if is_fridge:
                # Don't classify fridges, only regress
                self.classifiers[device] = None
                
                reg = LinearRegression()
                reg.fit(x_train[self.feature_cols], y_train[device])
                self.regressors[device] = reg
                
            else:
                # Binary target: ON / OFF
                y_binary = (y_train[device] > off_value).astype(int)
                
                # -------- Train Classifier --------
                clf = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    random_state=self.random_state,
                    n_jobs=-1,
                    class_weight='balanced',
                    min_samples_leaf=2  # More sensitive to rare patterns
                )
                clf.fit(x_train[self.feature_cols], y_binary)
                self.classifiers[device] = clf
                
                # -------- Train Regressor (ON samples only) --------
                on_mask = y_binary == 1
                
                reg = LinearRegression()
                
                reg.fit(
                    x_train.loc[on_mask, self.feature_cols],
                    y_train.loc[on_mask, device]
                )
                
                self.regressors[device] = reg
                
        print("Training completed!\n")
        
    def predict(self, x_test):
        """
        Predict device consumptions.
        
        Parameters:
        -----------
        x_test : pd.DataFrame
            Test features
            
        Returns:
        --------
        pd.DataFrame
            Predictions for all devices
        """
        if self.feature_cols is None:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")
        
        # Add temporal features
        x_test = self.add_temporal_features(x_test)
        
        # Handle missing values
        x_test = self.handle_missing_values(x_test)
        
        predictions = pd.DataFrame(
            index=x_test.index,
            columns=self.device_columns,
            dtype=float
        )
        
        for device in self.device_columns:
            off_value = self.off_values[device]
            
            # Check if device is a fridge (always-on, no classifier)
            if self.classifiers[device] is None:
                # Predict directly with regressor
                predictions[device] = self.regressors[device].predict(
                    x_test[self.feature_cols]
                )
            else:
                # Predict ON / OFF
                on_pred = self.classifiers[device].predict(x_test[self.feature_cols])
                
                # Default to OFF value
                predictions[device] = float(off_value)
                
                # Predict consumption where ON
                on_indices = np.where(on_pred == 1)[0]
                
                if len(on_indices) > 0:
                    predictions.loc[x_test.index[on_indices], device] = self.regressors[device].predict(
                        x_test.loc[x_test.index[on_indices], self.feature_cols]
                    )
        
        return predictions
    
    def predict_probabilities(self, x_test):
        """
        Predict probabilities of devices being ON.
        
        Parameters:
        -----------
        x_test : pd.DataFrame
            Test features
            
        Returns:
        --------
        pd.DataFrame
            Probabilities for each device being ON
        """
        if self.feature_cols is None:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")
        
        # Add temporal features
        x_test = self.add_temporal_features(x_test)
        
        # Handle missing values
        x_test = self.handle_missing_values(x_test)
        
        probabilities = pd.DataFrame(
            index=x_test.index,
            columns=self.device_columns,
            dtype=float
        )
        
        for device in self.device_columns:
            # Check if device is a fridge (no classifier)
            if self.classifiers[device] is None:
                # Always-on devices have probability 1.0
                probabilities[device] = 1.0
            else:
                # Get probability of class 1 (ON)
                proba = self.classifiers[device].predict_proba(x_test[self.feature_cols])
                probabilities[device] = proba[:, 1]
        
        return probabilities