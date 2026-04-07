import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib

def train_models():
    # Load Data
    try:
        df = pd.read_csv('tire_wear_data.csv')
    except Exception as e:
        print("Data file not found. Ensure you run generate_data.py first.")
        return
        
    X = df[['Lap_Number', 'Tire_Type', 'Track_Temp', 'Track_Type']]
    y = df['Lap_Time']
    
    # Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Baseline Model (Linear Regression)
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))
    print(f"Linear Regression RMSE (Baseline): {lr_rmse:.4f}")
    
    # Primary Model (Random Forest Regressor)
    # Excellent for capturing non-linear tire degradation and "cliffs"
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
    print(f"Random Forest RMSE: {rf_rmse:.4f}")
    
    # Save the RF Model for Streamlit app
    joblib.dump(rf_model, 'model.joblib')
    print("Saved Random Forest Model to model.joblib.")

if __name__ == '__main__':
    train_models()
