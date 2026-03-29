import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def main():
    # 1. Load data
    print("Loading data.csv...")
    try:
        df = pd.read_csv('data.csv')
    except FileNotFoundError:
        print("Error: data.csv not found in the current directory.")
        return

    # 2. Preprocessing
    print("Preprocessing data...")
    # Drop non-numeric or high-cardinality columns for simplicity
    columns_to_drop = ['date', 'street', 'city', 'statezip', 'country']
    df = df.drop(columns=columns_to_drop, errors='ignore')

    # Drop rows with missing target values just in case
    df = df.dropna(subset=['price'])
    
    # Separate features (X) and target variable (y)
    X = df.drop(columns=['price'])
    y = df['price']

    # 3. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Initialize and Train Model
    print("Training RandomForestRegressor model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # 5. Evaluate Model
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("-" * 30)
    print("Model Performance:")
    print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
    print(f"R-squared (R2) Score:         {r2:.4f}")
    print("-" * 30)

    # 6. Save Model
    model_filename = 'house_price_model.pkl'
    joblib.dump(model, model_filename)
    print(f"Model successfully saved to '{model_filename}'")

if __name__ == "__main__":
    main()
