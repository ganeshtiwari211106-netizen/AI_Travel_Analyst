import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from data_cleaning import load_and_clean_data

def train_and_save_model(model_dir="saved_models"):
    print("[+] Loading clean dataset...")
    df = load_and_clean_data()
    
    X = df.drop(columns=['Price'])
    y = df['Price']
    
    # One-Hot Encoding for categorical features
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    # 80/20 Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )
    
    print("[+] Training HistGradientBoostingRegressor (Sequential Residual Boosting)...")
    model = HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.08,
        max_depth=12,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    
    print("\n--- Advanced Model Evaluation ---")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Absolute Error (MAE): ₹{mae:.2f}")
    
    # Save the updated model and feature schema
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, f"{model_dir}/flight_model.pkl")
    joblib.dump(list(X_encoded.columns), f"{model_dir}/model_features.pkl")
    print(f"\n[+] Upgraded model and schema saved to '{model_dir}/' successfully.")

if __name__ == "__main__":
    train_and_save_model()