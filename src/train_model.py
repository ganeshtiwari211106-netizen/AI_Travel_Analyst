import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from data_cleaning import load_and_clean_data

def train_and_save_model(model_dir="saved_models"):
    print("Loading cleaned dataset with enriched features...")
    df = load_and_clean_data()
    
    X = df.drop(columns=['Price'])
    y = df['Price']
    
    # One-Hot Encoding for all categorical variables
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    
    print("Training optimized Random Forest model (100 estimators)...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=22,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    print("\n--- Optimized Model Evaluation ---")
    print(f"R² Score: {r2_score(y_test, preds):.4f}")
    print(f"Mean Absolute Error (MAE): ₹{mean_absolute_error(y_test, preds):.2f}")
    
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, f"{model_dir}/flight_model.pkl")
    joblib.dump(list(X_encoded.columns), f"{model_dir}/model_features.pkl")
    print(f"Optimized model artifacts saved to '{model_dir}' folder.")

if __name__ == "__main__":
    train_and_save_model()