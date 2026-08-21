import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from data_cleaning import load_and_clean_data

def train_and_save_model(model_dir="saved_models"):
    print("Loading clean data...")
    df = load_and_clean_data()
    
    print("Engineering features and encoding categories...")
    # Separate the target (Price) from the features
    X = df.drop(columns=['Price'])
    y = df['Price']
    
    # One-Hot Encoding: Converts text categories into binary (0 or 1) columns so the math model can read them
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    # Split the data: 80% for training the model, 20% for testing it
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest Regressor (this might take a few seconds)...")
    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    print("\n--- Model Evaluation ---")
    preds = model.predict(X_test)
    print(f"R² Score: {r2_score(y_test, preds):.4f}")
    print(f"Mean Absolute Error (MAE): ₹{mean_absolute_error(y_test, preds):.2f}")
    
    # Save the trained model and the exact list of features it expects
    import os
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, f"{model_dir}/flight_model.pkl")
    joblib.dump(list(X_encoded.columns), f"{model_dir}/model_features.pkl")
    print(f"\nSuccess! Model and features saved to the '{model_dir}' folder.")

if __name__ == "__main__":
    train_and_save_model()