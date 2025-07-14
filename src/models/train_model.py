import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
import os

def main():
    X = pd.read_csv("data/processed_data/X_train_scaled.csv")
    y = pd.read_csv("data/processed_data/y_train.csv").values.ravel()

    best_params = joblib.load("models/data/best_params.pkl")
    model = RandomForestRegressor(**best_params, random_state=42)
    model.fit(X, y)

    os.makedirs("models/models", exist_ok=True)
    joblib.dump(model, "models/models/model.pkl")

if __name__ == "__main__":
    main()
