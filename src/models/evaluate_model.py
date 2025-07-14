import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import json
import os

def main():
    X_test = pd.read_csv("data/processed_data/X_test_scaled.csv")
    y_test = pd.read_csv("data/processed_data/y_test.csv")

    model = joblib.load("models/models/model.pkl")
    predictions = model.predict(X_test)

    os.makedirs("data", exist_ok=True)
    pd.DataFrame({"y_true": y_test.values.ravel(), "y_pred": predictions}).to_csv("data/processed_data/predictions.csv", index=False)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    os.makedirs("metrics", exist_ok=True)
    with open("metrics/scores.json", "w") as f:
        json.dump({"mse": mse, "r2": r2}, f)

if __name__ == "__main__":
    main()
