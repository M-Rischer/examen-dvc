import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import joblib
import os

def main():
    X = pd.read_csv("data/processed_data/X_train_scaled.csv")
    y = pd.read_csv("data/processed_data/y_train.csv").values.ravel()

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None]
    }

    model = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1)
    grid_search.fit(X, y)

    os.makedirs("models/data", exist_ok=True)
    joblib.dump(grid_search.best_params_, "models/data/best_params.pkl")

if __name__ == "__main__":
    main()
