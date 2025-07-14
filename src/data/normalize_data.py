import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os
import logging

def normalize_data(train_path="data/processed_data/X_train.csv",
                   test_path="data/processed_data/X_test.csv",
                   output_train_scaled="data/processed_data/X_train_scaled.csv",
                   output_test_scaled="data/processed_data/X_test_scaled.csv",
                   scaler_path="models/data/scaler.pkl"):
    # Setup logger
    logger = logging.getLogger(__name__)

    # Load datasets
    logger.info("Loading training and testing data...")
    X_train = pd.read_csv(train_path)
    X_test = pd.read_csv(test_path)

    # Select only numeric columns for scaling
    X_train_numeric = X_train.select_dtypes(include=['number'])
    X_test_numeric = X_test.select_dtypes(include=['number'])

    # Fit scaler on training data and transform both sets
    logger.info("Fitting StandardScaler and transforming data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_numeric)
    X_test_scaled = scaler.transform(X_test_numeric)

    # Convert scaled arrays back to DataFrames with original columns
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train_numeric.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test_numeric.columns)

    # Ensure output directories exist
    os.makedirs(os.path.dirname(output_train_scaled), exist_ok=True)
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)

    # Save scaled data
    logger.info(f"Saving scaled training data to {output_train_scaled}")
    X_train_scaled_df.to_csv(output_train_scaled, index=False)

    logger.info(f"Saving scaled testing data to {output_test_scaled}")
    X_test_scaled_df.to_csv(output_test_scaled, index=False)

    # Save the scaler object for reuse
    logger.info(f"Saving scaler object to {scaler_path}")
    joblib.dump(scaler, scaler_path)

    logger.info("Normalization complete.")

def main():
    normalize_data()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    main()
