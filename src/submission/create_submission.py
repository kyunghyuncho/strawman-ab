"""
This script loads the raw cross-validation and private test set predictions,
formats them into the required submission format, and saves them as CSV files.
"""
import pandas as pd
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src import config

def create_submission_files():
    """
    Loads raw predictions, formats them, and saves the final submission files.
    """
    print("Starting submission file creation...")

    # --- Define paths for input and output files ---
    cv_preds_raw_path = config.RESULTS_DIR / 'gdp_a1_cv_predictions_raw.csv'
    private_preds_raw_path = config.RESULTS_DIR / 'private_test_predictions_raw.csv'
    
    cv_submission_path = config.RESULTS_DIR / 'gdp_a1_cv_predictions.csv'
    private_submission_path = config.RESULTS_DIR / 'private_test_predictions.csv'

    # --- Load Raw Predictions ---
    try:
        cv_preds_raw = pd.read_csv(cv_preds_raw_path)
        print("Successfully loaded raw CV predictions.")
    except FileNotFoundError:
        print(f"Error: Raw CV prediction file not found at {cv_preds_raw_path}")
        print("Please ensure you have run src/evaluation/evaluate_model.py successfully.")
        return

    try:
        private_preds_raw = pd.read_csv(private_preds_raw_path)
        print("Successfully loaded raw private test predictions.")
    except FileNotFoundError:
        print(f"Error: Raw private test prediction file not found at {private_preds_raw_path}")
        print("Please ensure you have run src/models/predict_model.py successfully.")
        return

    # --- Format and Save CV Submission File ---
    # The required columns are 'antibody_name', the fold column, and the target properties
    cv_submission_cols = ['antibody_name', config.FOLD_COLUMN] + config.TARGET_PROPERTIES
    
    # Ensure all required columns exist in the dataframe
    cv_submission_df = cv_preds_raw[cv_submission_cols]

    # --- Fill missing values with the mean of each respective column ---
    print("\nHandling missing values in CV predictions...")
    for col in config.TARGET_PROPERTIES:
        if cv_submission_df[col].isnull().any():
            # Calculate mean on the raw data before it's altered
            mean_val = cv_preds_raw[col].mean()
            cv_submission_df.loc[:, col] = cv_submission_df[col].fillna(mean_val)
            print(f"  - Filled {cv_submission_df[col].isnull().sum()} NaNs in '{col}' with mean: {mean_val:.4f}")
    
    # Save the formatted CV predictions
    cv_submission_df.to_csv(cv_submission_path, index=False)
    print(f"\nCV submission file saved to: {cv_submission_path}")

    # --- Format and Save Private Test Submission File ---
    # The required columns are 'antibody_name' and the target properties
    private_submission_cols = ['antibody_name'] + config.TARGET_PROPERTIES
    private_submission_df = private_preds_raw[private_submission_cols]

    # --- Fill missing values with the mean of each respective column ---
    print("\nHandling missing values in private test predictions...")
    for col in config.TARGET_PROPERTIES:
        if private_submission_df[col].isnull().any():
            # Calculate mean on the raw data before it's altered
            mean_val = private_preds_raw[col].mean()
            private_submission_df.loc[:, col] = private_submission_df[col].fillna(mean_val)
            print(f"  - Filled {private_submission_df[col].isnull().sum()} NaNs in '{col}' with mean: {mean_val:.4f}")

    # Save the formatted private test predictions
    private_submission_df.to_csv(private_submission_path, index=False)
    print(f"Private test submission file saved to: {private_submission_path}")
    
    print("\nSubmission files created successfully.")

if __name__ == '__main__':
    # Create the results directory if it doesn't exist
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    create_submission_files()
