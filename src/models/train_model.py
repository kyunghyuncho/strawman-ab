import pandas as pd
import joblib
from scipy.sparse import hstack
from sklearn.linear_model import Ridge
from sklearn.utils import resample
import logging
import sys
import os
import numpy as np

# Ensure the source directory is in the Python path to import config
# This allows the script to be run from the root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import (
    DATA_FILE, TARGET_PROPERTIES, FOLD_COLUMN, FOLDS, VH_SEQUENCE_COL,
    VL_SEQUENCE_COL, HC_SUBTYPE_COL, K_BOOTSTRAP, MODEL_PARAMS, ARTEFACTS_DIR,
    LOG_TRANSFORM_TARGETS
)

# Configure logging to provide informative output during execution
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train():
    """
    Trains and saves model ensembles for each target property using 5-fold cross-validation.

    This function orchestrates the entire training pipeline:
    1. Loads the dataset and pre-fitted feature transformers.
    2. Iterates through each of the five target properties.
    3. For each target, it performs a 5-fold cross-validation scheme.
    4. In each fold, an ensemble of `K_BOOTSTRAP` regressors is trained on
       bootstrap-resampled data from the other four folds.
    5. The trained ensembles for all five folds are saved to a single file
       per target property in the `artefacts/` directory.
    """
    logging.info("Starting model training process...")

    # Create the artefacts directory if it doesn't exist
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load the main dataset from the path specified in the config
    try:
        df = pd.read_csv(DATA_FILE)
        logging.info(f"Successfully loaded data from {DATA_FILE}. Shape: {df.shape}")
    except FileNotFoundError:
        logging.error(f"Error: The data file was not found at {DATA_FILE}. Please ensure it is in the correct location.")
        return

    # Load the pre-fitted transformers created by the feature generation script
    try:
        vectorizer_vh = joblib.load(ARTEFACTS_DIR / 'vectorizer_vh.joblib')
        vectorizer_vl = joblib.load(ARTEFACTS_DIR / 'vectorizer_vl.joblib')
        encoder_ohe = joblib.load(ARTEFACTS_DIR / 'encoder_ohe.joblib')
        logging.info("Successfully loaded pre-fitted transformers (vectorizers and OHE).")
    except FileNotFoundError as e:
        logging.error(f"Error loading transformers: {e}. Please run `src/features/build_features.py` first.")
        return

    # Loop through each target property to train a separate set of models
    for target in TARGET_PROPERTIES:
        logging.info(f"--- Processing Target: {target} ---")

        # Create a copy of the dataframe and drop rows where the current target is missing
        # This ensures that we only train on data with valid target values
        df_target = df.dropna(subset=[target]).copy()
        logging.info(f"Training on {len(df_target)} samples for {target} after dropping NaNs.")

        # Apply log transformation if the target is in the specified list
        if target in LOG_TRANSFORM_TARGETS:
            df_target[target] = np.log1p(df_target[target])
            logging.info(f"Applied log1p transformation to target: {target}")

        # This dictionary will store the ensembles for each of the 5 folds
        all_fold_ensembles = {}

        # Outer cross-validation loop (iterating through folds 1 to 5)
        for fold_i in FOLDS:
            logging.info(f"  - Processing Fold {fold_i}/{len(FOLDS)}...")

            # Split data into training and hold-out (test) sets for this fold
            # The training set consists of all data NOT in the current fold
            train_mask = (df_target[FOLD_COLUMN] != fold_i)
            df_train = df_target[train_mask]
            
            y_train = df_train[target]

            # Transform the text and categorical features for the training set
            # using the pre-fitted transformers.
            X_train_vh = vectorizer_vh.transform(df_train[VH_SEQUENCE_COL])
            X_train_vl = vectorizer_vl.transform(df_train[VL_SEQUENCE_COL])
            X_train_ohe = encoder_ohe.transform(df_train[[HC_SUBTYPE_COL]])

            # Concatenate the sparse matrices into a single feature matrix
            X_train = hstack([X_train_vh, X_train_vl, X_train_ohe], format='csr')
            logging.info(f"    Training data shape for fold {fold_i}: {X_train.shape}")

            current_ensemble_models = []
            # Inner bootstrap loop to create the ensemble of models
            for k in range(K_BOOTSTRAP):
                # Resample the training data with replacement to create a bootstrap sample
                X_boot, y_boot = resample(X_train, y_train)

                # Initialize and train the sparse linear regression model
                model = Ridge(**MODEL_PARAMS)
                model.fit(X_boot, y_boot)

                current_ensemble_models.append(model)
            
            logging.info(f"    Trained ensemble of {K_BOOTSTRAP} models for fold {fold_i}.")
            # Store the trained ensemble for the current fold
            all_fold_ensembles[fold_i] = current_ensemble_models

        # After iterating through all folds, save the complete dictionary of ensembles
        # for the current target property. This file contains 5 * K_BOOTSTRAP models.
        model_filename = ARTEFACTS_DIR / f'models_{target}.joblib'
        joblib.dump(all_fold_ensembles, model_filename)
        logging.info(f"Saved all 5 fold-ensembles for {target} to {model_filename}")

    logging.info("Model training process completed for all target properties.")

if __name__ == '__main__':
    train()
