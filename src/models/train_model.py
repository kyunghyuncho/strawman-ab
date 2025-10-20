import pandas as pd
import joblib
from scipy.sparse import hstack
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.utils import resample
import logging
import sys
import os
import numpy as np
import argparse
import json

# Ensure the source directory is in the Python path to import config
# This allows the script to be run from the root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import (
    DATA_FILE, TARGET_PROPERTIES, FOLD_COLUMN, FOLDS, VH_SEQUENCE_COL,
    VL_SEQUENCE_COL, HC_SUBTYPE_COL, K_BOOTSTRAP, MODEL_PARAMS, ARTEFACTS_DIR,
    LOG_TRANSFORM_TARGETS, MODEL_TYPE, TRIM_OUTLIERS, TRIM_QUANTILE
)

# Configure logging to provide informative output during execution
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train(model_type: str, use_best: bool = True):
    """
    Trains and saves model ensembles for each target property using 5-fold cross-validation.

    This function orchestrates the entire training pipeline:
    1. Loads the dataset and pre-fitted feature transformers.
    2. Iterates through each of the five target properties.
    3. For each target, it performs a 5-fold cross-validation scheme.
    4. In each fold, an ensemble of `K_BOOTSTRAP` regressors is trained on
       bootstrap-resampled data from the other four folds.
    5. The trained ensembles for all five folds are saved to a single file
       per target property in the `artefacts/` directory, named according to the model type.
    """
    logging.info(f"Starting model training process for model_type='{model_type}'...")

    # Create the artefacts directory if it doesn't exist
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load the main dataset from the path specified in the config
    try:
        df = pd.read_csv(DATA_FILE)
        logging.info(f"Successfully loaded data from {DATA_FILE}. Shape: {df.shape}")
    except FileNotFoundError:
        logging.error(f"Error: The data file was not found at {DATA_FILE}. Please ensure it is in the correct location.")
        return

    # Try to load global transformers (optional). We'll prefer per-target if present.
    vectorizer_vh_global = None
    vectorizer_vl_global = None
    encoder_ohe_global = None
    try:
        vectorizer_vh_global = joblib.load(ARTEFACTS_DIR / 'vectorizer_vh.joblib')
        vectorizer_vl_global = joblib.load(ARTEFACTS_DIR / 'vectorizer_vl.joblib')
        encoder_ohe_global = joblib.load(ARTEFACTS_DIR / 'encoder_ohe.joblib')
        logging.info("Loaded global transformers (vectorizers and OHE). Will override with per-target if present.")
    except FileNotFoundError:
        logging.warning("Global transformers not found. Will look for per-target transformers for each target.")

    # --- Model Selection ---
    if model_type == 'ridge':
        model_class = Ridge
    elif model_type == 'gbr':
        model_class = GradientBoostingRegressor
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    params_template = MODEL_PARAMS[model_type].copy()
    # Load best params for possible per-target overrides
    best_params_path = ARTEFACTS_DIR / 'best_params.json'
    best_params = None
    if use_best and best_params_path.exists():
        try:
            with open(best_params_path, 'r') as f:
                best_params = json.load(f)
        except Exception as e:
            logging.warning(f"Could not read best_params.json: {e}")
    logging.info(f"Using model: {model_class.__name__} with base parameters: {params_template}")

    # Loop through each target property to train a separate set of models
    for target in TARGET_PROPERTIES:
        logging.info(f"--- Processing Target: {target} ---")

        # Determine transformers for this target
        vectorizer_vh = None
        vectorizer_vl = None
        encoder_ohe = None
        vh_path = ARTEFACTS_DIR / f'vectorizer_vh_{target}.joblib'
        vl_path = ARTEFACTS_DIR / f'vectorizer_vl_{target}.joblib'
        ohe_path = ARTEFACTS_DIR / f'encoder_ohe_{target}.joblib'
        try:
            if vh_path.exists() and vl_path.exists() and ohe_path.exists():
                vectorizer_vh = joblib.load(vh_path)
                vectorizer_vl = joblib.load(vl_path)
                encoder_ohe = joblib.load(ohe_path)
                logging.info(f"Using per-target transformers for {target}.")
            elif all(x is not None for x in [vectorizer_vh_global, vectorizer_vl_global, encoder_ohe_global]):
                vectorizer_vh = vectorizer_vh_global
                vectorizer_vl = vectorizer_vl_global
                encoder_ohe = encoder_ohe_global
                logging.info(f"Per-target transformers for {target} not found. Using global transformers.")
            else:
                missing = []
                if not vh_path.exists():
                    missing.append(vh_path.name)
                if not vl_path.exists():
                    missing.append(vl_path.name)
                if not ohe_path.exists():
                    missing.append(ohe_path.name)
                logging.error(
                    "No transformers available for target '%s'. Missing per-target: %s. "
                    "Either run 'python -m src.features.build_features --target %s' "
                    "to create per-target transformers, or run 'python -m src.features.build_features' "
                    "to create global transformers.",
                    target,
                    ", ".join(missing) if missing else "unknown",
                    target,
                )
                return
        except Exception as e:
            logging.error(f"Failed to load transformers for {target}: {e}")
            return

        # Determine model params for this target
        params = params_template.copy()
        if model_type == 'ridge' and use_best and best_params:
            try:
                # Prefer per_target alpha, then global, then default
                per_target = (best_params.get('per_target') or {}).get(target) or {}
                global_best = best_params.get('global') or {}
                alpha = per_target.get('alpha') or global_best.get('alpha') or best_params.get('alpha')
                if alpha is not None and float(alpha) > 0:
                    params['alpha'] = float(alpha)
                    logging.info(f"Overriding Ridge alpha for {target} with tuned value: {alpha}")
            except Exception:
                pass

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
            df_train_full = df_target[train_mask]

            # --- Outlier Trimming ---
            if TRIM_OUTLIERS:
                lower_bound = df_train_full[target].quantile(TRIM_QUANTILE)
                upper_bound = df_train_full[target].quantile(1 - TRIM_QUANTILE)
                
                original_count = len(df_train_full)
                df_train = df_train_full[
                    (df_train_full[target] >= lower_bound) & (df_train_full[target] <= upper_bound)
                ]
                trimmed_count = original_count - len(df_train)
                logging.info(f"    Trimmed {trimmed_count} outliers from training data for fold {fold_i} ({((trimmed_count/original_count)*100):.2f}%).")
            else:
                df_train = df_train_full
            
            y_train = df_train[target]

            # Transform the text and categorical features for the training set
            # using the pre-fitted transformers.
            X_train_vh = vectorizer_vh.transform(df_train[VH_SEQUENCE_COL])
            X_train_vl = vectorizer_vl.transform(df_train[VL_SEQUENCE_COL])
            # Ensure subtype has no missing values (encoder was fitted with 'Unknown' for missing)
            X_train_ohe = encoder_ohe.transform(df_train[[HC_SUBTYPE_COL]].fillna('Unknown'))

            # Concatenate the sparse matrices into a single feature matrix
            X_train = hstack([X_train_vh, X_train_vl, X_train_ohe], format='csr')
            logging.info(f"    Training data shape for fold {fold_i}: {X_train.shape}")

            current_ensemble_models = []
            # Inner bootstrap loop to create the ensemble of models
            for k in range(K_BOOTSTRAP):
                # Resample the training data with replacement to create a bootstrap sample
                X_boot, y_boot = resample(X_train, y_train)

                # Initialize and train the regression model
                model = model_class(**params)
                model.fit(X_boot, y_boot)

                current_ensemble_models.append(model)
            
            logging.info(f"    Trained ensemble of {K_BOOTSTRAP} models for fold {fold_i}.")
            # Store the trained ensemble for the current fold
            all_fold_ensembles[fold_i] = current_ensemble_models

        # After iterating through all folds, save the complete dictionary of ensembles
        # for the current target property. This file contains 5 * K_BOOTSTRAP models.
        model_filename = ARTEFACTS_DIR / f'models_{target}_{model_type}.joblib'
        joblib.dump(all_fold_ensembles, model_filename)
        logging.info(f"Saved all 5 fold-ensembles for {target} to {model_filename}")

    logging.info(f"Model training process completed for {model_type} models.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train models for antibody property prediction.")
    parser.add_argument(
        '--model-type', 
        type=str, 
        default=MODEL_TYPE, 
        choices=['ridge', 'gbr'],
        help="The type of model to train ('ridge' or 'gbr'). Defaults to the value in config.py."
    )
    parser.add_argument('--no-use-best', action='store_true', help='Do not use tuned alpha from best_params.json')
    args = parser.parse_args()
    
    train(model_type=args.model_type, use_best=not args.no_use_best)
