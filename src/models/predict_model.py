"""
Generate predictions for the private test set using the trained models.
"""
import pandas as pd
import numpy as np
import joblib
from scipy.sparse import hstack
import sys
import os
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src import config

def predict(model_type: str):
    """
    Generate and save predictions for the private held-out test set.
    """
    print(f"--- Generating predictions for the private test set using model_type='{model_type}' ---")

    # Load the test data
    test_file = config.DATA_DIR / 'heldout-set-sequences.csv'
    df_test = pd.read_csv(test_file)

    # Load the global transformers (optional fallback)
    vectorizer_vh_global = None
    vectorizer_vl_global = None
    encoder_ohe_global = None
    try:
        vectorizer_vh_global = joblib.load(config.ARTEFACTS_DIR / 'vectorizer_vh.joblib')
        vectorizer_vl_global = joblib.load(config.ARTEFACTS_DIR / 'vectorizer_vl.joblib')
        encoder_ohe_global = joblib.load(config.ARTEFACTS_DIR / 'encoder_ohe.joblib')
        print("Loaded global transformers (vectorizers and OHE). Will override with per-target if present.")
    except FileNotFoundError:
        print("Global transformers not found. Will use per-target transformers if available for each target.")

    # Initialize a dataframe to store all predictions
    predictions_df = pd.DataFrame({'antibody_name': df_test['antibody_name']})

    # --- Loop through each target property ---
    for target in config.TARGET_PROPERTIES:
        print(f"  Predicting for target: {target}...")

        # Load per-target transformers if available
        vectorizer_vh = None
        vectorizer_vl = None
        encoder_ohe = None
        vh_path = config.ARTEFACTS_DIR / f'vectorizer_vh_{target}.joblib'
        vl_path = config.ARTEFACTS_DIR / f'vectorizer_vl_{target}.joblib'
        ohe_path = config.ARTEFACTS_DIR / f'encoder_ohe_{target}.joblib'
        if vh_path.exists() and vl_path.exists() and ohe_path.exists():
            vectorizer_vh = joblib.load(vh_path)
            vectorizer_vl = joblib.load(vl_path)
            encoder_ohe = joblib.load(ohe_path)
        elif all(x is not None for x in [vectorizer_vh_global, vectorizer_vl_global, encoder_ohe_global]):
            vectorizer_vh = vectorizer_vh_global
            vectorizer_vl = vectorizer_vl_global
            encoder_ohe = encoder_ohe_global
        else:
            missing = []
            if not vh_path.exists():
                missing.append(vh_path.name)
            if not vl_path.exists():
                missing.append(vl_path.name)
            if not ohe_path.exists():
                missing.append(ohe_path.name)
            print(
                f"Error: No transformers available for target '{target}'. Missing per-target: {', '.join(missing)}. "
                "Please run: python -m src.features.build_features --target all"
            )
            continue

        # Transform test features using the chosen transformers
        X_test_vh = vectorizer_vh.transform(df_test[config.VH_SEQUENCE_COL])
        X_test_vl = vectorizer_vl.transform(df_test[config.VL_SEQUENCE_COL])
        X_test_ohe = encoder_ohe.transform(df_test[[config.HC_SUBTYPE_COL]].fillna('Unknown'))
        X_test_final = hstack([X_test_vh, X_test_vl, X_test_ohe])

        # Load the full set of model ensembles for the target
        model_path = config.ARTEFACTS_DIR / f'models_{target}_{model_type}.joblib'
        try:
            all_fold_ensembles = joblib.load(model_path)
        except FileNotFoundError:
            print(f"Error: Model file not found at {model_path}. Please train the '{model_type}' models first.")
            continue

        all_ensemble_predictions = []

        # --- Average predictions across all 5 fold-specific ensembles ---
        for fold_i in config.FOLDS:
            current_ensemble_models = all_fold_ensembles[fold_i]
            
            # Average predictions from the bootstrap models within this fold's ensemble
            preds_k = [model.predict(X_test_final) for model in current_ensemble_models]
            y_pred_ensemble_avg = np.mean(preds_k, axis=0)
            all_ensemble_predictions.append(y_pred_ensemble_avg)
        
        # Final prediction is the average of the 5 ensemble predictions
        final_predictions = np.mean(all_ensemble_predictions, axis=0)
        
        # Inverse transform predictions if the target was log-transformed
        if target in config.LOG_TRANSFORM_TARGETS:
            final_predictions = np.expm1(final_predictions)

        # Add predictions to the dataframe
        predictions_df[target] = final_predictions

    # Save the final predictions dataframe
    output_path = config.RESULTS_DIR / f'private_test_predictions_raw_{model_type}.csv'
    predictions_df.to_csv(output_path, index=False)
    print(f"\nTest set predictions saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate predictions using trained models.")
    parser.add_argument(
        '--model-type', 
        type=str, 
        default=config.MODEL_TYPE, 
        choices=['ridge', 'gbr'],
        help="The type of model to use for predictions ('ridge' or 'gbr')."
    )
    args = parser.parse_args()

    predict(model_type=args.model_type)
