"""
Generate predictions for the private test set using the trained models.
"""
import pandas as pd
import numpy as np
import joblib
from scipy.sparse import hstack
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src import config

def predict():
    """
    Generate and save predictions for the private held-out test set.
    """
    print("--- Generating predictions for the private test set ---")

    # Load the test data
    test_file = config.DATA_DIR / 'heldout-set-sequences.csv'
    df_test = pd.read_csv(test_file)

    # Load the pre-fitted transformers
    vectorizer_vh = joblib.load(config.ARTEFACTS_DIR / 'vectorizer_vh.joblib')
    vectorizer_vl = joblib.load(config.ARTEFACTS_DIR / 'vectorizer_vl.joblib')
    encoder_ohe = joblib.load(config.ARTEFACTS_DIR / 'encoder_ohe.joblib')

    # Transform the test data features
    X_test_vh = vectorizer_vh.transform(df_test[config.VH_SEQUENCE_COL])
    X_test_vl = vectorizer_vl.transform(df_test[config.VL_SEQUENCE_COL])
    X_test_ohe = encoder_ohe.transform(df_test[[config.HC_SUBTYPE_COL]])
    X_test_final = hstack([X_test_vh, X_test_vl, X_test_ohe])

    # Initialize a dataframe to store all predictions
    predictions_df = pd.DataFrame({'antibody_name': df_test['antibody_name']})

    # --- Loop through each target property ---
    for target in config.TARGET_PROPERTIES:
        print(f"  Predicting for target: {target}...")

        # Load the full set of model ensembles for the target
        model_path = config.ARTEFACTS_DIR / f'models_{target}.joblib'
        all_fold_ensembles = joblib.load(model_path)

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
    output_path = config.RESULTS_DIR / 'private_test_predictions_raw.csv'
    predictions_df.to_csv(output_path, index=False)
    print(f"\nTest set predictions saved to {output_path}")


if __name__ == '__main__':
    predict()
