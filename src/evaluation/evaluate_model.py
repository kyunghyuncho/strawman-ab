"""
Evaluate the model using 5-fold cross-validation and save the out-of-fold predictions.
Supports choosing model type (ridge/gbr) and expects model artefacts saved per
target with suffix models_{target}_{model_type}.joblib.
"""
import pandas as pd
import numpy as np
import joblib
from scipy.sparse import hstack
from scipy.stats import spearmanr
from sklearn.utils import resample
import sys
import os
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


from src import config

def evaluate(model_type: str):
    """
    Train the model using 5-fold cross-validation and save out-of-fold predictions.
    """
    # Load the full training data
    df = pd.read_csv(config.DATA_FILE)

    # Load global pre-fitted transformers (fallback). Per-target overrides are loaded in-loop
    vectorizer_vh_global = joblib.load(config.ARTEFACTS_DIR / 'vectorizer_vh.joblib')
    vectorizer_vl_global = joblib.load(config.ARTEFACTS_DIR / 'vectorizer_vl.joblib')
    encoder_ohe_global = joblib.load(config.ARTEFACTS_DIR / 'encoder_ohe.joblib')

    # Initialize a dataframe to store all out-of-fold predictions
    oof_preds_df = pd.DataFrame({
        'antibody_name': df['antibody_name'],
        config.FOLD_COLUMN: df[config.FOLD_COLUMN]
    })

    # --- Loop through each target property ---
    for target in config.TARGET_PROPERTIES:
        print(f"--- Evaluating for target: {target} ---")

        cv_spearman_scores = []
        fold_predictions = []

        # Choose transformers for this target (per-target if available, otherwise global)
        vectorizer_vh = vectorizer_vh_global
        vectorizer_vl = vectorizer_vl_global
        encoder_ohe = encoder_ohe_global
        try:
            vh_path = config.ARTEFACTS_DIR / f'vectorizer_vh_{target}.joblib'
            vl_path = config.ARTEFACTS_DIR / f'vectorizer_vl_{target}.joblib'
            ohe_path = config.ARTEFACTS_DIR / f'encoder_ohe_{target}.joblib'
            if vh_path.exists() and vl_path.exists() and ohe_path.exists():
                vectorizer_vh = joblib.load(vh_path)
                vectorizer_vl = joblib.load(vl_path)
                encoder_ohe = joblib.load(ohe_path)
                print(f"  Using per-target transformers for {target}.")
        except Exception as e:
            print(f"  Warning: falling back to global transformers for {target}: {e}")

        # --- Outer CV Loop ---
        for fold_i in config.FOLDS:
            print(f"  Training on folds other than {fold_i}...")

            # --- Data Splitting ---
            # Use all data for testing to ensure predictions for all antibodies
            df_test = df[df[config.FOLD_COLUMN] == fold_i]
            
            # For training, use data from other folds AND where the target is not missing
            train_mask = (df[config.FOLD_COLUMN] != fold_i) & (df[target].notna())
            df_train = df[train_mask]

            # Skip fold if no training data is available
            if df_train.empty:
                print(f"    Skipping fold {fold_i} for target {target} due to no training samples.")
                # Add empty predictions for this fold to maintain structure
                fold_preds_df = pd.DataFrame({'antibody_name': df_test['antibody_name'], target: np.nan})
                fold_predictions.append(fold_preds_df)
                continue

            y_train = df_train[target]
            y_test = df_test[target] # This will contain NaNs

            # Transform features using pre-fitted transformers
            X_train_vh = vectorizer_vh.transform(df_train[config.VH_SEQUENCE_COL])
            X_train_vl = vectorizer_vl.transform(df_train[config.VL_SEQUENCE_COL])
            X_train_ohe = encoder_ohe.transform(df_train[[config.HC_SUBTYPE_COL]])
            X_train = hstack([X_train_vh, X_train_vl, X_train_ohe])

            # If test set is empty, continue
            if df_test.empty:
                print(f"    Skipping fold {fold_i} for target {target} due to no test samples.")
                continue

            X_test_vh = vectorizer_vh.transform(df_test[config.VH_SEQUENCE_COL])
            X_test_vl = vectorizer_vl.transform(df_test[config.VL_SEQUENCE_COL])
            X_test_ohe = encoder_ohe.transform(df_test[[config.HC_SUBTYPE_COL]])
            X_test = hstack([X_test_vh, X_test_vl, X_test_ohe])

            # Load the pre-trained model ensemble for this fold
            model_path = config.ARTEFACTS_DIR / f'models_{target}_{model_type}.joblib'
            all_fold_ensembles = joblib.load(model_path)
            current_ensemble_models = all_fold_ensembles[fold_i]

            # Evaluate on the test fold
            fold_preds_all_models = [model.predict(X_test) for model in current_ensemble_models]
            y_pred_fold_avg = np.mean(fold_preds_all_models, axis=0)

            # Inverse transform predictions if the target was log-transformed
            if target in config.LOG_TRANSFORM_TARGETS:
                y_pred_fold_avg = np.expm1(y_pred_fold_avg)

            # Store predictions along with antibody names for later merging
            fold_preds_df = pd.DataFrame({
                'antibody_name': df_test['antibody_name'],
                target: y_pred_fold_avg
            })
            fold_predictions.append(fold_preds_df)

            # --- Calculate Spearman correlation only on non-missing targets ---
            # Create a temporary dataframe for calculating correlation
            temp_eval_df = pd.DataFrame({'y_true': y_test, 'y_pred': y_pred_fold_avg})
            
            # Drop rows where the true value is missing
            temp_eval_df.dropna(inplace=True)

            # Calculate and store Spearman correlation if there are samples left
            if not temp_eval_df.empty:
                spearman_corr = spearmanr(temp_eval_df['y_true'], temp_eval_df['y_pred'])[0]
                cv_spearman_scores.append(spearman_corr)
                print(f"    Fold {fold_i} Spearman Correlation: {spearman_corr:.4f}")
            else:
                print(f"    Fold {fold_i} Spearman Correlation: N/A (no true values)")

        # --- After all folds for the current target ---
        # Report average CV score
        if cv_spearman_scores:
            avg_spearman = np.mean(cv_spearman_scores)
            print(f"  Average Spearman for {target}: {avg_spearman:.4f}\n")
        else:
            print(f"  Average Spearman for {target}: N/A (no scores calculated)\n")

        # Merge the predictions for this target into the main OOF dataframe
        target_oof_preds = pd.concat(fold_predictions)
        oof_preds_df = oof_preds_df.merge(target_oof_preds, on='antibody_name', how='left')

    # Save the complete OOF predictions dataframe
    output_path = config.RESULTS_DIR / 'gdp_a1_cv_predictions_raw.csv'
    oof_preds_df.to_csv(output_path, index=False)
    print(f"Out-of-fold predictions saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate models with 5-fold CV and save OOF predictions')
    parser.add_argument('--model-type', type=str, default=config.MODEL_TYPE, choices=['ridge', 'gbr'])
    args = parser.parse_args()
    evaluate(model_type=args.model_type)
