"""
Evaluate the model using 5-fold cross-validation and save the out-of-fold predictions.
"""
import pandas as pd
import numpy as np
import joblib
from scipy.sparse import hstack
from scipy.stats import spearmanr
from sklearn.utils import resample

from src import config

def evaluate():
    """
    Train the model using 5-fold cross-validation and save out-of-fold predictions.
    """
    # Load the full training data
    df = pd.read_csv(config.DATA_FILE)

    # Load the pre-fitted transformers
    vectorizer_vh = joblib.load(config.ARTEFACTS_DIR / 'vectorizer_vh.joblib')
    vectorizer_vl = joblib.load(config.ARTEFACTS_DIR / 'vectorizer_vl.joblib')
    encoder_ohe = joblib.load(config.ARTEFACTS_DIR / 'encoder_ohe.joblib')

    # Initialize a dataframe to store all out-of-fold predictions
    oof_preds_df = pd.DataFrame({
        'antibody_name': df['antibody_name'],
        config.FOLD_COLUMN: df[config.FOLD_COLUMN]
    })

    # --- Loop through each target property ---
    for target in config.TARGET_PROPERTIES:
        print(f"--- Evaluating for target: {target} ---")

        # Drop rows where the target is missing
        df_target = df.dropna(subset=[target]).copy()
        
        cv_spearman_scores = []
        fold_predictions = []

        # --- Outer CV Loop ---
        for fold_i in config.FOLDS:
            print(f"  Training on folds other than {fold_i}...")

            # Split data into training and testing sets for the current fold
            train_mask = (df_target[config.FOLD_COLUMN] != fold_i)
            test_mask = (df_target[config.FOLD_COLUMN] == fold_i)
            
            df_train = df_target[train_mask]
            df_test = df_target[test_mask]
            
            y_train = df_train[target]
            y_test = df_test[target]

            # Transform features using pre-fitted transformers
            X_train_vh = vectorizer_vh.transform(df_train[config.VH_SEQUENCE_COL])
            X_train_vl = vectorizer_vl.transform(df_train[config.VL_SEQUENCE_COL])
            X_train_ohe = encoder_ohe.transform(df_train[[config.HC_SUBTYPE_COL]])
            X_train = hstack([X_train_vh, X_train_vl, X_train_ohe])

            X_test_vh = vectorizer_vh.transform(df_test[config.VH_SEQUENCE_COL])
            X_test_vl = vectorizer_vl.transform(df_test[config.VL_SEQUENCE_COL])
            X_test_ohe = encoder_ohe.transform(df_test[[config.HC_SUBTYPE_COL]])
            X_test = hstack([X_test_vh, X_test_vl, X_test_ohe])

            # Load the pre-trained model ensemble for this fold
            model_path = config.ARTEFACTS_DIR / f'models_{target}.joblib'
            all_fold_ensembles = joblib.load(model_path)
            current_ensemble_models = all_fold_ensembles[fold_i]

            # Evaluate on the test fold
            fold_preds_all_models = [model.predict(X_test) for model in current_ensemble_models]
            y_pred_fold_avg = np.mean(fold_preds_all_models, axis=0)

            # Store predictions along with antibody names for later merging
            fold_preds_df = pd.DataFrame({
                'antibody_name': df_test['antibody_name'],
                target: y_pred_fold_avg
            })
            fold_predictions.append(fold_preds_df)

            # Calculate and store Spearman correlation
            spearman_corr = spearmanr(y_test, y_pred_fold_avg)[0]
            cv_spearman_scores.append(spearman_corr)
            print(f"    Fold {fold_i} Spearman Correlation: {spearman_corr:.4f}")

        # --- After all folds for the current target ---
        # Report average CV score
        avg_spearman = np.mean(cv_spearman_scores)
        print(f"  Average Spearman for {target}: {avg_spearman:.4f}\n")

        # Merge the predictions for this target into the main OOF dataframe
        target_oof_preds = pd.concat(fold_predictions)
        oof_preds_df = oof_preds_df.merge(target_oof_preds, on='antibody_name', how='left')

    # Save the complete OOF predictions dataframe
    output_path = config.RESULTS_DIR / 'gdp_a1_cv_predictions_raw.csv'
    oof_preds_df.to_csv(output_path, index=False)
    print(f"Out-of-fold predictions saved to {output_path}")


if __name__ == '__main__':
    evaluate()
