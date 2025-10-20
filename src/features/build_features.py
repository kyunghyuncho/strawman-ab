"""
This script handles the global preprocessing and feature definition phase.
It loads the raw data, fits the preprocessing transformers (TF-IDF vectorizers for
VH/VL sequences and a One-Hot Encoder for the subtype), and saves these fitted
transformers to disk for later use in training and prediction.

It can optionally consume tuned hyperparameters from artefacts/best_params.json
or via CLI flags to override ngram_max and total vocabulary size.
"""

import argparse
import json
import pandas as pd
import joblib
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# It's good practice to add the project's 'src' directory to the path
# to ensure modules are found correctly.
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src import config

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def get_feature_pipelines(ngram_range=(1, config.N_GRAM_MAX), vocab_size=config.K_VOCAB):
    """
    Create TF-IDF vectorizers for VH and VL sequences and a OneHotEncoder for subtype.

    Parameters:
    - ngram_range: tuple[int, int] inclusive ngram range for character analyzer
    - vocab_size: int total vocabulary cap (split equally between VH and VL)

    Returns:
    - vectorizer_vh: TfidfVectorizer
    - vectorizer_vl: TfidfVectorizer
    - encoder_ohe: OneHotEncoder
    """
    max_features_each = max(1, vocab_size // 2)
    tfidf_base = dict(config.TFIDF_PARAMS)
    # Override ngram_range per trial
    tfidf_base["ngram_range"] = ngram_range

    vectorizer_vh = TfidfVectorizer(max_features=max_features_each, **tfidf_base)
    vectorizer_vl = TfidfVectorizer(max_features=max_features_each, **tfidf_base)
    encoder_ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    return vectorizer_vh, vectorizer_vl, encoder_ohe


def get_preprocessor(vectorizer_vh, vectorizer_vl, encoder_ohe):
    """
    Build a ColumnTransformer that applies the provided transformers to the
    correct dataframe columns per config.

    Note: The estimators will be cloned by sklearn when fit inside a Pipeline.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("vh_tfidf", vectorizer_vh, config.VH_SEQUENCE_COL),
            ("vl_tfidf", vectorizer_vl, config.VL_SEQUENCE_COL),
            ("hc_ohe", encoder_ohe, [config.HC_SUBTYPE_COL]),
        ],
        sparse_threshold=0.3,
        remainder='drop'
    )
    return preprocessor

def main():
    """
    Main function to execute the feature building process.
    """
    parser = argparse.ArgumentParser(description="Fit and persist feature transformers")
    parser.add_argument("--ngram-max", type=int, default=None, help="Override max n-gram size (upper bound of ngram_range)")
    parser.add_argument("--vocab-size", type=int, default=None, help="Override total vocabulary size (split VH/VL)")
    parser.add_argument("--use-best", action="store_true", help="Use artefacts/best_params.json if available")
    parser.add_argument("--target", type=str, default=None, help="Build per-target vectorizers: pass a target name or 'all'. Omit to build global vectorizers.")
    args = parser.parse_args()

    logging.info("Starting feature building process...")

    # --- 1. Load Data ---
    logging.info(f"Loading data from: {config.DATA_FILE}")
    try:
        df = pd.read_csv(config.DATA_FILE)
        logging.info(f"Data loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        logging.error(f"Data file not found at {config.DATA_FILE}. Exiting.")
        return

    # --- 2. Handle Missing Sequences ---
    # Drop rows where either of the protein sequences is missing.
    initial_rows = len(df)
    df.dropna(subset=[config.VH_SEQUENCE_COL, config.VL_SEQUENCE_COL], inplace=True)
    rows_dropped = initial_rows - len(df)
    if rows_dropped > 0:
        logging.warning(f"Dropped {rows_dropped} rows due to missing sequence data.")
    
    # Also fill any missing subtype values with a placeholder, e.g., 'Unknown'
    df[config.HC_SUBTYPE_COL] = df[config.HC_SUBTYPE_COL].fillna('Unknown')
    logging.info("Handled missing values.")

    # --- 3. Define & Fit Preprocessing Transformers ---

    # If per-target vectorizers requested, handle that branch and return
    if args.target:
        target_arg = args.target.strip()
        # Load tuned params if requested
        best_params_path = config.ARTEFACTS_DIR / "best_params.json"
        best = None
        if args.use_best and best_params_path.exists():
            try:
                with open(best_params_path, "r") as f:
                    best = json.load(f)
            except Exception as e:
                logging.warning(f"Failed to read best params: {e}")

        if target_arg.lower() == 'all':
            targets = list(config.TARGET_PROPERTIES)
        else:
            if target_arg not in config.TARGET_PROPERTIES:
                logging.error(f"Unknown target '{target_arg}'. Valid: {config.TARGET_PROPERTIES}")
                return
            targets = [target_arg]

        config.ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)

        for t in targets:
            # Resolve ngram_max and vocab_size with precedence: CLI > per_target > global > config
            t_ngram = args.ngram_max
            t_vocab = args.vocab_size
            if best:
                try:
                    per_target = (best.get('per_target') or {}).get(t) or {}
                    global_best = best.get('global') or {}
                    if t_ngram is None:
                        t_ngram = int(per_target.get('ngram_max') or 0) or int(global_best.get('ngram_max') or 0) or None
                    if t_vocab is None:
                        t_vocab = int(per_target.get('vocab_size') or 0) or int(global_best.get('vocab_size') or 0) or None
                except Exception:
                    pass
            t_ngram = t_ngram if t_ngram is not None else config.N_GRAM_MAX
            t_vocab = t_vocab if t_vocab is not None else config.K_VOCAB

            logging.info(f"Building per-target vectorizers for {t} with ngram_max={t_ngram}, vocab_size={t_vocab}")

            # Fit vectorizers on all sequences (consistent with global behavior)
            vectorizer_vh = TfidfVectorizer(
                max_features=t_vocab // 2,
                **{**config.TFIDF_PARAMS, "ngram_range": (1, t_ngram)}
            )
            vectorizer_vh.fit(df[config.VH_SEQUENCE_COL])

            vectorizer_vl = TfidfVectorizer(
                max_features=t_vocab // 2,
                **{**config.TFIDF_PARAMS, "ngram_range": (1, t_ngram)}
            )
            vectorizer_vl.fit(df[config.VL_SEQUENCE_COL])

            encoder_ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
            encoder_ohe.fit(df[[config.HC_SUBTYPE_COL]])

            # Save with target suffix
            vh_path = config.ARTEFACTS_DIR / f"vectorizer_vh_{t}.joblib"
            vl_path = config.ARTEFACTS_DIR / f"vectorizer_vl_{t}.joblib"
            ohe_path = config.ARTEFACTS_DIR / f"encoder_ohe_{t}.joblib"
            joblib.dump(vectorizer_vh, vh_path)
            joblib.dump(vectorizer_vl, vl_path)
            joblib.dump(encoder_ohe, ohe_path)
            logging.info(f"Saved per-target transformers for {t} -> {vh_path.name}, {vl_path.name}, {ohe_path.name}")

        logging.info("Per-target feature building process completed successfully.")
        return

    # Determine hyperparameters
    tuned_ngram_max = None
    tuned_vocab = None
    best_params_path = config.ARTEFACTS_DIR / "best_params.json"
    if args.use_best and best_params_path.exists():
        try:
            with open(best_params_path, "r") as f:
                best = json.load(f)
            tuned_ngram_max = int(best.get("ngram_max") or 0) or None
            tuned_vocab = int(best.get("vocab_size") or 0) or None
            logging.info(f"Using tuned hyperparameters from {best_params_path}: ngram_max={tuned_ngram_max}, vocab_size={tuned_vocab}")
        except Exception as e:
            logging.warning(f"Failed to read best params: {e}. Falling back to defaults.")

    # CLI overrides take precedence
    if args.ngram_max is not None:
        tuned_ngram_max = args.ngram_max
    if args.vocab_size is not None:
        tuned_vocab = args.vocab_size

    ngram_max = tuned_ngram_max if tuned_ngram_max is not None else config.N_GRAM_MAX
    vocab_total = tuned_vocab if tuned_vocab is not None else config.K_VOCAB
    logging.info(f"Feature settings -> ngram_range=(1,{ngram_max}), total_vocab={vocab_total} (each={max(1, vocab_total//2)})")

    # VH N-gram Vectorizer
    logging.info("Fitting VH N-gram TfidfVectorizer...")
    vectorizer_vh = TfidfVectorizer(
        max_features=vocab_total // 2,
        **{**config.TFIDF_PARAMS, "ngram_range": (1, ngram_max)}
    )
    vectorizer_vh.fit(df[config.VH_SEQUENCE_COL])
    logging.info(f"VH vectorizer fitted. Vocabulary size: {len(vectorizer_vh.vocabulary_)}")

    # VL N-gram Vectorizer
    logging.info("Fitting VL N-gram TfidfVectorizer...")
    vectorizer_vl = TfidfVectorizer(
        max_features=vocab_total // 2,
        **{**config.TFIDF_PARAMS, "ngram_range": (1, ngram_max)}
    )
    vectorizer_vl.fit(df[config.VL_SEQUENCE_COL])
    logging.info(f"VL vectorizer fitted. Vocabulary size: {len(vectorizer_vl.vocabulary_)}")

    # Subtype One-Hot Encoder
    logging.info("Fitting Subtype OneHotEncoder...")
    encoder_ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    # The encoder expects a 2D array, so we reshape the series.
    encoder_ohe.fit(df[[config.HC_SUBTYPE_COL]])
    logging.info(f"Subtype encoder fitted. Categories: {encoder_ohe.categories_}")

    # --- 4. Persistence ---
    # Ensure the artefacts directory exists
    config.ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
    logging.info(f"Artefacts directory ensured at: {config.ARTEFACTS_DIR}")

    # Save the three fitted transformers
    vh_vectorizer_path = config.ARTEFACTS_DIR / "vectorizer_vh.joblib"
    vl_vectorizer_path = config.ARTEFACTS_DIR / "vectorizer_vl.joblib"
    ohe_encoder_path = config.ARTEFACTS_DIR / "encoder_ohe.joblib"

    joblib.dump(vectorizer_vh, vh_vectorizer_path)
    logging.info(f"VH vectorizer saved to {vh_vectorizer_path}")
    
    joblib.dump(vectorizer_vl, vl_vectorizer_path)
    logging.info(f"VL vectorizer saved to {vl_vectorizer_path}")

    joblib.dump(encoder_ohe, ohe_encoder_path)
    logging.info(f"Subtype encoder saved to {ohe_encoder_path}")

    logging.info("Feature building process completed successfully.")

if __name__ == '__main__':
    main()
