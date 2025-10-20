"""
This script handles the global preprocessing and feature definition phase.
It loads the raw data, fits the preprocessing transformers (TF-IDF vectorizers for
VH/VL sequences and a One-Hot Encoder for the subtype), and saves these fitted
transformers to disk for later use in training and prediction.
"""

import pandas as pd
import joblib
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

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

def main():
    """
    Main function to execute the feature building process.
    """
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
    
    # VH N-gram Vectorizer
    logging.info("Fitting VH N-gram TfidfVectorizer...")
    vectorizer_vh = TfidfVectorizer(
        max_features=config.K_VOCAB // 2,
        **config.TFIDF_PARAMS
    )
    vectorizer_vh.fit(df[config.VH_SEQUENCE_COL])
    logging.info(f"VH vectorizer fitted. Vocabulary size: {len(vectorizer_vh.vocabulary_)}")

    # VL N-gram Vectorizer
    logging.info("Fitting VL N-gram TfidfVectorizer...")
    vectorizer_vl = TfidfVectorizer(
        max_features=config.K_VOCAB // 2,
        **config.TFIDF_PARAMS
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
