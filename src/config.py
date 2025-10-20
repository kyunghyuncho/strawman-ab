"""
This file contains all the global variables and configuration parameters for the project.
"""

from pathlib import Path

# --- File Paths ---
# Using pathlib for robust path handling. Assumes a standard project structure
# where this config file is in `src/` and data/artefacts are at the root.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent
ARTEFACTS_DIR = BASE_DIR.parent / "artefacts"
RESULTS_DIR = BASE_DIR.parent / "results"

# Input data file
DATA_FILE = DATA_DIR / "GDPa1_v1.2_20250814.csv"

# --- Data Columns ---
VH_SEQUENCE_COL = 'vh_protein_sequence'
VL_SEQUENCE_COL = 'vl_protein_sequence'
HC_SUBTYPE_COL = 'hc_subtype'
FOLD_COLUMN = 'hierarchical_cluster_IgG_isotype_stratified_fold'

# --- Target Properties ---
TARGET_PROPERTIES = ['Titer', 'HIC', 'PR_CHO', 'Tm2', 'AC-SINS_pH7.4']
# Targets to be log-transformed (log1p)
LOG_TRANSFORM_TARGETS = [] # ['Titer', 'HIC', 'PR_CHO', 'Tm2']

# --- Cross-Validation ---
FOLDS = [0, 1, 2, 3, 4]

# --- Feature Engineering Hyperparameters ---
# The maximum N for N-grams (1 to N_GRAM_MAX will be used)
N_GRAM_MAX = 32
# Total vocabulary size (split between VH and VL)
K_VOCAB = 50000

# --- Model & Ensemble Hyperparameters ---
# Number of bootstrap models to train in each ensemble
K_BOOTSTRAP = 5
# Random state for reproducibility in bootstrap sampling and model training
RANDOM_STATE = 42

# --- TF-IDF Parameters ---
# These will be passed to the TfidfVectorizer
TFIDF_PARAMS = {
    'analyzer': 'char',
    'ngram_range': (1, N_GRAM_MAX),
    'use_idf': True,
    'smooth_idf': True,
}

# --- Model Selection ---
# Choose between 'ridge' and 'gbr'
MODEL_TYPE = 'ridge' 

# --- Model Parameters ---
# Parameters for the sparse linear regressor (Ridge)
RIDGE_MODEL_PARAMS = {
    'alpha': 1e-8,  # Regularization strength
    'fit_intercept': True,
    'tol': 1e-3,
    'solver': 'sparse_cg'
}

# Parameters for Gradient Boosting Regressor
GBR_MODEL_PARAMS = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 3,
    'random_state': RANDOM_STATE,
    'verbose': 1
}

# A dictionary to hold all model parameters
MODEL_PARAMS = {
    'ridge': RIDGE_MODEL_PARAMS,
    'gbr': GBR_MODEL_PARAMS
}
