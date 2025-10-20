# N-gram Regression Model for Antibody Properties

This repository contains the code used for the Ginkgo Antibody Developability (AbDev) Benchmark. You can find the public leaderboard and submission interface here:

- Ginkgo AbDev Benchmark (Hugging Face Space): https://huggingface.co/spaces/ginkgo-datapoints/abdev-leaderboard

The pipeline below reproduces our baseline and generates leaderboard-ready submission files.

## 1. Project Goal

The primary objective is to build a predictive model for several key antibody properties using their heavy and light chain amino acid (AA) sequences. The model is a sparse linear regression ensemble trained on N-gram features, with performance evaluated using a 5-fold cross-validation scheme based on predefined folds.

---

## 2. Core Methodology

The model combines N-gram sequence features with one-hot encoded antibody subtypes.

1.  **Feature Engineering**:
    *   **N-grams**: Separate vocabularies are generated for heavy (`vh_protein_sequence`) and light (`vl_protein_sequence`) chains. Each sequence is tokenized into overlapping character N-grams (e.g., "AVG", "VGL").
    *   **TF-IDF**: A `TfidfVectorizer` is used for each chain to build a vocabulary and transform sequences into sparse vectors of TF-IDF weighted N-gram counts.
    *   **Subtype**: The `hc_subtype` categorical feature is one-hot encoded.
    *   **Final Vector**: The final feature vector for an antibody is the horizontal concatenation of its VH TF-IDF vector, VL TF-IDF vector, and subtype OHE vector.

2.  **Target Transformation**:
    *   To handle skewed distributions, a `log1p` transformation is applied to `Titer`, `HIC`, `PR_CHO`, and `Tm2` before training.
    *   The `AC-SINS_pH7.4` property, which contains negative values, is not transformed.
    *   Predictions are converted back to their original scale using `expm1`.

3.  **Model**:
    *   A **sparse linear regression** model (`Ridge`) is used, which is well-suited for high-dimensional, sparse feature sets.

4.  **Ensemble**:
    *   For each training fold, an ensemble of `Ridge` regressors is trained on bootstrap-resampled versions of the training data. The final prediction is the average of the predictions from all models in the ensemble.

5.  **Evaluation**:
    *   The model is evaluated using the 5 predefined folds from the `hierarchical_cluster_IgG_isotype_stratified_fold` column.
    *   The primary metric is the **Spearman Rank Correlation** between the ensemble predictions and the true target values, averaged across the 5 folds.

---

## 3. Project Structure

```
.
├── GDPa1_v1.2_20250814.csv
├── heldout-set-sequences.csv
├── implementation.md
├── overall_plan.md
├── README.md
├── requirements.txt
├── artefacts/
│   ├── encoder_ohe.joblib
│   ├── models_AC-SINS_pH7.4.joblib
│   ├── models_HIC.joblib
│   ├── models_PR_CHO.joblib
│   ├── models_Titer.joblib
│   ├── models_Tm2.joblib
│   ├── vectorizer_vh.joblib
│   └── vectorizer_vl.joblib
├── results/
│   ├── gdp_a1_cv_predictions.csv
│   ├── gdp_a1_cv_predictions_raw.csv
│   ├── private_test_predictions.csv
│   └── private_test_predictions_raw.csv
└── src/
    ├── __init__.py
    ├── config.py
    ├── evaluation/
    │   ├── __init__.py
    │   └── evaluate_model.py
    ├── features/
    │   ├── __init__.py
    │   └── build_features.py
    ├── models/
    │   ├── __init__.py
    │   ├── predict_model.py
    │   └── train_model.py
    └── submission/
        ├── __init__.py
        └── create_submission.py
```

---

## 4. How to Run the Pipeline

Follow these steps to reproduce the results.

1.  **Install Dependencies**:
    Install the required packages from `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

2.  **Generate Features**:
    This script fits the TF-IDF vectorizers and the one-hot encoder on the entire dataset and saves them to the `artefacts/` directory.
    ```bash
    python src/features/build_features.py
    ```

3.  **Train Models**:
    This script trains an ensemble of models for each of the five target properties using 5-fold cross-validation. The trained model ensembles are saved to the `artefacts/` directory.
    ```bash
    python src/models/train_model.py
    ```

4.  **Evaluate Models and Generate CV Predictions**:
    This script loads the trained models and generates out-of-fold predictions for the training data (`GDPa1_v1.2_20250814.csv`). It also calculates and prints the Spearman correlation for each target property. The raw predictions are saved.
    ```bash
    python src/evaluation/evaluate_model.py
    ```

5.  **Generate Predictions for the Private Test Set**:
    This script uses the trained models to make predictions on the private test set (`heldout-set-sequences.csv`). The raw predictions are saved.
    ```bash
    python src/models/predict_model.py
    ```

6.  **Create Final Submission Files**:
    This script formats the raw cross-validation and private test predictions into the final CSV files required for submission.
    ```bash
    python src/submission/create_submission.py
    ```

---

## 5. Results

The final, formatted prediction files are saved in the `results/` directory:

*   `gdp_a1_cv_predictions.csv`: Out-of-fold predictions for the training set.
*   `private_test_predictions.csv`: Predictions for the private test set.

These CSVs are formatted to be compatible with the Ginkgo AbDev leaderboard upload flow. Refer to the Space linked above for any updates to the submission format or evaluation process.
