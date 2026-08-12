# PhishGuard: Phishing Website Detection System

PhishGuard is an end-to-end machine-learning project for classifying URLs as legitimate or phishing. It covers URL collection, feature extraction, preprocessing, training and evaluating multiple classifiers, visualisation, SHAP-based explainability, and deployment through a FastAPI prediction service.

> **Champion model:** Multilayer Perceptron (MLP) — **94.65%** test accuracy, **93.75%** F1 score, and **0.9892** ROC-AUC.

## Contents

- [System overview](#system-overview)
- [Repository structure](#repository-structure)
- [Dataset and splits](#dataset-and-splits)
- [Feature schema](#feature-schema)
- [Preprocessing](#preprocessing)
- [Models and evaluation](#models-and-evaluation)
- [Results](#results)
- [Explainability](#explainability)
- [Installation](#installation)
- [Running the pipeline](#running-the-pipeline)
- [Running the API](#running-the-api)
- [API reference](#api-reference)
- [Limitations](#limitations)

## System Architecture
```
┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
│ 1. Data          │   │ 2. Feature       │   │ 3. Preprocessing │
│    Collection    │──▶│    Extraction    │──▶│  (clean, scale,  │
│ PhishTank +      │   │ 28 lexical/host  │   │   balance)       │
│ Tranco/legit     │   │ URL features     │   │                  │
└──────────────────┘   └──────────────────┘   └────────┬─────────┘
                                                        │
        ┌───────────────────────────────────────────────┘
        ▼
┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
│ 4. Model         │   │ 5. Backend API   │   │ 6. Frontend      │
│  Training &      │──▶│    (FastAPI)     │──▶│    Dashboard     │
│  Evaluation      │   │  + SHAP explain  │   │  (React + charts)│
│  8 models, CV    │   │  + SQLite store  │   │  risk gauge, etc.│
└──────────────────┘   └──────────────────┘   └──────────────────┘
```


The API loads the trained model, the fitted scaler, and the saved feature order. At prediction time it extracts available features from the submitted URL, fills any missing trained features with defaults derived from the training data, orders the feature vector exactly as used at training, scales it, and returns the predicted class and confidence.

## Repository Structure

```text
phishing_detection_project/
├── main.py                 # Pipeline orchestration
├── data_collector.py       # URL collection and labelling
├── feature_extractor.py    # URL-to-feature extraction
├── preprocessor.py         # Data cleaning, balancing, scaling and splitting
├── model_trainer.py        # Model creation and hyperparameter tuning
├── model_evaluator.py      # Evaluation metrics and visualisations
├── model_visualizer.py     # Plot generation utilities (if present)
├── shap_analyzer.py        # SHAP analysis utilities (if present)
├── train.py                # Training and evaluation entry point
├── app.py                  # FastAPI application
├── api_test.py             # API smoke-test script
├── frontend/
│   └── index.html           # React dashboard (CDN React + Tailwind + Chart.js)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── requirements.txt        # Python dependencies
├── collected_urls.csv      # Labelled URL dataset
├── data/
│   ├── features/           # Extracted feature data
│   └── processed/          # Processed CSVs, scaler and feature names
├── models/                 # Saved trained models, including best_model.pkl
└── results/                # Metrics, plots and explainability outputs
```

## Dataset and Splits

The dataset contains **8,115 labelled URLs**. Label `0` denotes a legitimate URL and label `1` denotes a phishing URL.

| Partition | Instances | Legitimate | Phishing |
|---|---:|---:|---:|
| Training | 6,262 | 3,131 | 3,131 |
| Validation | 694 | 423 | 271 |
| Test | 1,159 | 651 | 508 |
| **Total** | **8,115** | — | — |

The balanced training partition is obtained through random undersampling of the majority class. The test partition remains suitable for final held-out evaluation.

## Feature Schema

The trained model uses the following **30 features**, in this exact saved order:

| # | Feature | Description |
|---:|---|---|
| 1 | `url_length` | Total URL length. |
| 2 | `digit_ratio` | Proportion of digits in the URL. |
| 3 | `special_char_ratio` | Proportion of special characters. |
| 4 | `url_entropy` | Shannon entropy of the URL text. |
| 5 | `has_ip_address` | Whether the host is an IP address. |
| 6 | `has_at_symbol` | Whether `@` appears in the URL. |
| 7 | `has_non_standard_port` | Whether a non-standard port is used. |
| 8 | `uses_http` | Whether the URL uses HTTP rather than HTTPS. |
| 9 | `domain_age_days` | Domain age in days. |
| 10 | `registration_length_days` | Domain registration duration in days. |
| 11 | `domain_expiry_proximity_days` | Days remaining until domain expiry. |
| 12 | `whois_data_available` | Whether WHOIS data was available. |
| 13 | `has_a_or_aaaa_record` | Presence of an A or AAAA DNS record. |
| 14 | `has_ns_record` | Presence of an NS record. |
| 15 | `has_mx_record` | Presence of an MX record. |
| 16 | `num_name_servers` | Number of name servers. |
| 17 | `num_forms` | Number of forms detected on the page. |
| 18 | `has_password_field` | Presence of a password input field. |
| 19 | `has_identity_field` | Presence of an identity-related input field. |
| 20 | `has_sensitive_input` | Presence of sensitive inputs. |
| 21 | `has_external_form_action` | Whether a form submits to an external domain. |
| 22 | `has_insecure_form_action` | Whether a form action uses HTTP. |
| 23 | `credential_language_present` | Presence of credential-related language. |
| 24 | `brand_domain_mismatch` | Brand/domain mismatch indicator. |
| 25 | `num_iframes` | Number of iframes. |
| 26 | `hidden_or_suspicious_iframe` | Hidden or suspicious iframe indicator. |
| 27 | `page_fetch_success` | Whether webpage retrieval succeeded. |
| 28 | `http_status_category` | Category of the HTTP status response. |
| 29 | `redirect_count` | Number of redirects followed. |
| 30 | `final_domain_changed` | Whether the final domain differs from the submitted domain. |

The authoritative feature list is stored in `data/processed/feature_names.txt`. Model input must preserve this order.

## Preprocessing

`DataPreprocessor` performs the following operations:

1. Removes rows with missing or invalid labels.
2. Replaces `-1` sentinel values with missing values for recognised content features where applicable.
3. Imputes missing numeric values using the median of the respective feature.
4. Removes duplicate feature rows.
5. Caps outliers using bounds of three interquartile ranges below Q1 and above Q3.
6. Balances classes through random undersampling of the majority class.
7. Separates `url` and `label` from model features.
8. Uses a stratified train/test split.
9. Fits `StandardScaler` on training features and applies it to the test features.
10. Saves processed data, `scaler.pkl`, and `feature_names.txt`.

> The implementation uses random undersampling. It does not use SMOTE.

## Models and Evaluation

Eight classifiers are evaluated:

1. **Multilayer Perceptron** (MLP)
2. Support Vector Machine (SVM)
3. Random Forest
4. XGBoost
5. Logistic Regression
6. K-Nearest Neighbours (KNN)
7. Decision Tree
8. Naive Bayes

Evaluation reports accuracy, precision, recall, F1 score, ROC-AUC, false-positive rate, false-negative rate, training time, and cross-validation statistics. Where configured, model tuning is performed using grid search and stratified cross-validation.

## Results

Results below are calculated on the held-out test set of **1,159 instances**.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FNR |
|---|---:|---:|---:|---:|---:|---:|
| **MLP** | **0.9465** | **0.9607** | 0.9154 | **0.9375** | **0.9892** | 0.0846 |
| SVM | 0.9379 | 0.9395 | **0.9173** | 0.9283 | 0.9871 | **0.0827** |
| Random Forest | 0.9301 | 0.9296 | 0.9094 | 0.9194 | 0.9795 | 0.0906 |
| XGBoost | 0.9284 | 0.9363 | 0.8976 | 0.9166 | 0.9847 | 0.1024 |
| Logistic Regression | 0.9215 | 0.9080 | 0.9134 | 0.9107 | 0.9831 | 0.0866 |
| KNN | 0.9154 | 0.9218 | 0.8819 | 0.9014 | 0.9615 | 0.1181 |
| Decision Tree | 0.9051 | 0.8948 | 0.8878 | 0.8913 | 0.9059 | 0.1122 |
| Naive Bayes | 0.8991 | 0.9393 | 0.8228 | 0.8772 | 0.9655 | 0.1772 |

### Champion Model: MLP

The MLP attained the strongest overall test F1 score and accuracy. Its test confusion matrix is:

| Actual class | Predicted legitimate | Predicted phishing |
|---|---:|---:|
| Legitimate | 632 | 19 |
| Phishing | 43 | 465 |

Its ten-fold cross-validation performance was **F1 = 0.9428 ± 0.0108** and **AUC = 0.9865 ± 0.0032**.

Generated visual outputs include:

- `results/plots/model_comparison.png`
- `results/plots/cv_boxplots.png`
- `results/plots/roc_curves.png`
- `results/plots/precision_recall_curves.png`
- `results/plots/confusion_matrices.png`
- `results/plots/feature_importance_xgboost.png`
- `results/plots/feature_importance_random_forest.png`
- `results/plots/feature_importance_decision_tree.png`

## Explainability
1
SHAP (SHapley Additive exPlanations) is used to inspect how features contribute to classifier decisions. Global explanation outputs are stored as bar and summary plots, for example:

- `results/shap/shap_bar_xgboost.png`
- `results/shap/shap_summary_xgboost.png`
- `results/shap/shap_bar_random_forest.png`
- `results/shap/shap_summary_random_forest.png`
- `results/shap/shap_bar_decision_tree.png`
- `results/shap/shap_summary_decision_tree.png`
- `results/shap/shap_bar_svm.png`
- `results/shap/shap_summary_svm.png`

A SHAP bar plot ranks features by mean absolute contribution. A SHAP summary plot additionally shows the direction and distribution of contribution across observations. These explanations are model-specific and should only be interpreted against the exact feature schema used by the corresponding trained model.

## Installation

### Prerequisites

- Python 3.10 or later
- `pip`

```bash
python -m venv venv
```

Activate the virtual environment:

```bash
# macOS/Linux
source venv/bin/activate

# Windows PowerShell
venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Pipeline

Run the preprocessing/data pipeline through the project entry point:

```bash
python main.py
```

Train and evaluate the models:

```bash
python train.py
```

Before starting the API, ensure that these artefacts exist:

```text
models/best_model.pkl
data/processed/scaler.pkl
data/processed/feature_names.txt
```

## Running the API

Start the FastAPI server:

```bash
python app.py
```

Alternatively, run it with Uvicorn:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

The server is available locally at `http://localhost:8000`. FastAPI's interactive documentation is available at `http://localhost:8000/docs` while the server is running.

## API Reference

### `GET /`

Returns service status and the loaded model name.

Example response:

```json
{
  "status": "online",
  "message": "Phishing Detection API is running",
  "model_used": "MLPClassifier"
}
```

### `POST /predict`

Classifies a URL. The request body accepts a single `url` field. A scheme is added automatically when one is omitted.

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"url":"http://example.com"}'
```

Response fields:

| Field | Description |
|---|---|
| `url` | Normalised URL used for extraction. |
| `is_phishing` | `true` where the model predicts label `1`. |
| `confidence` | Highest class probability expressed as a percentage. |
| `probability_score` | Model probability assigned to phishing class `1`. |
| `features_extracted` | Raw features extracted for the submitted URL. |

## Limitations

1. **External dependencies:** Several domain, DNS, WHOIS, and webpage-content features depend on live external services and may be unavailable, delayed, or inconsistent.
2. **Inference fallbacks:** When a trained feature is unavailable at API inference, the service fills it with a default based on the training data. This is a practical fallback, not a substitute for live feature collection.
3. **Dataset dependence:** Reported metrics apply to the data and split used for this project; performance may differ on new, time-separated, or adversarial phishing campaigns.
4. **False negatives:** Even the champion model missed 43 phishing URLs in the test set. The system should complement, not replace, established security controls.
5. **Concept drift:** Phishing techniques change over time, so the model and data should be refreshed and re-evaluated periodically.
6. **Prototype status:** This project is an academic prototype and requires additional operational security measures before production deployment.

## Academic Use

PhishGuard was developed as an MSc phishing-website detection project. The results, documentation and interpretability outputs should be read alongside the dissertation methodology, testing evidence and stated limitations.
