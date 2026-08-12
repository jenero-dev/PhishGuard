"""
train.py
========
PhishGuard - Phishing Website Detection System
Component 2: ML Training Pipeline

Orchestrates the full training workflow:
    1. Load preprocessed data from Component 1
    2. Train all 8 ML models
    3. Stratified cross-validation
    4. Hyperparameter tuning (optional)
    5. Validation set evaluation
    6. External holdout evaluation (optional)
    7. Generate all evaluation artefacts
    8. Save best model for Component 3 (API)

Usage
-----
    python train.py
    python train.py --no-tune
    python train.py --no-tune --no-holdout
    python train.py --data-dir data/processed --model-dir models

Author  : MSc Cybersecurity Project - University of the West of Scotland
Project : PhishGuard
"""

import os
import sys
import time
import json
import logging
import argparse
import warnings
from datetime import datetime

import joblib
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ── Argument parser ───────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='PhishGuard — Component 2: ML Training Pipeline'
    )
    parser.add_argument(
        '--data-dir', type=str, default='data/processed',
        help='Directory containing preprocessed data (default: data/processed)'
    )
    parser.add_argument(
        '--model-dir', type=str, default='models',
        help='Directory to save trained models (default: models)'
    )
    parser.add_argument(
        '--results-dir', type=str, default='results',
        help='Directory to save results and plots (default: results)'
    )
    parser.add_argument(
        '--holdout-dir', type=str, default='data/holdout',
        help='Directory containing external holdout data (default: data/holdout)'
    )
    parser.add_argument(
        '--cv-folds', type=int, default=10,
        help='Number of cross-validation folds (default: 10)'
    )
    parser.add_argument(
        '--no-tune', action='store_true',
        help='Skip hyperparameter tuning'
    )
    parser.add_argument(
        '--no-holdout', action='store_true',
        help='Skip external holdout evaluation'
    )
    return parser.parse_args()


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(data_dir: str) -> dict:
    """Load all preprocessed data splits from Component 1."""
    print('\n  Loading preprocessed data...')

    required = {
        'X_train': 'X_train.csv',
        'X_val'  : 'X_val.csv',
        'X_test' : 'X_test.csv',
        'y_train': 'y_train.csv',
        'y_val'  : 'y_val.csv',
        'y_test' : 'y_test.csv',
    }

    missing = [
        name for name, fname in required.items()
        if not os.path.exists(os.path.join(data_dir, fname))
    ]
    if missing:
        print(f'\n  [ERROR] Missing files in {data_dir}: {missing}')
        print('  Run Component 1 first:')
        print('    python main.py --mode csv --csv-path collected_urls.csv')
        sys.exit(1)

    data = {}
    for key, fname in required.items():
        path = os.path.join(data_dir, fname)
        df   = pd.read_csv(path)
        data[key] = df.squeeze() if key.startswith('y') else df

    # Load feature names
    fn_path = os.path.join(data_dir, 'feature_names.txt')
    if os.path.exists(fn_path):
        with open(fn_path) as f:
            data['feature_names'] = [l.strip() for l in f if l.strip()]
    else:
        data['feature_names'] = list(data['X_train'].columns)

    # Load scaler
    scaler_path = os.path.join(data_dir, 'scaler.pkl')
    if os.path.exists(scaler_path):
        data['scaler'] = joblib.load(scaler_path)
    else:
        data['scaler'] = None
        logger.warning('scaler.pkl not found — holdout evaluation will be skipped')

    # Load leakage report
    lr_path = os.path.join(data_dir, 'leakage_report.json')
    if os.path.exists(lr_path):
        with open(lr_path) as f:
            data['leakage_report'] = json.load(f)
    else:
        data['leakage_report'] = {}

    print(f'  X_train : {data["X_train"].shape}')
    print(f'  X_val   : {data["X_val"].shape}')
    print(f'  X_test  : {data["X_test"].shape}')
    print(f'  y_train : {data["y_train"].shape} | '
          f'dist: {dict(data["y_train"].value_counts())}')
    print(f'  y_val   : {data["y_val"].shape}   | '
          f'dist: {dict(data["y_val"].value_counts())}')
    print(f'  y_test  : {data["y_test"].shape}  | '
          f'dist: {dict(data["y_test"].value_counts())}')
    print(f'  Features: {len(data["feature_names"])}')

    return data


# ── Leakage report display ────────────────────────────────────────────────────

def display_leakage_report(report: dict) -> None:
    """Print the retrieval feature leakage report."""
    if not report:
        return
    print('\n  Retrieval feature leakage report:')
    print(f'  {"Feature":<35} {"Phish":>6} {"Legit":>6} {"Diff":>6}  Flag')
    print('  ' + '-' * 65)
    for feat, info in report.items():
        flag = '⚠ ' + info['flag'] if info['flag'] != 'OK' else '  OK'
        print(f'  {feat:<35} {info["phishing_mean"]:>6.3f} '
              f'{info["legit_mean"]:>6.3f} {info["abs_diff"]:>6.3f}  {flag}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print('=' * 65)
    print('  PHISHGUARD — COMPONENT 2: ML TRAINING PIPELINE')
    print(f'  Started : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 65)

    t_start = time.time()

    # Create output directories
    os.makedirs(args.model_dir,   exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # ── Step 1: Load data ─────────────────────────────────────────────────────
    print('\n' + '=' * 65)
    print('  STEP 1: LOADING DATA')
    print('=' * 65)

    data = load_data(args.data_dir)

    X_train = data['X_train']
    X_val   = data['X_val']
    X_test  = data['X_test']
    y_train = data['y_train']
    y_val   = data['y_val']
    y_test  = data['y_test']
    feature_names  = data['feature_names']
    scaler         = data['scaler']
    leakage_report = data['leakage_report']

    # Display leakage report
    display_leakage_report(leakage_report)

    # ── Step 2: Train all models ──────────────────────────────────────────────
    print('\n' + '=' * 65)
    print('  STEP 2: TRAINING ALL MODELS')
    print('=' * 65)

    from model_trainer import ModelTrainer
    trainer = ModelTrainer()
    trainer.train_all(X_train, y_train, X_test, y_test)

    # ── Step 3: Cross-validation ──────────────────────────────────────────────
    print('\n' + '=' * 65)
    print(f'  STEP 3: CROSS-VALIDATION ({args.cv_folds}-fold)')
    print('=' * 65)

    trainer.cross_validate_all(X_train, y_train, cv=args.cv_folds)

    # ── Step 4: Hyperparameter tuning ─────────────────────────────────────────
    if not args.no_tune:
        print('\n' + '=' * 65)
        print('  STEP 4: HYPERPARAMETER TUNING')
        print('=' * 65)
        trainer.hyperparameter_tune(X_train, y_train)
    else:
        print('\n  [STEP 4] Hyperparameter tuning skipped (--no-tune)')

    # ── Step 5: Validation set evaluation ────────────────────────────────────
    print('\n' + '=' * 65)
    print('  STEP 5: VALIDATION SET EVALUATION')
    print('=' * 65)

    from model_evaluator import ModelEvaluator
    evaluator = ModelEvaluator(output_dir=args.results_dir)
    evaluator._evaluate_on_validation(trainer, X_val, y_val)

    # ── Step 6: Save models ───────────────────────────────────────────────────
    print('\n' + '=' * 65)
    print('  STEP 6: SAVING MODELS')
    print('=' * 65)

    trainer.save_best_model(os.path.join(args.model_dir, 'best_model.pkl'))
    trainer.save_all_models(args.model_dir)

    # Save feature names alongside models (for API use)
    fn_out = os.path.join(args.model_dir, 'feature_names.txt')
    with open(fn_out, 'w') as f:
        f.write('\n'.join(feature_names))
    print(f'  Feature names saved → {fn_out}')

    # Save scaler alongside models
    if scaler is not None:
        scaler_out = os.path.join(args.model_dir, 'scaler.pkl')
        joblib.dump(scaler, scaler_out)
        print(f'  Scaler saved → {scaler_out}')

    # ── Step 7: Generate evaluation artefacts ─────────────────────────────────
    print('\n' + '=' * 65)
    print('  STEP 7: EVALUATION ARTEFACTS')
    print('=' * 65)

    comparison_df = evaluator.generate_all(
        trainer, X_test, y_test, X_val, y_val
    )

    # ── Step 8: External holdout evaluation ───────────────────────────────────
    if not args.no_holdout and scaler is not None:
        holdout_path = os.path.join(
            args.holdout_dir, 'external_phishing_holdout.csv'
        )
        print('\n' + '=' * 65)
        print('  STEP 8: EXTERNAL HOLDOUT EVALUATION')
        print('=' * 65)
        evaluator.evaluate_external_holdout(
            trainer, holdout_path, scaler, feature_names
        )
    else:
        if args.no_holdout:
            print('\n  [STEP 8] Holdout evaluation skipped (--no-holdout)')
        else:
            print('\n  [STEP 8] Holdout evaluation skipped (scaler not available)')

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start

    print('\n' + '=' * 65)
    print('  TRAINING PIPELINE COMPLETE')
    print(f'  Total time : {elapsed:.1f}s')
    print('=' * 65)

    print('\n  Model Comparison (sorted by Test F1):')
    print(comparison_df.to_string(index=False))

    print(f'\n  Best Model : {trainer.best_model_name}')
    best = trainer.results[trainer.best_model_name]
    print(f'    Test F1  : {best["f1"]:.4f}')
    print(f'    Test AUC : {best["roc_auc"]:.4f}')
    print(f'    FPR      : {best["fpr"]:.4f}')
    print(f'    FNR      : {best["fnr"]:.4f}')
    if 'val_f1' in best:
        print(f'    Val F1   : {best["val_f1"]:.4f}')

    print(f'\n  Output:')
    print(f'    Models  → {args.model_dir}/')
    print(f'    Results → {args.results_dir}/')
    print(f'    Plots   → {args.results_dir}/plots/')
    print(f'    Reports → {args.results_dir}/reports/')
    print(f'\n  → Best model: {args.model_dir}/best_model.pkl')
    print(f'  → Next step : python app.py')


if __name__ == '__main__':
    main()
