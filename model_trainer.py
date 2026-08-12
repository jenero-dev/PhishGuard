"""
model_trainer.py
================
PhishGuard - Phishing Website Detection System
Component 2: Model Training

Trains 8 ML classifiers and evaluates them under:
    - Standard test-set evaluation
    - 10-fold stratified cross-validation
    - Hyperparameter tuning on the best model

Models
------
    Random Forest, Decision Tree, SVM, Logistic Regression,
    KNN, Naive Bayes, MLP, XGBoost

Author  : MSc Cybersecurity Project - University of the West of Scotland
Project : PhishGuard
"""

import os
import time
import json
import logging

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble        import RandomForestClassifier
from sklearn.tree            import DecisionTreeClassifier
from sklearn.svm             import SVC
from sklearn.linear_model    import LogisticRegression
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.naive_bayes     import GaussianNB
from sklearn.neural_network  import MLPClassifier
from xgboost                 import XGBClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)
from sklearn.model_selection import (
    StratifiedKFold, cross_validate, GridSearchCV,
)

logger = logging.getLogger(__name__)


# ── Hyperparameter grids ──────────────────────────────────────────────────────

PARAM_GRIDS = {
    'Random Forest': {
        'n_estimators' : [100, 200],
        'max_depth'    : [None, 20, 40],
        'min_samples_split': [2, 5],
    },
    'XGBoost': {
        'n_estimators' : [100, 200],
        'max_depth'    : [4, 6, 8],
        'learning_rate': [0.05, 0.1],
    },
    'Logistic Regression': {
        'C'      : [0.1, 1.0, 10.0],
        'solver' : ['lbfgs', 'liblinear'],
    },
    'SVM': {
        'C'     : [0.1, 1.0, 10.0],
        'kernel': ['rbf', 'linear'],
    },
}


# ═════════════════════════════════════════════════════════════════════════════
# ModelTrainer
# ═════════════════════════════════════════════════════════════════════════════

class ModelTrainer:
    """
    Trains, cross-validates, and tunes 8 ML models for phishing detection.

    Usage
    -----
    trainer = ModelTrainer()
    trainer.train_all(X_train, y_train, X_test, y_test)
    trainer.cross_validate_all(X_train, y_train)
    trainer.hyperparameter_tune(X_train, y_train)
    """

    def __init__(self):
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1),
            'Decision Tree': DecisionTreeClassifier(
                random_state=42),
            'SVM': SVC(
                probability=True, random_state=42),
            'Logistic Regression': LogisticRegression(
                max_iter=1000, random_state=42),
            'KNN': KNeighborsClassifier(
                n_neighbors=5, n_jobs=-1),
            'Naive Bayes': GaussianNB(),
            'MLP': MLPClassifier(
                hidden_layer_sizes=(128, 64), max_iter=500, random_state=42),
            'XGBoost': XGBClassifier(
                n_estimators=100, use_label_encoder=False,
                eval_metric='logloss', random_state=42,
                verbosity=0, n_jobs=-1),
        }

        self.results        = {}   # per-model evaluation results
        self.trained_models = {}   # fitted model objects
        self.best_model_name = None
        self.best_model      = None

    # ── Training ──────────────────────────────────────────────────────────────

    def train_all(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test:  pd.DataFrame,
        y_test:  pd.Series,
    ) -> None:
        """Train all 8 models and evaluate on the test set."""
        print('\n' + '=' * 65)
        print('  TRAINING ALL MODELS')
        print('=' * 65)

        for name, model in self.models.items():
            print(f'\n  [{name}]')
            t0 = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - t0
            self.trained_models[name] = model

            y_pred = model.predict(X_test)
            y_prob = self._get_proba(model, X_test)

            acc  = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec  = recall_score(y_test, y_pred, zero_division=0)
            f1   = f1_score(y_test, y_pred, zero_division=0)
            auc  = roc_auc_score(y_test, y_prob) if y_prob is not None else 0.0
            cm   = confusion_matrix(y_test, y_pred)

            # False positive / negative rates
            tn, fp, fn, tp = cm.ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

            self.results[name] = {
                'accuracy'        : acc,
                'precision'       : prec,
                'recall'          : rec,
                'f1'              : f1,
                'roc_auc'         : auc,
                'fpr'             : fpr,
                'fnr'             : fnr,
                'confusion_matrix': cm,
                'y_pred'          : y_pred,
                'y_prob'          : y_prob,
                'train_time'      : train_time,
                'model'           : model,
            }

            print(f'    Accuracy  : {acc:.4f}')
            print(f'    Precision : {prec:.4f}')
            print(f'    Recall    : {rec:.4f}')
            print(f'    F1        : {f1:.4f}')
            print(f'    AUC       : {auc:.4f}')
            print(f'    FPR       : {fpr:.4f}  (false positive rate)')
            print(f'    FNR       : {fnr:.4f}  (false negative rate)')
            print(f'    Time      : {train_time:.2f}s')

        # Select best model by F1 score
        self.best_model_name = max(
            self.results, key=lambda n: self.results[n]['f1']
        )
        self.best_model = self.trained_models[self.best_model_name]
        print(f'\n  Best model (by F1): {self.best_model_name}')

    # ── Cross-validation ──────────────────────────────────────────────────────

    def cross_validate_all(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        cv: int = 10,
    ) -> None:
        """Run stratified k-fold cross-validation on all models."""
        print('\n' + '=' * 65)
        print(f'  CROSS-VALIDATION ({cv}-fold stratified)')
        print('=' * 65)

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

        for name, model in self.models.items():
            print(f'\n  [{name}]')
            t0 = time.time()
            cv_results = cross_validate(
                model, X_train, y_train,
                cv=skf, scoring=scoring,
                return_train_score=False,
                n_jobs=-1,
            )
            elapsed = time.time() - t0

            f1_scores  = cv_results['test_f1']
            auc_scores = cv_results['test_roc_auc']

            self.results[name]['cv_f1_mean']  = f1_scores.mean()
            self.results[name]['cv_f1_std']   = f1_scores.std()
            self.results[name]['cv_auc_mean'] = auc_scores.mean()
            self.results[name]['cv_auc_std']  = auc_scores.std()
            self.results[name]['cv_scores']   = cv_results

            print(f'    CV F1  : {f1_scores.mean():.4f} ± {f1_scores.std():.4f}')
            print(f'    CV AUC : {auc_scores.mean():.4f} ± {auc_scores.std():.4f}')
            print(f'    Time   : {elapsed:.1f}s')

    # ── Hyperparameter tuning ─────────────────────────────────────────────────

    def hyperparameter_tune(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        cv: int = 5,
    ) -> None:
        """
        Run GridSearchCV on models that have a defined parameter grid.
        Updates the best model if a tuned version outperforms the original.
        """
        print('\n' + '=' * 65)
        print('  HYPERPARAMETER TUNING')
        print('=' * 65)

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        for name, grid in PARAM_GRIDS.items():
            if name not in self.models:
                continue
            print(f'\n  Tuning: {name}')
            base_model = self.models[name]

            gs = GridSearchCV(
                base_model, grid,
                cv=skf, scoring='f1',
                n_jobs=-1, refit=True,
                verbose=0,
            )
            t0 = time.time()
            gs.fit(X_train, y_train)
            elapsed = time.time() - t0

            best_params = gs.best_params_
            best_cv_f1  = gs.best_score_

            print(f'    Best params : {best_params}')
            print(f'    Best CV F1  : {best_cv_f1:.4f}')
            print(f'    Time        : {elapsed:.1f}s')

            # Update stored model with tuned version
            self.trained_models[name] = gs.best_estimator_
            self.results[name]['tuned_params'] = best_params
            self.results[name]['tuned_cv_f1']  = best_cv_f1
            self.results[name]['model']        = gs.best_estimator_

        # Re-select best model after tuning
        self.best_model_name = max(
            self.results,
            key=lambda n: self.results[n].get('tuned_cv_f1',
                          self.results[n]['f1'])
        )
        self.best_model = self.trained_models[self.best_model_name]
        print(f'\n  Best model after tuning: {self.best_model_name}')

    # ── Comparison table ──────────────────────────────────────────────────────

    def get_comparison_table(self) -> pd.DataFrame:
        """Return a DataFrame summarising all model results."""
        rows = []
        for name, res in self.results.items():
            row = {
                'Model'          : name,
                'Test Accuracy'  : round(res['accuracy'],  4),
                'Test Precision' : round(res['precision'], 4),
                'Test Recall'    : round(res['recall'],    4),
                'Test F1'        : round(res['f1'],        4),
                'Test AUC'       : round(res['roc_auc'],   4),
                'FPR'            : round(res['fpr'],        4),
                'FNR'            : round(res['fnr'],        4),
                'Train Time (s)' : round(res['train_time'], 2),
            }
            if 'cv_f1_mean' in res:
                row['CV F1 Mean'] = round(res['cv_f1_mean'], 4)
                row['CV F1 Std']  = round(res['cv_f1_std'],  4)
                row['CV AUC Mean']= round(res['cv_auc_mean'], 4)
            if 'tuned_cv_f1' in res:
                row['Tuned CV F1'] = round(res['tuned_cv_f1'], 4)
            rows.append(row)

        df = pd.DataFrame(rows)
        df = df.sort_values('Test F1', ascending=False).reset_index(drop=True)
        return df

    # ── Save models ───────────────────────────────────────────────────────────

    def save_best_model(self, path: str = 'models/best_model.pkl') -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.best_model, path)
        print(f'\n  Best model saved → {path}')

    def save_all_models(self, output_dir: str = 'models') -> None:
        os.makedirs(output_dir, exist_ok=True)
        for name, model in self.trained_models.items():
            fname = name.lower().replace(' ', '_') + '.pkl'
            joblib.dump(model, os.path.join(output_dir, fname))
        print(f'  All models saved → {output_dir}/')

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _get_proba(model, X) -> np.ndarray | None:
        """Return class-1 probability array, or None if unavailable."""
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)[:, 1]
        if hasattr(model, 'decision_function'):
            scores = model.decision_function(X)
            # Sigmoid transform for binary decision function
            return 1.0 / (1.0 + np.exp(-scores))
        return None
