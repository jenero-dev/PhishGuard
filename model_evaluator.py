"""
model_evaluator.py
==================
PhishGuard - Phishing Website Detection System
Component Model Evaluation

Generates all evaluation artefacts:
    - Confusion matrices
    - ROC curves
    - Precision-Recall curves
    - Cross-validation box plots
    - Feature importance plots
    - Model comparison chart
    - External holdout evaluation
    - Leakage inspection report
    - Full results JSON

Author  : MSc Cybersecurity Project - University of the West of Scotland
Project : PhishGuard
"""

import os
import json
import logging

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc,
    precision_recall_curve,
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
    classification_report,
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Generates all evaluation plots and reports for PhishGuard.

    Parameters
    ----------
    output_dir : str — root directory for saving results
    """

    def __init__(self, output_dir: str = 'results'):
        self.output_dir = output_dir
        self.plot_dir   = os.path.join(output_dir, 'plots')
        self.report_dir = os.path.join(output_dir, 'reports')
        os.makedirs(self.plot_dir,   exist_ok=True)
        os.makedirs(self.report_dir, exist_ok=True)

    # ── Main entry point ──────────────────────────────────────────────────────

    def generate_all(
        self,
        trainer,
        X_test:  pd.DataFrame,
        y_test:  pd.Series,
        X_val:   pd.DataFrame = None,
        y_val:   pd.Series    = None,
    ) -> pd.DataFrame:
        """
        Generate all evaluation artefacts.

        Parameters
        ----------
        trainer : ModelTrainer — fitted trainer object
        X_test  : pd.DataFrame — scaled test features
        y_test  : pd.Series    — test labels
        X_val   : pd.DataFrame — scaled validation features (optional)
        y_val   : pd.Series    — validation labels (optional)

        Returns
        -------
        pd.DataFrame — model comparison table
        """
        print('\n' + '=' * 65)
        print('  GENERATING EVALUATION ARTEFACTS')
        print('=' * 65)

        self._plot_confusion_matrices(trainer, y_test)
        self._plot_roc_curves(trainer, y_test)
        self._plot_precision_recall_curves(trainer, y_test)
        self._plot_cv_boxplots(trainer)
        self._plot_feature_importance(trainer, X_test)
        self._plot_model_comparison(trainer)

        if X_val is not None and y_val is not None:
            self._evaluate_on_validation(trainer, X_val, y_val)

        comparison_df = trainer.get_comparison_table()
        self._save_results(trainer, comparison_df, y_test)

        print('\n' + '=' * 65)
        print('  ALL EVALUATION ARTEFACTS GENERATED')
        print('=' * 65)

        return comparison_df

    # ── Confusion matrices ────────────────────────────────────────────────────

    def _plot_confusion_matrices(self, trainer, y_test):
        print('\n  Generating confusion matrices...')
        n_models = len(trainer.results)
        cols = 3
        rows = (n_models + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axes = axes.flatten() if n_models > 1 else [axes]

        for idx, (name, result) in enumerate(trainer.results.items()):
            cm   = result['confusion_matrix']
            disp = ConfusionMatrixDisplay(
                cm, display_labels=['Legitimate', 'Phishing']
            )
            disp.plot(ax=axes[idx], cmap='Blues', colorbar=False)
            axes[idx].set_title(name, fontsize=10, fontweight='bold')

        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        path = os.path.join(self.plot_dir, 'confusion_matrices.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'    Saved → {path}')

    # ── ROC curves ────────────────────────────────────────────────────────────

    def _plot_roc_curves(self, trainer, y_test):
        print('  Generating ROC curves...')
        fig, ax = plt.subplots(figsize=(9, 7))
        colors  = plt.cm.Set2(np.linspace(0, 1, len(trainer.results)))

        for (name, result), color in zip(trainer.results.items(), colors):
            if result['y_prob'] is not None:
                fpr, tpr, _ = roc_curve(y_test, result['y_prob'])
                roc_auc     = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=color, lw=2,
                        label=f'{name} (AUC={roc_auc:.4f})')

        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC=0.50)')
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves — All Models', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)

        path = os.path.join(self.plot_dir, 'roc_curves.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'    Saved → {path}')

    # ── Precision-Recall curves ───────────────────────────────────────────────

    def _plot_precision_recall_curves(self, trainer, y_test):
        print('  Generating Precision-Recall curves...')
        fig, ax = plt.subplots(figsize=(9, 7))
        colors  = plt.cm.Set2(np.linspace(0, 1, len(trainer.results)))

        for (name, result), color in zip(trainer.results.items(), colors):
            if result['y_prob'] is not None:
                prec, rec, _ = precision_recall_curve(y_test, result['y_prob'])
                ax.plot(rec, prec, color=color, lw=2, label=name)

        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
        ax.legend(loc='lower left', fontsize=9)
        ax.grid(True, alpha=0.3)

        path = os.path.join(self.plot_dir, 'precision_recall_curves.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'    Saved → {path}')

    # ── CV box plots ──────────────────────────────────────────────────────────

    def _plot_cv_boxplots(self, trainer):
        print('  Generating cross-validation box plots...')
        cv_data = []
        for name, result in trainer.results.items():
            if 'cv_scores' in result:
                for s in result['cv_scores']['test_f1']:
                    cv_data.append({'Model': name, 'F1 Score': s})

        if not cv_data:
            print('    [SKIP] No CV data available')
            return

        df  = pd.DataFrame(cv_data)
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=df, x='Model', y='F1 Score', ax=ax, palette='Set2')
        ax.set_title('Cross-Validation F1 Score Distribution',
                     fontsize=14, fontweight='bold')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
        ax.set_ylim(max(0, df['F1 Score'].min() - 0.05), 1.01)
        ax.grid(True, alpha=0.3, axis='y')

        path = os.path.join(self.plot_dir, 'cv_boxplots.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'    Saved → {path}')

    # ── Feature importance ────────────────────────────────────────────────────

    def _plot_feature_importance(self, trainer, X_test):
        print('  Generating feature importance plots...')
        tree_models = ['Random Forest', 'XGBoost', 'Decision Tree']

        for name in tree_models:
            if name not in trainer.results:
                continue
            model = trainer.results[name]['model']
            if not hasattr(model, 'feature_importances_'):
                continue

            importances   = model.feature_importances_
            feature_names = X_test.columns.tolist()
            indices       = np.argsort(importances)[::-1][:20]

            fig, ax = plt.subplots(figsize=(10, 7))
            ax.barh(
                range(len(indices)),
                importances[indices][::-1],
                color='steelblue', alpha=0.85
            )
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels([feature_names[i] for i in indices[::-1]])
            ax.set_xlabel('Feature Importance', fontsize=12)
            ax.set_title(f'Feature Importance — {name}',
                         fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')

            safe = name.lower().replace(' ', '_')
            path = os.path.join(self.plot_dir, f'feature_importance_{safe}.png')
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f'    Saved → {path}')

    # ── Model comparison chart ────────────────────────────────────────────────

    def _plot_model_comparison(self, trainer):
        print('  Generating model comparison chart...')
        df      = trainer.get_comparison_table()
        metrics = ['Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1']
        avail   = [m for m in metrics if m in df.columns]

        fig, ax = plt.subplots(figsize=(13, 6))
        x       = np.arange(len(df))
        width   = 0.18
        colors  = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']

        for i, metric in enumerate(avail):
            offset = (i - len(avail) / 2 + 0.5) * width
            ax.bar(x + offset, df[metric], width,
                   label=metric, color=colors[i], alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(df['Model'], rotation=30, ha='right')
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Performance Comparison',
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=9)
        ax.set_ylim(max(0, df[avail].min().min() - 0.05), 1.01)
        ax.grid(True, alpha=0.3, axis='y')

        path = os.path.join(self.plot_dir, 'model_comparison.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'    Saved → {path}')

    # ── Validation set evaluation ─────────────────────────────────────────────

    def _evaluate_on_validation(self, trainer, X_val, y_val):
        """Evaluate all models on the validation set and print results."""
        print('\n  Validation set evaluation:')
        print(f'  {"Model":<22} {"Acc":>6} {"F1":>6} {"AUC":>6} '
              f'{"FPR":>6} {"FNR":>6}')
        print('  ' + '-' * 55)

        for name, model in trainer.trained_models.items():
            y_pred = model.predict(X_val)
            y_prob = trainer._get_proba(model, X_val)

            acc  = accuracy_score(y_val, y_pred)
            f1   = f1_score(y_val, y_pred, zero_division=0)
            auc_ = roc_auc_score(y_val, y_prob) if y_prob is not None else 0.0
            cm   = confusion_matrix(y_val, y_pred)
            tn, fp, fn, tp = cm.ravel()
            fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            fnr  = fn / (fn + tp) if (fn + tp) > 0 else 0.0

            trainer.results[name]['val_accuracy'] = acc
            trainer.results[name]['val_f1']       = f1
            trainer.results[name]['val_auc']      = auc_
            trainer.results[name]['val_fpr']      = fpr
            trainer.results[name]['val_fnr']      = fnr

            print(f'  {name:<22} {acc:>6.4f} {f1:>6.4f} {auc_:>6.4f} '
                  f'{fpr:>6.4f} {fnr:>6.4f}')

    # ── External holdout evaluation ───────────────────────────────────────────

    def evaluate_external_holdout(
        self,
        trainer,
        holdout_path: str,
        scaler,
        feature_names: list,
    ) -> dict:
        """
        Evaluate the best model on the external phishing holdout.

        Parameters
        ----------
        trainer       : ModelTrainer
        holdout_path  : str  — path to external_phishing_holdout.csv
        scaler        : fitted StandardScaler
        feature_names : list — ordered feature names

        Returns
        -------
        dict — evaluation metrics on holdout
        """
        if not os.path.exists(holdout_path):
            print(f'\n  [HOLDOUT] File not found: {holdout_path}')
            return {}

        print(f'\n  Evaluating on external holdout: {holdout_path}')
        df = pd.read_csv(holdout_path)

        if 'label' not in df.columns:
            df['label'] = 1   # all records in verified_online are phishing

        y_holdout = df['label'].astype(int)

        # Align features
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0
        X_holdout = df[feature_names].fillna(0).astype(float)
        X_scaled  = scaler.transform(X_holdout)

        model    = trainer.best_model
        y_pred   = model.predict(X_scaled)
        y_prob   = trainer._get_proba(model, X_scaled)

        acc  = accuracy_score(y_holdout, y_pred)
        prec = precision_score(y_holdout, y_pred, zero_division=0)
        rec  = recall_score(y_holdout, y_pred, zero_division=0)
        f1   = f1_score(y_holdout, y_pred, zero_division=0)
        auc_ = roc_auc_score(y_holdout, y_prob) if y_prob is not None else 0.0

        results = {
            'model'    : trainer.best_model_name,
            'samples'  : len(y_holdout),
            'accuracy' : round(acc,  4),
            'precision': round(prec, 4),
            'recall'   : round(rec,  4),
            'f1'       : round(f1,   4),
            'roc_auc'  : round(auc_, 4),
        }

        print(f'    Model    : {trainer.best_model_name}')
        print(f'    Samples  : {len(y_holdout)}')
        print(f'    Accuracy : {acc:.4f}')
        print(f'    Recall   : {rec:.4f}  (phishing detection rate)')
        print(f'    F1       : {f1:.4f}')
        print(f'    AUC      : {auc_:.4f}')

        # Save holdout results
        path = os.path.join(self.report_dir, 'external_holdout_results.json')
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'    Saved → {path}')

        return results

    # ── Save results ──────────────────────────────────────────────────────────

    def _save_results(self, trainer, comparison_df, y_test):
        print('\n  Saving evaluation results...')

        # Comparison CSV
        csv_path = os.path.join(self.output_dir, 'model_comparison.csv')
        comparison_df.to_csv(csv_path, index=False)
        print(f'    Comparison table → {csv_path}')

        # Per-model classification reports
        for name, result in trainer.results.items():
            if 'y_pred' not in result:
                continue
            safe = name.lower().replace(' ', '_')
            path = os.path.join(self.report_dir, f'{safe}_report.txt')
            with open(path, 'w') as f:
                f.write(f'Classification Report: {name}\n')
                f.write('=' * 50 + '\n\n')
                f.write(f'Confusion Matrix:\n{result["confusion_matrix"]}\n\n')
                f.write(f'Accuracy  : {result["accuracy"]:.4f}\n')
                f.write(f'Precision : {result["precision"]:.4f}\n')
                f.write(f'Recall    : {result["recall"]:.4f}\n')
                f.write(f'F1 Score  : {result["f1"]:.4f}\n')
                f.write(f'ROC AUC   : {result["roc_auc"]:.4f}\n')
                f.write(f'FPR       : {result["fpr"]:.4f}\n')
                f.write(f'FNR       : {result["fnr"]:.4f}\n')
                if 'cv_f1_mean' in result:
                    f.write(f'\nCV F1 Mean : {result["cv_f1_mean"]:.4f}\n')
                    f.write(f'CV F1 Std  : {result["cv_f1_std"]:.4f}\n')
                if 'tuned_params' in result:
                    f.write(f'\nTuned Params : {result["tuned_params"]}\n')
                    f.write(f'Tuned CV F1  : {result["tuned_cv_f1"]:.4f}\n')
                if 'val_f1' in result:
                    f.write(f'\nValidation F1  : {result["val_f1"]:.4f}\n')
                    f.write(f'Validation AUC : {result["val_auc"]:.4f}\n')
                    f.write(f'Validation FPR : {result["val_fpr"]:.4f}\n')

        # Full results JSON
        json_results = {}
        for name, result in trainer.results.items():
            json_results[name] = {
                k: (v.tolist() if hasattr(v, 'tolist') else v)
                for k, v in result.items()
                if k not in ('y_pred', 'y_prob', 'confusion_matrix',
                             'cv_scores', 'model')
            }
            if 'confusion_matrix' in result:
                json_results[name]['confusion_matrix'] = \
                    result['confusion_matrix'].tolist()

        json_path = os.path.join(self.output_dir, 'full_results.json')
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        print(f'    Full results → {json_path}')
        print(f'    Reports      → {self.report_dir}/')
