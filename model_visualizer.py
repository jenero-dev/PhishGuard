"""
Component 2: Model Evaluation Visualizer
Generates publication-quality charts and figures for the dissertation.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve


class ModelVisualizer:
    """Generates visualizations for model evaluation results."""

    def __init__(self, output_dir='models/figures'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0',
                       '#00BCD4', '#FF5722', '#795548', '#607D8B', '#3F51B5']
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'figure.dpi': 150,
            'savefig.bbox': 'tight'
        })

    def plot_model_comparison(self, results):
        """Bar chart comparing all models across key metrics."""
        models = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']

        fig, ax = plt.subplots(figsize=(14, 7))
        x = np.arange(len(models))
        width = 0.15

        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            values = [results[m][metric] for m in models]
            bars = ax.bar(x + i * width, values, width, label=label, color=self.colors[i])

        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend(loc='lower right')
        ax.set_ylim(0.5, 1.05)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        path = os.path.join(self.output_dir, 'model_comparison.png')
        plt.savefig(path)
        plt.close()
        print(f"   ✅ Saved: {path}")

    def plot_confusion_matrices(self, results, y_test):
        """Plot confusion matrices for all models."""
        n_models = len(results)
        cols = 3
        rows = (n_models + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
        axes = axes.flatten() if n_models > 1 else [axes]

        for idx, (name, metrics) in enumerate(results.items()):
            cm = np.array(metrics['confusion_matrix'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=['Legitimate', 'Phishing'],
                       yticklabels=['Legitimate', 'Phishing'])
            axes[idx].set_title(f'{name}')
            axes[idx].set_ylabel('Actual')
            axes[idx].set_xlabel('Predicted')

        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle('Confusion Matrices - All Models', fontsize=16, y=1.02)
        plt.tight_layout()

        path = os.path.join(self.output_dir, 'confusion_matrices.png')
        plt.savefig(path)
        plt.close()
        print(f"   ✅ Saved: {path}")

    def plot_roc_curves(self, results):
        """Plot ROC curves for all models on one chart."""
        fig, ax = plt.subplots(figsize=(10, 8))

        for idx, (name, metrics) in enumerate(results.items()):
            fpr = metrics['roc_curve']['fpr']
            tpr = metrics['roc_curve']['tpr']
            auc = metrics['roc_auc']
            ax.plot(fpr, tpr, color=self.colors[idx % len(self.colors)],
                   linewidth=2, label=f'{name} (AUC = {auc:.4f})')

        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves - All Models')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()

        path = os.path.join(self.output_dir, 'roc_curves.png')
        plt.savefig(path)
        plt.close()
        print(f"   ✅ Saved: {path}")

    def plot_precision_recall_curves(self, results):
        """Plot Precision-Recall curves for all models."""
        fig, ax = plt.subplots(figsize=(10, 8))

        for idx, (name, metrics) in enumerate(results.items()):
            prec = metrics['pr_curve']['precision']
            rec = metrics['pr_curve']['recall']
            f1 = metrics['f1_score']
            ax.plot(rec, prec, color=self.colors[idx % len(self.colors)],
                   linewidth=2, label=f'{name} (F1 = {f1:.4f})')

        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curves - All Models')
        ax.legend(loc='lower left', fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()

        path = os.path.join(self.output_dir, 'precision_recall_curves.png')
        plt.savefig(path)
        plt.close()
        print(f"   ✅ Saved: {path}")

    def plot_cross_validation_scores(self, models_info):
        """Box plot of cross-validation scores."""
        fig, ax = plt.subplots(figsize=(12, 6))

        names = list(models_info.keys())
        cv_data = [models_info[name]['cv_scores'] for name in names]

        bp = ax.boxplot(cv_data, labels=names, patch_artist=True)

        for idx, patch in enumerate(bp['boxes']):
            patch.set_facecolor(self.colors[idx % len(self.colors)])
            patch.set_alpha(0.7)

        ax.set_xlabel('Models')
        ax.set_ylabel('Accuracy')
        ax.set_title('10-Fold Cross-Validation Accuracy Distribution')
        plt.xticks(rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        path = os.path.join(self.output_dir, 'cross_validation_boxplot.png')
        plt.savefig(path)
        plt.close()
        print(f"   ✅ Saved: {path}")

    def plot_feature_importance(self, feature_importance_df, top_n=15):
        """Plot feature importance from the best model."""
        if feature_importance_df is None:
            print("   ⚠️ Feature importance not available for this model type")
            return

        top_features = feature_importance_df.head(top_n)

        fig, ax = plt.subplots(figsize=(10, 8))
        bars = ax.barh(range(len(top_features)), top_features['Importance'].values,
                      color=self.colors[0], alpha=0.8)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['Feature'].values)
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Most Important Features')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()

        path = os.path.join(self.output_dir, 'feature_importance.png')
        plt.savefig(path)
        plt.close()
        print(f"   ✅ Saved: {path}")

    def plot_training_times(self, results):
        """Bar chart of model training times."""
        models = list(results.keys())
        times = [results[m]['training_time'] for m in models]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(models, times, color=self.colors[:len(models)], alpha=0.8)

        for bar, t in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                   f'{t:.2f}s', ha='center', va='bottom', fontsize=9)

        ax.set_xlabel('Models')
        ax.set_ylabel('Training Time (seconds)')
        ax.set_title('Model Training Time Comparison')
        plt.xticks(rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        path = os.path.join(self.output_dir, 'training_times.png')
        plt.savefig(path)
        plt.close()
        print(f"   ✅ Saved: {path}")

    def plot_metrics_heatmap(self, results):
        """Heatmap of all metrics across models."""
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'mcc']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC', 'MCC']
        models = list(results.keys())

        data = np.array([[results[m][metric] for metric in metrics] for m in models])

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(data, annot=True, fmt='.4f', cmap='YlOrRd',
                   xticklabels=metric_labels, yticklabels=models, ax=ax,
                   vmin=0.7, vmax=1.0)
        ax.set_title('Model Performance Heatmap')
        plt.tight_layout()

        path = os.path.join(self.output_dir, 'metrics_heatmap.png')
        plt.savefig(path)
        plt.close()
        print(f"   ✅ Saved: {path}")

    def generate_all_figures(self, results, models_info, y_test, feature_importance_df=None):
        """Generate all visualization figures."""
        print("\n" + "="*70)
        print("📊 GENERATING EVALUATION FIGURES")
        print("="*70)

        print("\n1. Model Comparison Bar Chart")
        self.plot_model_comparison(results)

        print("2. Confusion Matrices")
        self.plot_confusion_matrices(results, y_test)

        print("3. ROC Curves")
        self.plot_roc_curves(results)

        print("4. Precision-Recall Curves")
        self.plot_precision_recall_curves(results)

        print("5. Cross-Validation Box Plot")
        self.plot_cross_validation_scores(models_info)

        print("6. Feature Importance")
        self.plot_feature_importance(feature_importance_df)

        print("7. Training Times")
        self.plot_training_times(results)

        print("8. Metrics Heatmap")
        self.plot_metrics_heatmap(results)

        print(f"\n✅ All figures saved to: {self.output_dir}/")
