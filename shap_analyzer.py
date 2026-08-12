"""
SHAP Explainability Analyzer for Phishing Website Detection
===========================================================
Component 2 (Stage 4): Model Explainability.

Provides a single reusable class that:
    * Builds an appropriate SHAP explainer for any trained sklearn / XGBoost model
      (TreeExplainer for tree ensembles, LinearExplainer for linear models,
       KernelExplainer as a model-agnostic fallback).
    * Generates a global SHAP summary (beeswarm) plot + a mean-|SHAP| bar plot.
    * Produces per-prediction (local) SHAP values for the real-time backend, so
      the frontend "Feature Importance (SHAP)" panel can explain *why* a URL was
      flagged.

This module is imported by both `model_evaluator.py` (offline evaluation) and
`app.py` (online per-request explanations).

Author: MSc Cybersecurity Project - University of the West of Scotland
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import shap
    _HAS_SHAP = True
except Exception:  # pragma: no cover
    _HAS_SHAP = False


# Tree-based models where TreeExplainer is exact and fast
_TREE_MODELS = (
    "RandomForestClassifier", "DecisionTreeClassifier",
    "XGBClassifier", "GradientBoostingClassifier", "ExtraTreesClassifier",
)
# Linear models where LinearExplainer is appropriate
_LINEAR_MODELS = ("LogisticRegression",)


class ShapAnalyzer:
    """Wraps SHAP explainer construction and plotting/local-explanation logic."""

    def __init__(self, model, feature_names, background=None, max_background=100):
        """
        Parameters
        ----------
        model : fitted estimator
        feature_names : list[str]
        background : pd.DataFrame | np.ndarray | None
            Background/reference dataset used by Kernel/Linear explainers.
        max_background : int
            Cap on background rows (kept small for speed).
        """
        self.model = model
        self.feature_names = list(feature_names)
        self.model_name = type(model).__name__
        self.explainer = None
        self.available = _HAS_SHAP

        if background is not None:
            bg = np.asarray(background)
            if len(bg) > max_background:
                rng = np.random.RandomState(42)
                idx = rng.choice(len(bg), max_background, replace=False)
                bg = bg[idx]
            self.background = bg
        else:
            self.background = None

        if self.available:
            self._build_explainer()

    # ------------------------------------------------------------------
    def _build_explainer(self):
        """Choose the most appropriate explainer for the model type."""
        try:
            if self.model_name in _TREE_MODELS:
                self.explainer = shap.TreeExplainer(self.model)
            elif self.model_name in _LINEAR_MODELS and self.background is not None:
                self.explainer = shap.LinearExplainer(self.model, self.background)
            elif self.background is not None and hasattr(self.model, "predict_proba"):
                # Model-agnostic fallback (slower) using a summarised background
                bg = shap.kmeans(self.background, min(10, len(self.background)))
                self.explainer = shap.KernelExplainer(
                    lambda x: self.model.predict_proba(x)[:, 1], bg
                )
            else:
                self.explainer = None
        except Exception as e:  # pragma: no cover
            print(f"  [SHAP] Could not build explainer for {self.model_name}: {e}")
            self.explainer = None

    # ------------------------------------------------------------------
    @staticmethod
    def _phishing_shap(shap_values):
        """
        Normalise the many possible SHAP output shapes into a single 2D array
        (n_samples, n_features) corresponding to the *phishing* (positive) class.
        """
        # New SHAP API: Explanation object
        if hasattr(shap_values, "values"):
            vals = shap_values.values
        else:
            vals = shap_values

        vals = np.array(vals)

        # (n_classes, n_samples, n_features) -> take positive class
        if vals.ndim == 3:
            # shape could be (n_samples, n_features, n_classes) or (n_classes, n_samples, n_features)
            if vals.shape[-1] == 2:          # (..., n_features, n_classes)
                vals = vals[..., 1]
            else:                            # (n_classes, n_samples, n_features)
                vals = vals[1]
        return vals

    # ------------------------------------------------------------------
    def global_summary(self, X, out_dir, tag="best_model", max_samples=300):
        """
        Generate global SHAP plots:
            * beeswarm summary plot
            * mean(|SHAP|) bar plot
        Returns a dict of {feature: mean_abs_shap}.
        """
        if not self.available or self.explainer is None:
            print(f"  [SHAP] Skipping global summary for {self.model_name} (unavailable)")
            return {}

        os.makedirs(out_dir, exist_ok=True)
        X = pd.DataFrame(X, columns=self.feature_names)
        if len(X) > max_samples:
            X = X.sample(max_samples, random_state=42)

        try:
            shap_values = self.explainer.shap_values(X) \
                if hasattr(self.explainer, "shap_values") else self.explainer(X)
        except Exception as e:  # pragma: no cover
            print(f"  [SHAP] shap_values failed: {e}")
            return {}

        vals = self._phishing_shap(shap_values)

        # --- Beeswarm summary plot ---
        try:
            plt.figure()
            shap.summary_plot(vals, X, feature_names=self.feature_names, show=False)
            plt.title(f"SHAP Summary - {tag}", fontsize=12, fontweight="bold")
            plt.tight_layout()
            beeswarm_path = os.path.join(out_dir, f"shap_summary_{tag}.png")
            plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  [SHAP] Saved -> {beeswarm_path}")
        except Exception as e:  # pragma: no cover
            print(f"  [SHAP] beeswarm plot failed: {e}")

        # --- Mean |SHAP| bar plot ---
        mean_abs = np.abs(vals).mean(axis=0)
        order = np.argsort(mean_abs)[::-1]
        importance = {self.feature_names[i]: float(mean_abs[i]) for i in order}

        try:
            top = order[:20][::-1]
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(top)), mean_abs[top], color="#00d4ff", alpha=0.85)
            plt.yticks(range(len(top)), [self.feature_names[i] for i in top])
            plt.xlabel("mean(|SHAP value|)")
            plt.title(f"Global Feature Importance (SHAP) - {tag}",
                      fontsize=12, fontweight="bold")
            plt.tight_layout()
            bar_path = os.path.join(out_dir, f"shap_bar_{tag}.png")
            plt.savefig(bar_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  [SHAP] Saved -> {bar_path}")
        except Exception as e:  # pragma: no cover
            print(f"  [SHAP] bar plot failed: {e}")

        return importance

    # ------------------------------------------------------------------
    def explain_instance(self, x_row, top_k=10):
        """
        Compute per-prediction (local) SHAP values for a single scaled instance.

        Parameters
        ----------
        x_row : array-like, shape (n_features,) or (1, n_features)
        top_k : int
            Number of top contributing features to return.

        Returns
        -------
        dict with keys:
            base_value : float
            contributions : list[{feature, shap_value, abs_value}]  (sorted desc)
        """
        if not self.available or self.explainer is None:
            return {"base_value": 0.0, "contributions": []}

        X = np.atleast_2d(np.asarray(x_row, dtype=float))
        Xdf = pd.DataFrame(X, columns=self.feature_names)

        try:
            shap_values = self.explainer.shap_values(Xdf) \
                if hasattr(self.explainer, "shap_values") else self.explainer(Xdf)
            vals = self._phishing_shap(shap_values)
            vals = np.atleast_2d(vals)[0]

            # Base (expected) value for the positive class
            base = self.explainer.expected_value if hasattr(self.explainer, "expected_value") else 0.0
            if isinstance(base, (list, np.ndarray)):
                base = np.array(base).ravel()
                base = float(base[1]) if base.size >= 2 else float(base[0])
            else:
                base = float(base)
        except Exception as e:  # pragma: no cover
            print(f"  [SHAP] explain_instance failed: {e}")
            return {"base_value": 0.0, "contributions": []}

        order = np.argsort(np.abs(vals))[::-1][:top_k]
        contributions = [
            {
                "feature": self.feature_names[i],
                "shap_value": round(float(vals[i]), 6),
                "abs_value": round(float(abs(vals[i])), 6),
            }
            for i in order
        ]
        return {"base_value": round(base, 6), "contributions": contributions}
