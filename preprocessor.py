"""
preprocessor.py
===============
PhishGuard - Phishing Website Detection System
Component 1: Data Preprocessing

Responsibilities
----------------
1. Clean and validate raw feature data
2. Deduplicate by canonical URL and registered domain
3. Domain-aware train / validation / test splitting
   (all URLs from the same registered domain stay in one partition)
4. Fit imputation, outlier capping, and scaling on TRAINING data only
5. Apply fitted transformations to validation and test data
6. Class balancing applied to training partition only
7. Save all artefacts for reproducibility

Design principles
-----------------
- No preprocessing statistics are learned from validation or test data
- Class distributions in validation and test sets are left untouched
- WHOIS sentinel values (-1) are treated as "unknown", not as phishing
- Retrieval/context features are flagged for leakage inspection

Author  : MSc Cybersecurity Project - University of the West of Scotland
Project : PhishGuard
"""

import os
import json
import logging
import warnings

import numpy as np
import pandas as pd
import joblib
import tldextract
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ── Feature schema ────────────────────────────────────────────────────────────

# Authoritative 30-feature list — must match FeatureExtractor.FEATURE_NAMES
FEATURE_NAMES = [
    # A — Lexical (8)
    'url_length', 'digit_ratio', 'special_char_ratio', 'url_entropy',
    'has_ip_address', 'has_at_symbol', 'has_non_standard_port', 'uses_http',
    # B — Domain / DNS (8)
    'domain_age_days', 'registration_length_days', 'domain_expiry_proximity_days',
    'whois_data_available', 'has_a_or_aaaa_record', 'has_ns_record',
    'has_mx_record', 'num_name_servers',
    # C — Page content (10)
    'num_forms', 'has_password_field', 'has_identity_field', 'has_sensitive_input',
    'has_external_form_action', 'has_insecure_form_action',
    'credential_language_present', 'brand_domain_mismatch',
    'num_iframes', 'hidden_or_suspicious_iframe',
    # D — Retrieval / context (4)
    'page_fetch_success', 'http_status_category', 'redirect_count',
    'final_domain_changed',
]

# WHOIS fields that use -1 as "unavailable" sentinel (not zero)
WHOIS_SENTINEL_COLS = [
    'domain_age_days',
    'registration_length_days',
    'domain_expiry_proximity_days',
]

# Retrieval/context features — inspected for leakage before model training
RETRIEVAL_COLS = [
    'page_fetch_success',
    'http_status_category',
    'redirect_count',
    'final_domain_changed',
]

NON_FEATURE_COLS = ['url', 'label', 'registered_domain']


# ═════════════════════════════════════════════════════════════════════════════
# DataPreprocessor
# ═════════════════════════════════════════════════════════════════════════════

class DataPreprocessor:
    """
    Full preprocessing pipeline for PhishGuard.

    Parameters
    ----------
    random_state : int
        Seed for reproducibility.
    val_size     : float
        Fraction of data reserved for validation (default 0.10).
    test_size    : float
        Fraction of data reserved for testing (default 0.15).
    balance      : bool
        Whether to apply random undersampling to the training partition.
    """

    def __init__(
        self,
        random_state: int = 42,
        val_size:     float = 0.10,
        test_size:    float = 0.15,
        balance:      bool  = True,
    ):
        self.random_state = random_state
        self.val_size     = val_size
        self.test_size    = test_size
        self.balance      = balance

        self.scaler        = StandardScaler()
        self.feature_names = None          # set after preprocessing
        self._col_medians  = {}            # learned from training data only
        self._col_caps     = {}            # learned from training data only

    # ── Main pipeline ─────────────────────────────────────────────────────────

    def preprocess(self, df: pd.DataFrame) -> dict:
        """
        Run the full preprocessing pipeline.

        Parameters
        ----------
        df : pd.DataFrame
            Raw feature DataFrame (output of FeatureExtractor.extract_batch).
            Must contain 'url', 'label', and all 30 feature columns.

        Returns
        -------
        dict with keys:
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            feature_names, scaler,
            leakage_report
        """
        print('\n' + '=' * 65)
        print('  PREPROCESSING PIPELINE')
        print('=' * 65)

        # 1. Validate schema
        df = self._validate_schema(df)

        # 2. Clean labels
        df = self._clean_labels(df)

        # 3. Attach registered domain column
        df = self._attach_domain(df)

        # 4. Deduplicate
        df = self._deduplicate(df)

        # 5. Domain-aware split
        splits = self._domain_aware_split(df)
        train_df, val_df, test_df = splits['train'], splits['val'], splits['test']

        # 6. Separate features and labels
        X_train_raw = train_df[FEATURE_NAMES].copy()
        X_val_raw   = val_df[FEATURE_NAMES].copy()
        X_test_raw  = test_df[FEATURE_NAMES].copy()
        y_train = train_df['label'].astype(int).reset_index(drop=True)
        y_val   = val_df['label'].astype(int).reset_index(drop=True)
        y_test  = test_df['label'].astype(int).reset_index(drop=True)

        # 7. Handle WHOIS sentinel values (-1 → NaN for imputation)
        X_train_raw = self._sentinel_to_nan(X_train_raw)
        X_val_raw   = self._sentinel_to_nan(X_val_raw)
        X_test_raw  = self._sentinel_to_nan(X_test_raw)

        # 8. Impute missing values — fit on training only
        X_train_raw = self._fit_impute(X_train_raw)
        X_val_raw   = self._apply_impute(X_val_raw)
        X_test_raw  = self._apply_impute(X_test_raw)

        # 9. Cap outliers — fit on training only
        X_train_raw = self._fit_cap_outliers(X_train_raw)
        X_val_raw   = self._apply_cap_outliers(X_val_raw)
        X_test_raw  = self._apply_cap_outliers(X_test_raw)

        # 10. Balance training partition only
        if self.balance:
            X_train_raw, y_train = self._balance(X_train_raw, y_train)

        # 11. Scale — fit on training only
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train_raw),
            columns=FEATURE_NAMES
        )
        X_val_scaled = pd.DataFrame(
            self.scaler.transform(X_val_raw),
            columns=FEATURE_NAMES
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test_raw),
            columns=FEATURE_NAMES
        )

        self.feature_names = FEATURE_NAMES

        # 12. Leakage inspection for retrieval features
        leakage_report = self._inspect_retrieval_leakage(
            X_train_raw, y_train, X_test_raw, y_test
        )

        # Summary
        print(f'\n  Train : {len(X_train_scaled):>5} samples '
              f'| class dist: {dict(y_train.value_counts())}')
        print(f'  Val   : {len(X_val_scaled):>5} samples '
              f'| class dist: {dict(y_val.value_counts())}')
        print(f'  Test  : {len(X_test_scaled):>5} samples '
              f'| class dist: {dict(y_test.value_counts())}')
        print(f'  Features: {len(self.feature_names)}')

        return {
            'X_train'        : X_train_scaled,
            'X_val'          : X_val_scaled,
            'X_test'         : X_test_scaled,
            'y_train'        : y_train,
            'y_val'          : y_val,
            'y_test'         : y_test,
            'feature_names'  : self.feature_names,
            'scaler'         : self.scaler,
            'leakage_report' : leakage_report,
            # Raw (unscaled) test set for leakage analysis
            'X_test_raw'     : X_test_raw,
            'X_val_raw'      : X_val_raw,
        }

    # ── Step 1: Schema validation ─────────────────────────────────────────────

    def _validate_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        missing_cols = [c for c in FEATURE_NAMES if c not in df.columns]
        if missing_cols:
            logger.warning(
                'Missing feature columns (will be filled with 0): %s',
                missing_cols
            )
            for col in missing_cols:
                df[col] = 0
        if 'label' not in df.columns:
            raise ValueError("DataFrame must contain a 'label' column.")
        print(f'\n  [1] Schema validated — {len(df)} rows, '
              f'{len(df.columns)} columns')
        return df

    # ── Step 2: Label cleaning ────────────────────────────────────────────────

    def _clean_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.dropna(subset=['label'])
        df['label'] = df['label'].astype(int)
        df = df[df['label'].isin([0, 1])]
        removed = before - len(df)
        if removed:
            print(f'  [2] Removed {removed} rows with invalid labels')
        else:
            print(f'  [2] Labels clean — {len(df)} valid rows')
        return df.reset_index(drop=True)

    # ── Step 3: Attach registered domain ─────────────────────────────────────

    def _attach_domain(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'url' in df.columns:
            df['registered_domain'] = df['url'].apply(
                lambda u: tldextract.extract(str(u)).registered_domain.lower()
            )
        else:
            df['registered_domain'] = 'unknown'
        print(f'  [3] Registered domains identified: '
              f'{df["registered_domain"].nunique()}')
        return df

    # ── Step 4: Deduplication ─────────────────────────────────────────────────

    def _deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)

        # Remove exact URL duplicates
        if 'url' in df.columns:
            df['_url_norm'] = df['url'].str.strip().str.lower()
            df = df.drop_duplicates(subset=['_url_norm'])
            df = df.drop(columns=['_url_norm'])

        # Remove feature-level duplicates (same feature vector, same label)
        feature_subset = [c for c in FEATURE_NAMES if c in df.columns]
        df = df.drop_duplicates(subset=feature_subset + ['label'])

        removed = before - len(df)
        print(f'  [4] Deduplication: removed {removed} duplicates '
              f'({len(df)} remaining)')
        return df.reset_index(drop=True)

    # ── Step 5: Domain-aware split ────────────────────────────────────────────

    def _domain_aware_split(self, df: pd.DataFrame) -> dict:
        """
        Split data so that all URLs from the same registered domain
        remain in the same partition.

        Strategy
        --------
        1. Assign each unique domain to train / val / test
        2. Map URLs to their domain's partition
        """
        domains = df['registered_domain'].values
        labels  = df['label'].values

        # First split: carve out test set
        gss_test = GroupShuffleSplit(
            n_splits=1,
            test_size=self.test_size,
            random_state=self.random_state
        )
        train_val_idx, test_idx = next(
            gss_test.split(df, labels, groups=domains)
        )

        train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
        test_df      = df.iloc[test_idx].reset_index(drop=True)

        # Second split: carve out validation set from train_val
        val_fraction = self.val_size / (1 - self.test_size)
        gss_val = GroupShuffleSplit(
            n_splits=1,
            test_size=val_fraction,
            random_state=self.random_state
        )
        train_idx, val_idx = next(
            gss_val.split(
                train_val_df,
                train_val_df['label'].values,
                groups=train_val_df['registered_domain'].values
            )
        )

        train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
        val_df   = train_val_df.iloc[val_idx].reset_index(drop=True)

        # Verify no domain leakage
        train_domains = set(train_df['registered_domain'])
        val_domains   = set(val_df['registered_domain'])
        test_domains  = set(test_df['registered_domain'])

        tv_overlap  = train_domains & val_domains
        tt_overlap  = train_domains & test_domains
        vt_overlap  = val_domains   & test_domains

        print(f'\n  [5] Domain-aware split:')
        print(f'      Train  : {len(train_df):>5} rows | '
              f'{len(train_domains):>4} domains')
        print(f'      Val    : {len(val_df):>5} rows | '
              f'{len(val_domains):>4} domains')
        print(f'      Test   : {len(test_df):>5} rows | '
              f'{len(test_domains):>4} domains')
        print(f'      Domain leakage — train/val: {len(tv_overlap)} | '
              f'train/test: {len(tt_overlap)} | val/test: {len(vt_overlap)}')

        if tt_overlap:
            logger.warning(
                'Domain leakage detected between train and test: %s',
                list(tt_overlap)[:5]
            )

        return {'train': train_df, 'val': val_df, 'test': test_df}

    # ── Step 6: Sentinel → NaN ────────────────────────────────────────────────

    def _sentinel_to_nan(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replace -1 sentinel values in WHOIS columns with NaN."""
        for col in WHOIS_SENTINEL_COLS:
            if col in X.columns:
                X[col] = X[col].replace(-1, np.nan)
        return X

    # ── Step 7: Imputation ────────────────────────────────────────────────────

    def _fit_impute(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit column medians on training data and impute."""
        for col in X.columns:
            median = X[col].median()
            self._col_medians[col] = median if not np.isnan(median) else 0.0
            if X[col].isnull().any():
                X[col] = X[col].fillna(self._col_medians[col])
        missing = X.isnull().sum().sum()
        print(f'  [7] Imputation fitted — {missing} remaining nulls after fill')
        return X

    def _apply_impute(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply training medians to validation / test data."""
        for col in X.columns:
            if X[col].isnull().any():
                X[col] = X[col].fillna(self._col_medians.get(col, 0.0))
        return X

    # ── Step 8: Outlier capping ───────────────────────────────────────────────

    def _fit_cap_outliers(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit IQR-based caps on training data and apply."""
        capped = 0
        for col in X.columns:
            Q1  = X[col].quantile(0.25)
            Q3  = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lo  = Q1 - 3 * IQR
            hi  = Q3 + 3 * IQR
            self._col_caps[col] = (lo, hi)
            before = ((X[col] < lo) | (X[col] > hi)).sum()
            X[col] = X[col].clip(lo, hi)
            capped += before
        print(f'  [8] Outlier capping fitted — {capped} values capped in training')
        return X

    def _apply_cap_outliers(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply training caps to validation / test data."""
        for col in X.columns:
            if col in self._col_caps:
                lo, hi = self._col_caps[col]
                X[col] = X[col].clip(lo, hi)
        return X

    # ── Step 9: Class balancing ───────────────────────────────────────────────

    def _balance(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple:
        """Random undersampling of the majority class (training only)."""
        counts = y.value_counts()
        minority_label = counts.idxmin()
        majority_label = counts.idxmax()
        minority_n     = counts.min()

        print(f'\n  [9] Balancing — before: {dict(counts)}')

        maj_idx = y[y == majority_label].index
        min_idx = y[y == minority_label].index

        rng = np.random.default_rng(self.random_state)
        sampled_maj = rng.choice(maj_idx, size=minority_n, replace=False)

        keep_idx = np.concatenate([sampled_maj, min_idx.values])
        rng.shuffle(keep_idx)

        X_bal = X.loc[keep_idx].reset_index(drop=True)
        y_bal = y.loc[keep_idx].reset_index(drop=True)

        print(f'       Balancing — after : {dict(y_bal.value_counts())}')
        return X_bal, y_bal

    # ── Step 10: Retrieval leakage inspection ─────────────────────────────────

    def _inspect_retrieval_leakage(
        self,
        X_train: pd.DataFrame, y_train: pd.Series,
        X_test:  pd.DataFrame, y_test:  pd.Series,
    ) -> dict:
        """
        Check whether retrieval/context features are strongly correlated
        with the label in a way that could indicate dataset-specific leakage.

        Returns a report dict with per-feature statistics.
        """
        report = {}
        print('\n  [10] Retrieval feature leakage inspection:')

        for col in RETRIEVAL_COLS:
            if col not in X_train.columns:
                continue

            # Mean value per class in training
            train_combined = X_train.copy()
            train_combined['label'] = y_train.values
            means = train_combined.groupby('label')[col].mean()

            phish_mean = means.get(1, float('nan'))
            legit_mean = means.get(0, float('nan'))
            diff       = abs(phish_mean - legit_mean)

            # Flag if difference is large (potential leakage)
            flag = 'POTENTIAL LEAKAGE' if diff > 0.3 else 'OK'

            report[col] = {
                'phishing_mean': round(float(phish_mean), 4),
                'legit_mean'   : round(float(legit_mean), 4),
                'abs_diff'     : round(float(diff), 4),
                'flag'         : flag,
            }
            print(f'      {col:<30} phish={phish_mean:.3f} '
                  f'legit={legit_mean:.3f} diff={diff:.3f}  [{flag}]')

        return report

    # ── Save / load artefacts ─────────────────────────────────────────────────

    def save(self, data: dict, output_dir: str = 'data/processed') -> None:
        """Save all preprocessed data and fitted artefacts."""
        os.makedirs(output_dir, exist_ok=True)

        data['X_train'].to_csv(
            os.path.join(output_dir, 'X_train.csv'), index=False)
        data['X_val'].to_csv(
            os.path.join(output_dir, 'X_val.csv'), index=False)
        data['X_test'].to_csv(
            os.path.join(output_dir, 'X_test.csv'), index=False)
        data['y_train'].to_csv(
            os.path.join(output_dir, 'y_train.csv'), index=False)
        data['y_val'].to_csv(
            os.path.join(output_dir, 'y_val.csv'), index=False)
        data['y_test'].to_csv(
            os.path.join(output_dir, 'y_test.csv'), index=False)

        # Raw (unscaled) test/val for leakage analysis
        if 'X_test_raw' in data:
            data['X_test_raw'].to_csv(
                os.path.join(output_dir, 'X_test_raw.csv'), index=False)
        if 'X_val_raw' in data:
            data['X_val_raw'].to_csv(
                os.path.join(output_dir, 'X_val_raw.csv'), index=False)

        # Scaler
        joblib.dump(data['scaler'],
                    os.path.join(output_dir, 'scaler.pkl'))

        # Feature names
        with open(os.path.join(output_dir, 'feature_names.txt'), 'w') as f:
            f.write('\n'.join(data['feature_names']))

        # Leakage report
        with open(os.path.join(output_dir, 'leakage_report.json'), 'w') as f:
            json.dump(data['leakage_report'], f, indent=2)

        # Preprocessing metadata
        meta = {
            'num_features'   : len(data['feature_names']),
            'feature_names'  : data['feature_names'],
            'train_samples'  : len(data['X_train']),
            'val_samples'    : len(data['X_val']),
            'test_samples'   : len(data['X_test']),
            'train_class_dist': data['y_train'].value_counts().to_dict(),
            'val_class_dist'  : data['y_val'].value_counts().to_dict(),
            'test_class_dist' : data['y_test'].value_counts().to_dict(),
            'col_medians'    : self._col_medians,
            'col_caps'       : {k: list(v) for k, v in self._col_caps.items()},
        }
        with open(os.path.join(output_dir, 'preprocessing_meta.json'), 'w') as f:
            json.dump(meta, f, indent=2, default=str)

        print(f'\n  Saved preprocessed data → {output_dir}/')
        print(f'  Files: X_train/val/test.csv, y_train/val/test.csv, '
              f'scaler.pkl, feature_names.txt, leakage_report.json')
