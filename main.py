"""
main.py
=======
PhishGuard - Phishing Website Detection System
Component 1: Data Collection, Feature Extraction & Preprocessing

Pipeline
--------
1. Load URLs from CSV source
2. Extract 30 features per URL (lexical + domain/DNS + content + context)
3. Preprocess with domain-aware splitting and leakage prevention
4. Save all artefacts for Component 2 (model training)

Usage
-----
    python main.py --mode csv --csv-path collected_urls.csv
    python main.py --mode csv --csv-path collected_urls.csv --no-content
    python main.py --mode csv --csv-path collected_urls.csv --num-urls 5000

Arguments
---------
    --mode          : Data source mode (csv | collect)
    --csv-path      : Path to labelled URL CSV (required for csv mode)
    --num-urls      : Max URLs per class to use (default: 5000)
    --no-content    : Skip page fetching (URL-only features; faster but less accurate)
    --workers       : Parallel extraction threads (default: 8)
    --output-dir    : Root output directory (default: data)

Author  : MSc Cybersecurity Project - University of the West of Scotland
Project : PhishGuard
"""

import os
import sys
import time
import logging
import argparse
import warnings
from datetime import datetime

import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ── Directory structure ───────────────────────────────────────────────────────

DIRS = [
    'data/raw',
    'data/features',
    'data/processed',
    'data/holdout',
    'models',
    'results/plots',
    'results/reports',
]


def create_dirs(base: str = '.') -> None:
    for d in DIRS:
        os.makedirs(os.path.join(base, d), exist_ok=True)
    print('  [OK] Directory structure created')


# ── Step 1: Load URLs ─────────────────────────────────────────────────────────

def load_urls_from_csv(
    csv_path: str,
    num_urls: int = 5000,
    output_dir: str = 'data/raw',
) -> tuple:
    """
    Load and balance URLs from a labelled CSV file.

    Parameters
    ----------
    csv_path  : str  — path to CSV with 'url' and 'label' columns
    num_urls  : int  — maximum URLs per class
    output_dir: str  — where to save raw URL lists

    Returns
    -------
    (phishing_urls, legitimate_urls, all_urls, all_labels)
    """
    print(f'\n  Loading URLs from: {csv_path}')
    df = pd.read_csv(csv_path)

    # Normalise column names
    df.columns = df.columns.str.strip().str.lower()
    if 'url' not in df.columns or 'label' not in df.columns:
        raise ValueError(
            f"CSV must contain 'url' and 'label' columns. "
            f"Found: {list(df.columns)}"
        )

    df['url']   = df['url'].astype(str).str.strip()
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    df = df[df['label'].isin([0, 1])]

    # Remove exact URL duplicates
    df = df.drop_duplicates(subset=['url'])

    phishing   = df[df['label'] == 1].sample(
        n=min(num_urls, (df['label'] == 1).sum()),
        random_state=42
    )
    legitimate = df[df['label'] == 0].sample(
        n=min(num_urls, (df['label'] == 0).sum()),
        random_state=42
    )

    combined = pd.concat([phishing, legitimate]).sample(
        frac=1, random_state=42
    ).reset_index(drop=True)

    print(f'  Phishing   : {len(phishing)}')
    print(f'  Legitimate : {len(legitimate)}')
    print(f'  Total      : {len(combined)}')

    # Save raw URL lists
    os.makedirs(output_dir, exist_ok=True)
    phishing.to_csv(
        os.path.join(output_dir, 'phishing_urls.csv'), index=False)
    legitimate.to_csv(
        os.path.join(output_dir, 'legitimate_urls.csv'), index=False)

    return (
        phishing['url'].tolist(),
        legitimate['url'].tolist(),
        combined['url'].tolist(),
        combined['label'].tolist(),
    )


# ── Step 2: Feature extraction ────────────────────────────────────────────────

def extract_features(
    urls: list,
    labels: list,
    fetch_content: bool = True,
    workers: int = 8,
    output_dir: str = 'data/features',
) -> pd.DataFrame:
    """
    Extract 30 features from all URLs.

    Parameters
    ----------
    urls          : list of str
    labels        : list of int
    fetch_content : bool — whether to fetch pages and perform DNS/WHOIS
    workers       : int  — parallel threads
    output_dir    : str  — where to save extracted features

    Returns
    -------
    pd.DataFrame with 30 feature columns + 'url' + 'label'
    """
    from feature_extractor import FeatureExtractor

    mode_label = 'full (lexical + domain + content)' if fetch_content \
                 else 'URL-only (lexical features only)'
    print(f'\n  Extraction mode : {mode_label}')
    print(f'  URLs to process : {len(urls)}')
    print(f'  Workers         : {workers}')

    extractor = FeatureExtractor(fetch_content=fetch_content)

    t0 = time.time()
    feature_list = extractor.extract_batch(urls, labels, max_workers=workers)
    elapsed = time.time() - t0

    df = pd.DataFrame(feature_list)

    # Attach URL column
    df.insert(0, 'url', urls)

    print(f'\n  Extraction time : {elapsed:.1f}s')
    print(f'  Shape           : {df.shape}')
    print(f'  Errors          : {extractor.error_count}')

    # Verify all 30 features are present
    from feature_extractor import FeatureExtractor as FE
    missing = [f for f in FE.FEATURE_NAMES if f not in df.columns]
    if missing:
        logger.warning('Missing features after extraction: %s', missing)
        for col in missing:
            df[col] = 0

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'extracted_features.csv')
    df.to_csv(out_path, index=False)
    print(f'  Saved → {out_path}')

    return df


# ── Step 3: Preprocessing ─────────────────────────────────────────────────────

def preprocess(
    df: pd.DataFrame,
    output_dir: str = 'data/processed',
) -> dict:
    """
    Run the full preprocessing pipeline.

    Parameters
    ----------
    df         : pd.DataFrame — extracted features
    output_dir : str          — where to save processed data

    Returns
    -------
    dict — preprocessed data splits and artefacts
    """
    from preprocessor import DataPreprocessor

    preprocessor = DataPreprocessor(
        random_state=42,
        val_size=0.10,
        test_size=0.15,
        balance=True,
    )

    data = preprocessor.preprocess(df)
    preprocessor.save(data, output_dir)

    return data


# ── Step 4: External holdout preparation ──────────────────────────────────────

def prepare_external_holdout(
    verified_path: str,
    training_urls: set,
    training_domains: set,
    output_dir: str = 'data/holdout',
    max_records: int = 2000,
    fetch_content: bool = True,
    workers: int = 8,
) -> None:
    """
    Prepare an external phishing holdout from verified_online.csv.

    Only records whose URLs and registered domains do not appear in
    the training set are included.

    Parameters
    ----------
    verified_path    : str  — path to verified_online.csv
    training_urls    : set  — URLs used in training (for overlap check)
    training_domains : set  — registered domains used in training
    output_dir       : str  — where to save holdout features
    max_records      : int  — maximum holdout records to extract
    fetch_content    : bool — whether to fetch pages
    workers          : int  — parallel threads
    """
    import tldextract as tld

    print(f'\n  Preparing external phishing holdout from: {verified_path}')
    df = pd.read_csv(verified_path)
    df.columns = df.columns.str.strip().str.lower()

    if 'url' not in df.columns:
        logger.warning('verified_online.csv has no url column — skipping holdout')
        return

    df['url'] = df['url'].astype(str).str.strip()
    df = df.drop_duplicates(subset=['url'])

    # Filter out URLs / domains already in training
    df['_domain'] = df['url'].apply(
        lambda u: tld.extract(u).registered_domain.lower()
    )
    df = df[~df['url'].isin(training_urls)]
    df = df[~df['_domain'].isin(training_domains)]

    print(f'  Records after overlap removal: {len(df)}')

    if len(df) == 0:
        print('  No holdout records available after overlap removal.')
        return

    # Sample
    holdout = df.sample(
        n=min(max_records, len(df)), random_state=42
    ).reset_index(drop=True)

    holdout_urls   = holdout['url'].tolist()
    holdout_labels = [1] * len(holdout_urls)   # all phishing

    print(f'  Extracting features for {len(holdout_urls)} holdout URLs...')
    from feature_extractor import FeatureExtractor
    extractor = FeatureExtractor(fetch_content=fetch_content)
    feature_list = extractor.extract_batch(
        holdout_urls, holdout_labels, max_workers=workers
    )

    holdout_df = pd.DataFrame(feature_list)
    holdout_df.insert(0, 'url', holdout_urls)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'external_phishing_holdout.csv')
    holdout_df.to_csv(out_path, index=False)
    print(f'  Saved → {out_path}  ({len(holdout_df)} records)')


# ── Argument parser ───────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='PhishGuard — Component 1: Data & Feature Pipeline'
    )
    parser.add_argument(
        '--mode', choices=['csv', 'collect'], default='csv',
        help='Data source mode (default: csv)'
    )
    parser.add_argument(
        '--csv-path', type=str, default='collected_urls.csv',
        help='Path to labelled URL CSV (url, label columns)'
    )
    parser.add_argument(
        '--verified-path', type=str, default='verified_online.csv',
        help='Path to verified_online.csv for external holdout'
    )
    parser.add_argument(
        '--num-urls', type=int, default=5000,
        help='Max URLs per class (default: 5000)'
    )
    parser.add_argument(
        '--no-content', action='store_true',
        help='Skip page fetching — URL-only features (faster)'
    )
    parser.add_argument(
        '--workers', type=int, default=8,
        help='Parallel extraction threads (default: 8)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='data',
        help='Root output directory (default: data)'
    )
    parser.add_argument(
        '--skip-holdout', action='store_true',
        help='Skip external holdout preparation'
    )
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print('=' * 65)
    print('  PHISHGUARD — COMPONENT 1: DATA & FEATURE PIPELINE')
    print(f'  Started : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'  Mode    : {args.mode}')
    print('=' * 65)

    t_start = time.time()
    fetch_content = not args.no_content

    # ── Create directories ────────────────────────────────────────────────────
    create_dirs()

    # ── Step 1: Load URLs ─────────────────────────────────────────────────────
    print('\n' + '=' * 65)
    print('  STEP 1: LOADING DATA')
    print('=' * 65)

    if args.mode == 'csv':
        phishing_urls, legit_urls, all_urls, all_labels = load_urls_from_csv(
            csv_path=args.csv_path,
            num_urls=args.num_urls,
            output_dir=os.path.join(args.output_dir, 'raw'),
        )
    else:
        print('  [collect mode] Not yet implemented — use --mode csv')
        sys.exit(1)

    # ── Step 2: Feature extraction ────────────────────────────────────────────
    print('\n' + '=' * 65)
    print('  STEP 2: FEATURE EXTRACTION')
    print('=' * 65)

    features_df = extract_features(
        urls=all_urls,
        labels=all_labels,
        fetch_content=fetch_content,
        workers=args.workers,
        output_dir=os.path.join(args.output_dir, 'features'),
    )

    # ── Step 3: Preprocessing ─────────────────────────────────────────────────
    print('\n' + '=' * 65)
    print('  STEP 3: PREPROCESSING')
    print('=' * 65)

    data = preprocess(
        df=features_df,
        output_dir=os.path.join(args.output_dir, 'processed'),
    )

    # ── Step 4: External holdout ──────────────────────────────────────────────
    if not args.skip_holdout and os.path.exists(args.verified_path):
        print('\n' + '=' * 65)
        print('  STEP 4: EXTERNAL PHISHING HOLDOUT')
        print('=' * 65)

        import tldextract as tld
        training_urls    = set(all_urls)
        training_domains = set(
            tld.extract(u).registered_domain.lower() for u in all_urls
        )

        prepare_external_holdout(
            verified_path=args.verified_path,
            training_urls=training_urls,
            training_domains=training_domains,
            output_dir=os.path.join(args.output_dir, 'holdout'),
            max_records=2000,
            fetch_content=fetch_content,
            workers=args.workers,
        )
    else:
        if args.skip_holdout:
            print('\n  [STEP 4] External holdout skipped (--skip-holdout)')
        else:
            print(f'\n  [STEP 4] {args.verified_path} not found — holdout skipped')

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print('\n' + '=' * 65)
    print('  COMPONENT 1 COMPLETE')
    print(f'  Total time : {elapsed:.1f}s')
    print('=' * 65)
    print(f'\n  Output directories:')
    print(f'    Raw URLs   → {args.output_dir}/raw/')
    print(f'    Features   → {args.output_dir}/features/')
    print(f'    Processed  → {args.output_dir}/processed/')
    print(f'    Holdout    → {args.output_dir}/holdout/')
    print(f'\n  Leakage report → {args.output_dir}/processed/leakage_report.json')
    print(f'\n  Next step: python train.py')


if __name__ == '__main__':
    main()
