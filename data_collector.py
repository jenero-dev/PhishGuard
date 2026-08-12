"""
data_collector.py
=================
Phishing Website Detection System - Component 1
Collects phishing URLs from PhishTank and OpenPhish,
and legitimate URLs from curated sources.

MSc Project - University of the West of Scotland
"""

import os
import csv
import json
import time
import logging
import requests
import pandas as pd
from datetime import datetime
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataCollector:
    """
    Collects phishing and legitimate URLs from multiple sources.

    Sources:
        Phishing URLs:
            - PhishTank (verified phishing URLs)
            - OpenPhish (community-driven phishing feed)
            - Local CSV file (custom additions)

        Legitimate URLs:
            - Cisco Umbrella Top 1 Million
            - Majestic Million
            - Tranco List (combination of multiple top lists)
            - Local CSV file (custom additions)

    Parameters
    ----------
    output_dir : str, optional (default='data/raw')
        Directory to save collected URLs.
    phishtank_api_key : str, optional (default=None)
        API key for PhishTank (optional but recommended for higher rate limits).
    """

    def __init__(self, output_dir='data/raw', phishtank_api_key=None):
        self.output_dir = output_dir
        self.phishtank_api_key = phishtank_api_key
        os.makedirs(output_dir, exist_ok=True)

        # Storage
        self.phishing_urls = []
        self.legitimate_urls = []

    # ================================================================
    # PHISHING URL COLLECTION
    # ================================================================

    def collect_phishtank(self, limit=5000):
        """
        Collect verified phishing URLs from PhishTank.

        PhishTank provides a downloadable JSON feed of verified phishing URLs.
        URL: https://data.phishtank.com/data/online-valid.json

        Parameters
        ----------
        limit : int
            Maximum number of phishing URLs to collect.

        Returns
        -------
        list
            List of phishing URLs.
        """
        logger.info("Collecting phishing URLs from PhishTank...")

        try:
            url = "http://data.phishtank.com/data/online-valid.json"
            if self.phishtank_api_key:
                url = f"http://data.phishtank.com/data/{self.phishtank_api_key}/online-valid.json"

            headers = {
                'User-Agent': 'phishtank/MSc-Phishing-Detection-Project'
            }

            response = requests.get(url, headers=headers, timeout=60)

            if response.status_code == 200:
                data = response.json()
                urls = []
                for entry in data[:limit]:
                    if entry.get('verified') == 'yes' or entry.get('verified') == True:
                        urls.append(entry['url'])

                self.phishing_urls.extend(urls)
                logger.info(f"Collected {len(urls)} verified phishing URLs from PhishTank")
                return urls
            else:
                logger.warning(f"PhishTank returned status code: {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"Error collecting from PhishTank: {str(e)}")
            return []

    def collect_openphish(self, limit=5000):
        """
        Collect phishing URLs from OpenPhish free feed.

        OpenPhish provides a simple text file with phishing URLs.
        URL: https://openphish.com/feed.txt

        Parameters
        ----------
        limit : int
            Maximum number of phishing URLs to collect.

        Returns
        -------
        list
            List of phishing URLs.
        """
        logger.info("Collecting phishing URLs from OpenPhish...")

        try:
            url = "https://openphish.com/feed.txt"
            response = requests.get(url, timeout=30)

            if response.status_code == 200:
                urls = [line.strip() for line in response.text.strip().split('\n')
                        if line.strip().startswith('http')]
                urls = urls[:limit]
                self.phishing_urls.extend(urls)
                logger.info(f"Collected {len(urls)} phishing URLs from OpenPhish")
                return urls
            else:
                logger.warning(f"OpenPhish returned status code: {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"Error collecting from OpenPhish: {str(e)}")
            return []

    def load_phishing_from_csv(self, filepath):
        """
        Load phishing URLs from a local CSV file.

        Useful for adding manually curated phishing URLs or
        URLs from datasets like the UCI Phishing Dataset.

        Parameters
        ----------
        filepath : str
            Path to CSV file. Must have a column named 'url'.

        Returns
        -------
        list
            List of phishing URLs.
        """
        logger.info(f"Loading phishing URLs from {filepath}...")

        try:
            df = pd.read_csv(filepath)
            if 'url' in df.columns:
                urls = df['url'].dropna().tolist()
            elif 'URL' in df.columns:
                urls = df['URL'].dropna().tolist()
            else:
                # Assume first column is URLs
                urls = df.iloc[:, 0].dropna().tolist()

            self.phishing_urls.extend(urls)
            logger.info(f"Loaded {len(urls)} phishing URLs from CSV")
            return urls

        except Exception as e:
            logger.error(f"Error loading CSV: {str(e)}")
            return []

    # ================================================================
    # LEGITIMATE URL COLLECTION
    # ================================================================

    def collect_tranco(self, limit=5000):
        """
        Collect legitimate URLs from the Tranco List.

        The Tranco list is a research-oriented top sites ranking that
        combines Alexa, Umbrella, Majestic, and Chrome UX data.
        URL: https://tranco-list.eu/

        Parameters
        ----------
        limit : int
            Maximum number of legitimate URLs to collect.

        Returns
        -------
        list
            List of legitimate URLs (with https:// prefix).
        """
        logger.info("Collecting legitimate URLs from Tranco List...")

        try:
            # Download the latest Tranco list
            # First, get the latest list ID
            list_url = "https://tranco-list.eu/top-1m.csv.zip"
            response = requests.get(list_url, timeout=60)

            if response.status_code == 200:
                import zipfile
                import io

                z = zipfile.ZipFile(io.BytesIO(response.content))
                csv_filename = z.namelist()[0]

                with z.open(csv_filename) as f:
                    lines = f.read().decode('utf-8').strip().split('\n')

                urls = []
                for line in lines[:limit]:
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        domain = parts[1].strip()
                        urls.append(f"https://{domain}")

                self.legitimate_urls.extend(urls)
                logger.info(f"Collected {len(urls)} legitimate URLs from Tranco")
                return urls
            else:
                logger.warning(f"Tranco returned status code: {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"Error collecting from Tranco: {str(e)}")
            return []

    def collect_majestic_million(self, limit=5000):
        """
        Collect legitimate URLs from Majestic Million.

        URL: https://majestic.com/reports/majestic-million-csv

        Parameters
        ----------
        limit : int
            Maximum number of legitimate URLs to collect.

        Returns
        -------
        list
            List of legitimate URLs.
        """
        logger.info("Collecting legitimate URLs from Majestic Million...")

        try:
            url = "https://downloads.majestic.com/majestic_million.csv"
            response = requests.get(url, timeout=60, stream=True)

            if response.status_code == 200:
                urls = []
                reader = csv.DictReader(response.iter_lines(decode_unicode=True))

                for i, row in enumerate(reader):
                    if i >= limit:
                        break
                    domain = row.get('Domain', '').strip()
                    if domain:
                        urls.append(f"https://{domain}")

                self.legitimate_urls.extend(urls)
                logger.info(f"Collected {len(urls)} legitimate URLs from Majestic")
                return urls
            else:
                logger.warning(f"Majestic returned status code: {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"Error collecting from Majestic: {str(e)}")
            return []

    def load_legitimate_from_csv(self, filepath):
        """
        Load legitimate URLs from a local CSV file.

        Parameters
        ----------
        filepath : str
            Path to CSV file.

        Returns
        -------
        list
            List of legitimate URLs.
        """
        logger.info(f"Loading legitimate URLs from {filepath}...")

        try:
            df = pd.read_csv(filepath)
            if 'url' in df.columns:
                urls = df['url'].dropna().tolist()
            elif 'URL' in df.columns:
                urls = df['URL'].dropna().tolist()
            else:
                urls = df.iloc[:, 0].dropna().tolist()

            self.legitimate_urls.extend(urls)
            logger.info(f"Loaded {len(urls)} legitimate URLs from CSV")
            return urls

        except Exception as e:
            logger.error(f"Error loading CSV: {str(e)}")
            return []

    # ================================================================
    # GENERATE SYNTHETIC/FALLBACK DATA (For development/testing)
    # ================================================================

    def generate_sample_data(self, num_phishing=2000, num_legitimate=2000):
        """
        Generate sample URLs for development and testing purposes.

        This provides realistic-looking URLs when API sources are unavailable.
        These are synthetic URLs based on common phishing patterns.

        WARNING: Only use this for initial development. The final model
        should be trained on REAL phishing and legitimate URLs.

        Parameters
        ----------
        num_phishing : int
            Number of phishing-like URLs to generate.
        num_legitimate : int
            Number of legitimate-like URLs to generate.
        """
        import random
        import string

        logger.info("Generating sample/fallback URL data...")

        # --- Phishing URL patterns ---
        phishing_patterns = [
            "http://{random_domain}.com/{path}/login.php?user={random_str}",
            "http://192.168.{octet}.{octet}/{path}/secure-update.html",
            "http://{brand}-{word}.{tld}/{path}/verify.php",
            "http://{random_subdomain}.{random_domain}.{tld}/account/{random_str}",
            "http://secure-{brand}-login.{tld}/{path}/index.php",
            "http://{brand}.{random_domain}.{tld}/signin/?token={random_str}",
            "https://{random_domain}.{suspicious_tld}/{path}/update-info.html",
            "http://{brand}-support.{tld}/credential/restore?id={random_str}",
            "http://{random_domain}.com/{path}/{path}/{path}/login.html",
            "http://www.{brand}-{word}.{random_domain}.com/authenticate.php",
        ]

        brands = ['paypal', 'amazon', 'apple', 'microsoft', 'netflix',
                   'facebook', 'google', 'chase', 'wellsfargo', 'bankofamerica',
                   'instagram', 'linkedin', 'dropbox', 'adobe', 'outlook']

        words = ['login', 'secure', 'verify', 'update', 'account', 'support',
                 'service', 'confirm', 'alert', 'notification', 'billing',
                 'payment', 'restore', 'unlock', 'validation']

        suspicious_tlds = ['tk', 'ml', 'ga', 'cf', 'gq', 'xyz', 'top',
                           'buzz', 'click', 'icu', 'monster']

        normal_tlds = ['com', 'net', 'org', 'info', 'biz', 'co']

        paths = ['wp-content', 'wp-admin', 'images', 'assets', 'public',
                 'files', 'download', 'docs', 'data', 'uploads', 'includes']

        def random_str(length=8):
            return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

        def random_domain():
            return random_str(random.randint(5, 12))

        phishing_urls = []
        for _ in range(num_phishing):
            pattern = random.choice(phishing_patterns)
            url = pattern.format(
                random_domain=random_domain(),
                random_str=random_str(random.randint(6, 20)),
                random_subdomain=random_str(random.randint(3, 8)),
                brand=random.choice(brands),
                word=random.choice(words),
                tld=random.choice(normal_tlds + suspicious_tlds),
                suspicious_tld=random.choice(suspicious_tlds),
                path=random.choice(paths),
                octet=random.randint(1, 254)
            )
            phishing_urls.append(url)

        self.phishing_urls.extend(phishing_urls)

        # --- Legitimate URL patterns ---
        legit_domains = [
            'google.com', 'youtube.com', 'facebook.com', 'amazon.com',
            'wikipedia.org', 'twitter.com', 'instagram.com', 'linkedin.com',
            'reddit.com', 'netflix.com', 'microsoft.com', 'apple.com',
            'github.com', 'stackoverflow.com', 'medium.com', 'bbc.co.uk',
            'cnn.com', 'nytimes.com', 'theguardian.com', 'reuters.com',
            'spotify.com', 'zoom.us', 'slack.com', 'notion.so',
            'figma.com', 'canva.com', 'shopify.com', 'wordpress.com',
            'tumblr.com', 'pinterest.com', 'quora.com', 'yahoo.com',
            'bing.com', 'duckduckgo.com', 'dropbox.com', 'twitch.tv',
            'adobe.com', 'salesforce.com', 'oracle.com', 'ibm.com',
            'paypal.com', 'ebay.com', 'walmart.com', 'target.com',
            'bestbuy.com', 'homedepot.com', 'costco.com', 'kroger.com',
            'chase.com', 'bankofamerica.com', 'wellsfargo.com', 'citi.com',
            'harvard.edu', 'mit.edu', 'stanford.edu', 'oxford.ac.uk',
            'who.int', 'un.org', 'nasa.gov', 'nih.gov', 'cdc.gov',
            'bbc.com', 'espn.com', 'weather.com', 'imdb.com',
        ]

        legit_paths = [
            '', '/', '/about', '/contact', '/products', '/services',
            '/blog', '/news', '/help', '/support', '/login', '/signup',
            '/pricing', '/features', '/docs', '/api', '/careers',
            '/privacy', '/terms', '/faq', '/search', '/settings',
        ]

        legitimate_urls = []
        for _ in range(num_legitimate):
            domain = random.choice(legit_domains)
            path = random.choice(legit_paths)
            scheme = 'https'
            url = f"{scheme}://{domain}{path}"
            legitimate_urls.append(url)

        self.legitimate_urls.extend(legitimate_urls)

        logger.info(f"Generated {len(phishing_urls)} sample phishing URLs")
        logger.info(f"Generated {len(legitimate_urls)} sample legitimate URLs")

    # ================================================================
    # DATA CLEANING AND DEDUPLICATION
    # ================================================================

    def clean_and_deduplicate(self):
        """
        Remove duplicates and invalid URLs from both collections.

        Returns
        -------
        tuple
            (num_phishing, num_legitimate) after cleaning.
        """
        logger.info("Cleaning and deduplicating URLs...")

        # Remove duplicates
        self.phishing_urls = list(set(self.phishing_urls))
        self.legitimate_urls = list(set(self.legitimate_urls))

        # Remove any URL that appears in both lists (ambiguous)
        phishing_set = set(self.phishing_urls)
        legitimate_set = set(self.legitimate_urls)
        overlap = phishing_set & legitimate_set

        if overlap:
            logger.warning(f"Found {len(overlap)} URLs in both lists. Removing from legitimate list.")
            self.legitimate_urls = [u for u in self.legitimate_urls if u not in overlap]

        # Basic URL validation
        def is_valid_url(url):
            try:
                result = url.startswith(('http://', 'https://'))
                return result and len(url) > 10
            except:
                return False

        self.phishing_urls = [u for u in self.phishing_urls if is_valid_url(u)]
        self.legitimate_urls = [u for u in self.legitimate_urls if is_valid_url(u)]

        logger.info(f"After cleaning: {len(self.phishing_urls)} phishing, "
                     f"{len(self.legitimate_urls)} legitimate URLs")

        return len(self.phishing_urls), len(self.legitimate_urls)

    # ================================================================
    # BALANCE DATASET
    # ================================================================

    def balance_dataset(self):
        """
        Balance the dataset so both classes have equal representation.
        Uses undersampling of the majority class.
        """
        import random

        min_size = min(len(self.phishing_urls), len(self.legitimate_urls))

        if len(self.phishing_urls) > min_size:
            self.phishing_urls = random.sample(self.phishing_urls, min_size)
        if len(self.legitimate_urls) > min_size:
            self.legitimate_urls = random.sample(self.legitimate_urls, min_size)

        logger.info(f"Balanced dataset: {min_size} URLs per class (Total: {min_size * 2})")

    # ================================================================
    # SAVE TO FILE
    # ================================================================

    def save_urls(self, filename='collected_urls.csv'):
        """
        Save collected URLs to a CSV file with labels.

        Parameters
        ----------
        filename : str
            Name of the output CSV file.

        Returns
        -------
        str
            Path to the saved file.
        """
        filepath = os.path.join(self.output_dir, filename)

        data = []
        for url in self.phishing_urls:
            data.append({'url': url, 'label': 1})  # 1 = phishing
        for url in self.legitimate_urls:
            data.append({'url': url, 'label': 0})  # 0 = legitimate

        df = pd.DataFrame(data)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
        df.to_csv(filepath, index=False)

        logger.info(f"Saved {len(df)} URLs to {filepath}")
        logger.info(f"  Phishing: {len(self.phishing_urls)}")
        logger.info(f"  Legitimate: {len(self.legitimate_urls)}")

        return filepath

    def get_summary(self):
        """Print a summary of collected data."""
        print("\n" + "=" * 50)
        print("DATA COLLECTION SUMMARY")
        print("=" * 50)
        print(f"  Phishing URLs:   {len(self.phishing_urls):,}")
        print(f"  Legitimate URLs: {len(self.legitimate_urls):,}")
        print(f"  Total URLs:      {len(self.phishing_urls) + len(self.legitimate_urls):,}")
        print(f"  Output Dir:      {self.output_dir}")
        print("=" * 50)
