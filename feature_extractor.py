"""
feature_extractor.py
====================
PhishGuard - Phishing Website Detection System
Component 1: Feature Extraction

Extracts exactly 30 features across four categories:
    A. Lexical / URL structure      :  8 features
    B. Domain / DNS infrastructure  :  8 features
    C. Robust page-content          : 10 features
    D. Retrieval / context          :  4 features

Design principles:
    - www is treated as a conventional host prefix, NOT a suspicious subdomain
    - Failed WHOIS / DNS lookups are represented as "unavailable", not "phishing"
    - Failed page fetches are represented as "unavailable", not "phishing"
    - No single feature alone determines the result
    - Content and domain features carry more weight than lexical features

Author  : MSc Cybersecurity Project - University of the West of Scotland
Project : PhishGuard
"""

import re
import math
import time
import socket
import logging
import ipaddress
from urllib.parse import urlparse, urljoin
from collections import Counter

import requests
import tldextract
import dns.resolver
import whois
from bs4 import BeautifulSoup

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
REQUEST_TIMEOUT   = 8          # seconds for page fetch
WHOIS_TIMEOUT     = 6          # seconds for WHOIS lookup
DNS_TIMEOUT       = 4          # seconds for DNS queries
MAX_REDIRECTS     = 10         # maximum redirects to follow
MAX_CONTENT_BYTES = 500_000    # 500 KB cap on downloaded HTML

# Private / loopback ranges that must never be fetched
_PRIVATE_NETS = [
    ipaddress.ip_network('10.0.0.0/8'),
    ipaddress.ip_network('172.16.0.0/12'),
    ipaddress.ip_network('192.168.0.0/16'),
    ipaddress.ip_network('127.0.0.0/8'),
    ipaddress.ip_network('169.254.0.0/16'),
    ipaddress.ip_network('::1/128'),
    ipaddress.ip_network('fc00::/7'),
]

# Known brand names used for brand-domain mismatch detection
BRAND_NAMES = [
    'paypal', 'apple', 'google', 'microsoft', 'amazon', 'facebook',
    'netflix', 'instagram', 'linkedin', 'twitter', 'chase', 'wellsfargo',
    'citibank', 'bankofamerica', 'dropbox', 'adobe', 'dhl', 'fedex',
    'usps', 'irs', 'outlook', 'office365', 'icloud', 'yahoo', 'ebay',
    'steam', 'coinbase', 'binance', 'whatsapp', 'telegram', 'tiktok',
    'spotify', 'netflix', 'allegro', 'halifax', 'barclays', 'hsbc',
    'natwest', 'lloyds', 'santander',
]

# Credential-related terms visible in page text
CREDENTIAL_TERMS = [
    'sign in', 'signin', 'log in', 'login', 'verify', 'verify account',
    'confirm identity', 'confirm your', 'password', 'security code',
    'enter your', 'account suspended', 'unusual activity', 'verify now',
    'update your', 'validate', 'authenticate', 'unlock your account',
    'reset password', 'enter otp', 'one-time password',
]

# Sensitive input field name/type patterns
SENSITIVE_PATTERNS = re.compile(
    r'(card|cvv|cvc|otp|pin\b|ssn|social.?security|account.?number|'
    r'routing|iban|swift|passport|national.?id|tax.?id)',
    re.IGNORECASE
)

# HTTP status category mapping
def _status_category(code: int) -> int:
    """
    Map HTTP status code to a numeric category:
        1 = 2xx success
        2 = 3xx redirect
        3 = 4xx client error
        4 = 5xx server error
        0 = unknown / unavailable
    """
    if 200 <= code < 300:
        return 1
    if 300 <= code < 400:
        return 2
    if 400 <= code < 500:
        return 3
    if 500 <= code < 600:
        return 4
    return 0


# ── Safety helpers ────────────────────────────────────────────────────────────

def _is_private_ip(host: str) -> bool:
    """Return True if host resolves to a private / loopback address."""
    try:
        addr = ipaddress.ip_address(host)
        return any(addr in net for net in _PRIVATE_NETS)
    except ValueError:
        pass
    try:
        resolved = socket.gethostbyname(host)
        addr = ipaddress.ip_address(resolved)
        return any(addr in net for net in _PRIVATE_NETS)
    except Exception:
        return False


def _safe_to_fetch(url: str) -> bool:
    """Return True only if the URL is safe to fetch (not private/loopback)."""
    try:
        parsed = urlparse(url)
        host = parsed.hostname or ''
        if not host:
            return False
        if _is_private_ip(host):
            return False
        return True
    except Exception:
        return False


# ── Domain normalisation ──────────────────────────────────────────────────────

def _normalise_www(subdomain: str) -> str:
    """
    Remove the leading 'www' component from a subdomain string.
    'www'        -> ''
    'www.login'  -> 'login'
    'login'      -> 'login'
    """
    parts = [p for p in subdomain.split('.') if p]
    if parts and parts[0].lower() == 'www':
        parts = parts[1:]
    return '.'.join(parts)


def _registered_domain(url: str) -> str:
    """Return the registered domain (e.g. 'example.com') for a URL."""
    ext = tldextract.extract(url)
    return (ext.registered_domain or ext.domain or '').lower()


# ── Shannon entropy ───────────────────────────────────────────────────────────

def _shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    freq = Counter(text)
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


# ═════════════════════════════════════════════════════════════════════════════
# FeatureExtractor
# ═════════════════════════════════════════════════════════════════════════════

class FeatureExtractor:
    """
    Extracts 30 features from a URL.

    Feature categories
    ------------------
    A  Lexical / URL structure      (8)
    B  Domain / DNS infrastructure  (8)
    C  Robust page-content          (10)
    D  Retrieval / context          (4)

    Usage
    -----
    extractor = FeatureExtractor(fetch_content=True)
    features  = extractor.extract_features("https://example.com")
    """

    # Authoritative ordered feature list — must match training exactly
    FEATURE_NAMES = [
        # A — Lexical
        'url_length',
        'digit_ratio',
        'special_char_ratio',
        'url_entropy',
        'has_ip_address',
        'has_at_symbol',
        'has_non_standard_port',
        'uses_http',
        # B — Domain / DNS
        'domain_age_days',
        'registration_length_days',
        'domain_expiry_proximity_days',
        'whois_data_available',
        'has_a_or_aaaa_record',
        'has_ns_record',
        'has_mx_record',
        'num_name_servers',
        # C — Page content
        'num_forms',
        'has_password_field',
        'has_identity_field',
        'has_sensitive_input',
        'has_external_form_action',
        'has_insecure_form_action',
        'credential_language_present',
        'brand_domain_mismatch',
        'num_iframes',
        'hidden_or_suspicious_iframe',
        # D — Retrieval / context
        'page_fetch_success',
        'http_status_category',
        'redirect_count',
        'final_domain_changed',
    ]

    def __init__(self, fetch_content: bool = True):
        """
        Parameters
        ----------
        fetch_content : bool
            If True, the extractor will attempt to fetch the page and
            perform DNS / WHOIS lookups.  Set to False for fast URL-only
            extraction (categories A only; B–D will be filled with
            sentinel values).
        """
        self.fetch_content = fetch_content
        self.error_count   = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def extract_features(self, url: str, label: int = None) -> dict:
        """
        Extract all 30 features from a single URL.

        Parameters
        ----------
        url   : str  — the URL to analyse
        label : int  — optional ground-truth label (1=phishing, 0=legitimate)

        Returns
        -------
        dict  — {feature_name: value, ...}  (plus 'label' if provided)
        """
        try:
            url = url.strip()
            if not url.startswith(('http://', 'https://')):
                url = 'http://' + url

            parsed     = urlparse(url)
            ext        = tldextract.extract(url)
            reg_domain = (ext.registered_domain or ext.domain or '').lower()
            # Normalise: strip www from subdomain before any domain analysis
            clean_sub  = _normalise_www(ext.subdomain or '')

            features = {}

            # ── A: Lexical features ───────────────────────────────────────
            features.update(self._lexical_features(url, parsed))

            # ── B: Domain / DNS features ──────────────────────────────────
            if self.fetch_content:
                features.update(self._domain_dns_features(reg_domain))
            else:
                features.update(self._empty_domain_dns())

            # ── C & D: Page content + retrieval context ───────────────────
            if self.fetch_content and _safe_to_fetch(url):
                page_data = self._fetch_page(url)
                features.update(self._content_features(page_data, reg_domain))
                features.update(self._retrieval_features(page_data, reg_domain))
            else:
                features.update(self._empty_content())
                features.update(self._empty_retrieval())

            # Attach label if provided
            if label is not None:
                features['label'] = int(label)

            return features

        except Exception as exc:
            self.error_count += 1
            logger.debug("Feature extraction error for %s: %s", url[:80], exc)
            return self._empty_features(label)

    def extract_batch(self, urls: list, labels: list = None,
                      max_workers: int = 8) -> list:
        """
        Extract features from a list of URLs using a thread pool.

        Parameters
        ----------
        urls        : list of str
        labels      : list of int (optional)
        max_workers : int — parallel threads

        Returns
        -------
        list of dicts
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm

        self.error_count = 0
        results = [None] * len(urls)

        def _task(idx, url):
            lbl = labels[idx] if labels else None
            return idx, self.extract_features(url, lbl)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_task, i, u): i for i, u in enumerate(urls)}
            with tqdm(total=len(urls), desc='  Extracting features') as pbar:
                for future in as_completed(futures):
                    idx, feat = future.result()
                    results[idx] = feat
                    pbar.update(1)
                    if (pbar.n % 1000) == 0 and pbar.n > 0:
                        logger.info(
                            '  [%d/%d] extracted (%d errors)',
                            pbar.n, len(urls), self.error_count
                        )

        logger.info(
            'Extraction complete: %d URLs, %d errors',
            len(results), self.error_count
        )
        return results

    # ── A: Lexical features ───────────────────────────────────────────────────

    def _lexical_features(self, url: str, parsed) -> dict:
        """8 lexical / URL-structure features."""
        # Special characters: anything that is not alphanumeric, :/.-_
        special = re.sub(r'[a-zA-Z0-9:/.\-_]', '', url)

        port = parsed.port
        non_std_port = 1 if (port and port not in (80, 443)) else 0

        return {
            'url_length'          : len(url),
            'digit_ratio'         : sum(c.isdigit() for c in url) / max(len(url), 1),
            'special_char_ratio'  : len(special) / max(len(url), 1),
            'url_entropy'         : _shannon_entropy(url),
            'has_ip_address'      : int(bool(re.match(
                                        r'^https?://\d{1,3}(?:\.\d{1,3}){3}', url))),
            'has_at_symbol'       : int('@' in url),
            'has_non_standard_port': non_std_port,
            'uses_http'           : int(parsed.scheme == 'http'),
        }

    # ── B: Domain / DNS features ──────────────────────────────────────────────

    def _domain_dns_features(self, reg_domain: str) -> dict:
        """8 domain / DNS infrastructure features."""
        whois_f = self._whois_features(reg_domain)
        dns_f   = self._dns_features(reg_domain)
        return {**whois_f, **dns_f}

    def _whois_features(self, reg_domain: str) -> dict:
        """4 WHOIS-derived features."""
        defaults = {
            'domain_age_days'            : -1,
            'registration_length_days'   : -1,
            'domain_expiry_proximity_days': -1,
            'whois_data_available'       : 0,
        }
        if not reg_domain:
            return defaults
        try:
            import datetime
            w = whois.whois(reg_domain)

            # Creation date
            created = w.creation_date
            if isinstance(created, list):
                created = created[0]

            # Expiry date
            expires = w.expiration_date
            if isinstance(expires, list):
                expires = expires[0]

            now = datetime.datetime.utcnow()

            age_days     = (now - created).days  if created else -1
            expiry_prox  = (expires - now).days  if expires else -1
            reg_length   = (expires - created).days if (created and expires) else -1

            return {
                'domain_age_days'            : max(age_days, -1),
                'registration_length_days'   : max(reg_length, -1),
                'domain_expiry_proximity_days': max(expiry_prox, -1),
                'whois_data_available'       : 1,
            }
        except Exception:
            return defaults

    def _dns_features(self, reg_domain: str) -> dict:
        """4 DNS infrastructure features."""
        defaults = {
            'has_a_or_aaaa_record': 0,
            'has_ns_record'       : 0,
            'has_mx_record'       : 0,
            'num_name_servers'    : 0,
        }
        if not reg_domain:
            return defaults

        resolver = dns.resolver.Resolver()
        resolver.lifetime = DNS_TIMEOUT

        has_a, has_ns, has_mx, num_ns = 0, 0, 0, 0

        for rtype in ('A', 'AAAA'):
            try:
                resolver.resolve(reg_domain, rtype)
                has_a = 1
                break
            except Exception:
                pass

        try:
            ns_records = resolver.resolve(reg_domain, 'NS')
            has_ns = 1
            num_ns = len(ns_records)
        except Exception:
            pass

        try:
            resolver.resolve(reg_domain, 'MX')
            has_mx = 1
        except Exception:
            pass

        return {
            'has_a_or_aaaa_record': has_a,
            'has_ns_record'       : has_ns,
            'has_mx_record'       : has_mx,
            'num_name_servers'    : num_ns,
        }

    def _empty_domain_dns(self) -> dict:
        return {
            'domain_age_days'            : -1,
            'registration_length_days'   : -1,
            'domain_expiry_proximity_days': -1,
            'whois_data_available'       : 0,
            'has_a_or_aaaa_record'       : 0,
            'has_ns_record'              : 0,
            'has_mx_record'              : 0,
            'num_name_servers'           : 0,
        }

    # ── C: Page-content features ──────────────────────────────────────────────

    def _fetch_page(self, url: str) -> dict:
        """
        Safely fetch a page and return a structured result dict.

        Returns
        -------
        dict with keys:
            success        : bool
            status_code    : int
            redirect_count : int
            final_url      : str
            html           : str
            soup           : BeautifulSoup | None
        """
        result = {
            'success'       : False,
            'status_code'   : 0,
            'redirect_count': 0,
            'final_url'     : url,
            'html'          : '',
            'soup'          : None,
        }
        try:
            headers = {
                'User-Agent': (
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                    'AppleWebKit/537.36 (KHTML, like Gecko) '
                    'Chrome/124.0.0.0 Safari/537.36'
                ),
                'Accept-Language': 'en-US,en;q=0.9',
            }
            resp = requests.get(
                url,
                headers=headers,
                timeout=REQUEST_TIMEOUT,
                allow_redirects=True,
                stream=True,
                verify=False,       # many phishing sites have invalid certs
            )
            # Read up to MAX_CONTENT_BYTES
            content = b''
            for chunk in resp.iter_content(chunk_size=8192):
                content += chunk
                if len(content) >= MAX_CONTENT_BYTES:
                    break

            html = content.decode('utf-8', errors='replace')
            result.update({
                'success'       : True,
                'status_code'   : resp.status_code,
                'redirect_count': len(resp.history),
                'final_url'     : resp.url,
                'html'          : html,
                'soup'          : BeautifulSoup(html, 'html.parser'),
            })
        except requests.exceptions.SSLError:
            # Try without SSL verification already set; if still fails, mark unavailable
            result['status_code'] = 0
        except requests.exceptions.ConnectionError:
            result['status_code'] = 0
        except requests.exceptions.Timeout:
            result['status_code'] = 0
        except Exception as exc:
            logger.debug("Page fetch error: %s", exc)
            result['status_code'] = 0

        return result

    def _content_features(self, page: dict, reg_domain: str) -> dict:
        """10 robust page-content features."""
        empty = self._empty_content()

        if not page['success'] or page['soup'] is None:
            return empty

        soup = page['soup']

        # ── Forms ────────────────────────────────────────────────────────
        forms = soup.find_all('form')
        num_forms = len(forms)

        has_password_field = 0
        has_identity_field = 0
        has_sensitive_input = 0
        has_external_form_action = 0
        has_insecure_form_action = 0

        for form in forms:
            inputs = form.find_all('input')
            for inp in inputs:
                itype = (inp.get('type') or '').lower()
                iname = (inp.get('name') or '').lower()
                iid   = (inp.get('id')   or '').lower()
                combined = f"{itype} {iname} {iid}"

                if itype == 'password':
                    has_password_field = 1

                if itype in ('email', 'tel') or any(
                    kw in combined for kw in ('email', 'user', 'phone', 'mobile')
                ):
                    has_identity_field = 1

                if SENSITIVE_PATTERNS.search(combined):
                    has_sensitive_input = 1

            # Form action analysis
            action = (form.get('action') or '').strip()
            if action and not action.startswith(('#', 'javascript')):
                try:
                    action_full = urljoin(page['final_url'], action)
                    action_domain = _registered_domain(action_full)
                    if action_domain and action_domain != reg_domain:
                        has_external_form_action = 1
                    if action_full.startswith('http://'):
                        has_insecure_form_action = 1
                except Exception:
                    pass

        # ── Credential language ───────────────────────────────────────────
        visible_text = soup.get_text(separator=' ').lower()
        credential_language_present = int(
            any(term in visible_text for term in CREDENTIAL_TERMS)
        )

        # ── Brand / domain mismatch ───────────────────────────────────────
        brand_domain_mismatch = 0
        page_title = (soup.title.string or '').lower() if soup.title else ''
        full_text  = (page_title + ' ' + visible_text[:3000]).lower()

        for brand in BRAND_NAMES:
            if brand in full_text:
                # Brand name appears on page — check if domain matches
                if brand not in reg_domain:
                    brand_domain_mismatch = 1
                break

        # ── iFrames ───────────────────────────────────────────────────────
        iframes = soup.find_all('iframe')
        num_iframes = len(iframes)

        hidden_or_suspicious_iframe = 0
        for iframe in iframes:
            style  = (iframe.get('style')  or '').lower()
            width  = (iframe.get('width')  or '').strip()
            height = (iframe.get('height') or '').strip()
            if (
                'display:none' in style.replace(' ', '') or
                'visibility:hidden' in style.replace(' ', '') or
                width  in ('0', '1') or
                height in ('0', '1')
            ):
                hidden_or_suspicious_iframe = 1
                break

        return {
            'num_forms'                 : num_forms,
            'has_password_field'        : has_password_field,
            'has_identity_field'        : has_identity_field,
            'has_sensitive_input'       : has_sensitive_input,
            'has_external_form_action'  : has_external_form_action,
            'has_insecure_form_action'  : has_insecure_form_action,
            'credential_language_present': credential_language_present,
            'brand_domain_mismatch'     : brand_domain_mismatch,
            'num_iframes'               : num_iframes,
            'hidden_or_suspicious_iframe': hidden_or_suspicious_iframe,
        }

    def _empty_content(self) -> dict:
        return {
            'num_forms'                 : 0,
            'has_password_field'        : 0,
            'has_identity_field'        : 0,
            'has_sensitive_input'       : 0,
            'has_external_form_action'  : 0,
            'has_insecure_form_action'  : 0,
            'credential_language_present': 0,
            'brand_domain_mismatch'     : 0,
            'num_iframes'               : 0,
            'hidden_or_suspicious_iframe': 0,
        }

    # ── D: Retrieval / context features ───────────────────────────────────────

    def _retrieval_features(self, page: dict, original_reg_domain: str) -> dict:
        """4 retrieval / context features."""
        if not page['success']:
            return self._empty_retrieval()

        final_domain = _registered_domain(page['final_url'])
        domain_changed = int(
            bool(final_domain) and
            bool(original_reg_domain) and
            final_domain != original_reg_domain
        )

        return {
            'page_fetch_success'  : 1,
            'http_status_category': _status_category(page['status_code']),
            'redirect_count'      : min(page['redirect_count'], MAX_REDIRECTS),
            'final_domain_changed': domain_changed,
        }

    def _empty_retrieval(self) -> dict:
        return {
            'page_fetch_success'  : 0,
            'http_status_category': 0,
            'redirect_count'      : 0,
            'final_domain_changed': 0,
        }

    # ── Fallback ──────────────────────────────────────────────────────────────

    def _empty_features(self, label=None) -> dict:
        """Return a zero-filled feature dict (used on extraction failure)."""
        features = {name: 0 for name in self.FEATURE_NAMES}
        # Sentinel -1 for WHOIS fields that are genuinely unknown
        for key in ('domain_age_days', 'registration_length_days',
                    'domain_expiry_proximity_days'):
            features[key] = -1
        if label is not None:
            features['label'] = int(label)
        return features
