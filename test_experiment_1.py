import csv
import os
import re
import sys
import threading
import time
import tkinter as tk
import unicodedata
from tkinter import filedialog, messagebox, ttk
from urllib.parse import urlparse

RANDOM_STATE = 42
FEATURE_COUNT = 12
DATASET_PAIRINGS = {
    1: ('malicious_phish.csv', 'tranco_64X5X.csv'),
    2: ('urlhaus.abuse.ch.csv', 'top-1m.csv'),
    3: ('verified_online.csv', 'majestic_million.csv'),
}


def extract_12_features(url):
    url, parsed = _parse_url(url)
    domain = parsed.netloc if parsed.netloc else parsed.path.split('/')[0]
    return {
        'url_length': len(url),
        'domain_length': len(domain),
        'count_dots': url.count('.'),
        'count_hyphens': url.count('-'),
        'count_at': url.count('@'),
        'count_question': url.count('?'),
        'count_equal': url.count('='),
        'count_slash': url.count('/'),
        'has_ip': 1 if domain.replace('.', '').isdigit() else 0,
        'is_https': 1 if parsed.scheme.lower() == 'https' else 0,
        'digit_count': sum(c.isdigit() for c in url),
        'has_CYRILLIC': int(any(
            'CYRILLIC' in unicodedata.name(character, '')
            for character in url
        )),
    }


extract_11_features = extract_12_features


def _parse_url(url):
    value = str(url).strip()
    if not value:
        raise ValueError('URL cannot be empty.')
    if not re.match(r'^[A-Za-z][A-Za-z0-9+.-]*://', value):
        value = f'https://{value}'
    original_value = value
    try:
        parsed = urlparse(value)
        if parsed.hostname and any(ord(character) > 127 for character in parsed.hostname):
            ascii_host = parsed.hostname.encode('idna').decode('ascii')
            authority = ascii_host
            if parsed.port:
                authority += f':{parsed.port}'
            value = parsed._replace(netloc=authority).geturl()
            parsed = urlparse(value)
        hostname = parsed.hostname
    except ValueError as error:
        raise ValueError(f'Invalid URL: {error}') from error
    if not hostname:
        raise ValueError('URL must contain a valid host name.')
    return original_value, parsed


def _filter_parseable_urls(urls):
    valid_urls = []
    rejected_count = 0
    for url in urls:
        try:
            _parse_url(url)
        except ValueError:
            rejected_count += 1
            continue
        valid_urls.append(url)
    return valid_urls, rejected_count


class StaticRuleEngine:
    def explain_single(self, features):
        reasons = []
        if features['url_length'] > 75:
            reasons.append('URL is longer than 75 characters')
        if features['count_at'] > 0:
            reasons.append('URL contains an @ symbol')
        if features['has_ip'] == 1:
            reasons.append('Domain appears to be an IP address')
        if features['is_https'] == 0:
            reasons.append('URL does not use HTTPS')
        if features['count_hyphens'] > 2:
            reasons.append('URL contains more than two hyphens')
        return len(reasons), reasons

    def predict(self, X_df):
        return [self.predict_single(row) for _, row in X_df.iterrows()]

    def predict_single(self, features):
        score, _ = self.explain_single(features)
        return 1 if score >= 2 else 0

def load_and_preprocess_data():
    return load_dataset_pair(1)


def load_dataset_pair(experiment_id):
    import pandas as pd

    if experiment_id not in DATASET_PAIRINGS:
        raise ValueError(f'Unknown experiment pairing: {experiment_id}')

    phishing_file, legitimate_file = DATASET_PAIRINGS[experiment_id]
    data_dir = os.path.dirname(os.path.abspath(__file__))
    phishing_path = os.path.join(data_dir, phishing_file)
    legitimate_path = os.path.join(data_dir, legitimate_file)

    if experiment_id == 1:
        phishing_df = pd.read_csv(phishing_path)
        phishing_df = phishing_df[
            phishing_df['type'].str.strip().str.lower() == 'phishing'
        ]
        phishing_urls = phishing_df['url'].dropna().astype(str).tolist()
    elif experiment_id == 2:
        # URLhaus stores its column header on a line prefixed with '#'.
        with open(
            phishing_path,
            'r',
            encoding='utf-8-sig',
            errors='replace',
            newline='',
        ) as csv_file:
            header_line = next(csv_file).lstrip('#').strip()
            urlhaus_header = next(csv.reader([header_line]))
        phishing_df = pd.read_csv(
            phishing_path,
            skiprows=2,
            names=[column.strip() for column in urlhaus_header],
        )
        phishing_urls = phishing_df['url'].dropna().astype(str).tolist()
    else:
        phishing_df = pd.read_csv(phishing_path)
        phishing_urls = phishing_df['url'].dropna().astype(str).tolist()

    if experiment_id == 3:
        legitimate_df = pd.read_csv(legitimate_path)
        legitimate_urls = [
            _ensure_url_scheme(domain)
            for domain in legitimate_df['Domain'].dropna().astype(str)
        ]
    else:
        legitimate_df = pd.read_csv(legitimate_path, header=None)
        domain_col = 1 if legitimate_df.shape[1] > 1 else 0
        legitimate_urls = [
            _ensure_url_scheme(domain)
            for domain in legitimate_df[domain_col].dropna().astype(str)
        ]

    phishing_urls, rejected_phishing = _filter_parseable_urls(phishing_urls)
    legitimate_urls, rejected_legitimate = _filter_parseable_urls(legitimate_urls)
    if rejected_phishing or rejected_legitimate:
        print(
            f'Experiment {experiment_id}: excluded {rejected_phishing:,} malformed '
            f'phishing URLs and {rejected_legitimate:,} malformed legitimate URLs.'
        )

    phishing_urls = _deduplicate_urls(phishing_urls)
    phishing_keys = {normalize_url(url) for url in phishing_urls}
    legitimate_urls = [
        url for url in _deduplicate_urls(legitimate_urls)
        if normalize_url(url) not in phishing_keys
    ]
    sample_size = min(len(phishing_urls), len(legitimate_urls))
    if sample_size < 5:
        raise ValueError(
            f'Experiment {experiment_id} needs at least five distinct URLs in each class.'
        )

    import random
    rng = random.Random(RANDOM_STATE)
    phishing_urls = rng.sample(phishing_urls, sample_size)
    legitimate_urls = rng.sample(legitimate_urls, sample_size)
    combined_urls = phishing_urls + legitimate_urls
    combined_labels = [1] * sample_size + [0] * sample_size
    paired_rows = list(zip(combined_urls, combined_labels))
    rng.shuffle(paired_rows)
    combined_urls, combined_labels = zip(*paired_rows)

    df = pd.DataFrame({'url': combined_urls, 'label': combined_labels})
    X = pd.DataFrame(df['url'].apply(extract_11_features).tolist())
    y = df['label'].values
    return X, y


def _ensure_url_scheme(value):
    url = str(value).strip()
    if not url:
        return ''
    return url if '://' in url else f'https://{url}'


def _deduplicate_urls(urls):
    unique_urls = []
    seen = set()
    for url in urls:
        url = str(url).strip()
        key = normalize_url(url)
        if key and key not in seen:
            seen.add(key)
            unique_urls.append(url)
    return unique_urls


def calculate_metrics(y_true, y_pred):
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN, FP, FN, TP = cm.ravel()
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    fpr = FP / (FN + TP) if (FN + TP) > 0 else 0
    fnr = FN / (FN + TP) if (FN + TP) > 0 else 0
    return acc, prec, rec, f1, fpr, fnr

def evaluate_experiment(experiment_id=1):
    import numpy as np
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.model_selection import KFold

    X, y = load_dataset_pair(experiment_id)
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    model_factories = {
        'Rule-based': StaticRuleEngine,
        'LightGBM': lambda: lgb.LGBMClassifier(
            random_state=RANDOM_STATE,
            verbose=-1,
        ),
        'XGBoost': lambda: xgb.XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric='logloss',
        ),
    }
    metric_names = (
        'accuracy', 'precision', 'recall', 'f1', 'fpr', 'fnr', 'latency'
    )
    fold_metrics = {
        name: {metric: [] for metric in metric_names}
        for name in model_factories
    }
    for train_idx, test_idx in kf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        for name, model_factory in model_factories.items():
            model = model_factory()
            if name != 'Rule-based':
                model.fit(X_train, y_train)
            start_time = time.perf_counter()
            preds = model.predict(X_test)
            end_time = time.perf_counter()
            latency_per_url = ((end_time - start_time) / len(X_test)) * 1000
            accuracy, precision, recall, f1, fpr, fnr = calculate_metrics(
                y_test,
                preds,
            )
            for metric, value in zip(
                ('accuracy', 'precision', 'recall', 'f1', 'fpr', 'fnr', 'latency'),
                (accuracy, precision, recall, f1, fpr, fnr, latency_per_url),
            ):
                fold_metrics[name][metric].append(value)

    phishing_file, legitimate_file = DATASET_PAIRINGS[experiment_id]
    dataset_name = f'{phishing_file} + {legitimate_file}'
    summary = []
    for name, metrics in fold_metrics.items():
        row = {
            'experiment': experiment_id,
            'dataset': dataset_name,
            'model': name,
            'samples': len(y),
        }
        for metric, fold_values in metrics.items():
            row[metric] = float(np.mean(fold_values))
            row[f'{metric}_std'] = float(np.std(fold_values))
        summary.append(row)
    return summary


def run_experiment(experiment_id=1):
    import pandas as pd

    summary = evaluate_experiment(experiment_id)
    display_rows = []
    for row in summary:
        display_rows.append({
            'Engine': row['model'],
            'Accuracy (mean ± SD)': (
                f"{row['accuracy'] * 100:.2f}% ± "
                f"{row['accuracy_std'] * 100:.2f}%"
            ),
            'Precision (mean ± SD)': (
                f"{row['precision'] * 100:.2f}% ± "
                f"{row['precision_std'] * 100:.2f}%"
            ),
            'Recall (mean ± SD)': (
                f"{row['recall'] * 100:.2f}% ± "
                f"{row['recall_std'] * 100:.2f}%"
            ),
            'F1-Score (mean ± SD)': (
                f"{row['f1'] * 100:.2f}% ± "
                f"{row['f1_std'] * 100:.2f}%"
            ),
            'False Positive Rate (mean ± SD)': (
                f"{row['fpr'] * 100:.2f}% ± "
                f"{row['fpr_std'] * 100:.2f}%"
            ),
            'False Negative Rate (mean ± SD)': (
                f"{row['fnr'] * 100:.2f}% ± "
                f"{row['fnr_std'] * 100:.2f}%"
            ),
            'Latency': f"{row['latency']:.6f} ± {row['latency_std']:.6f} ms/URL",
        })
    print(f'\nEXPERIMENT {experiment_id}: {summary[0]["dataset"]}')
    print(f'Samples: {summary[0]["samples"]:,}; folds: 5; URL features: {FEATURE_COUNT}')
    print(pd.DataFrame(display_rows).to_string(index=False))
    return summary


def run_all_experiments():
    return [
        row
        for experiment_id in DATASET_PAIRINGS
        for row in run_experiment(experiment_id)
    ]


URL_COLUMNS = {'url', 'urls', 'domain', 'website', 'link', 'host'}
LABEL_COLUMNS = {
    'type',
    'label',
    'class',
    'category',
    'threat',
    'classification',
    'target',
    'is_malicious',
    'is_phishing',
}
PHISHING_LABELS = {
    '1', 'true', 'phish', 'phishing', 'malicious', 'malware',
    'malware_download', 'defacement', 'scam', 'fraud', 'spam', 'suspicious',
}
LEGITIMATE_LABELS = {
    '0', 'false', 'benign', 'legitimate', 'legit', 'safe', 'clean',
    'good', 'normal', 'non-phishing', 'nonphishing',
}


def normalize_url(url):
    value = str(url).strip()
    if not value:
        return ''
    try:
        _, parsed = _parse_url(value)
        hostname = parsed.hostname
        port = f':{parsed.port}' if parsed.port else ''
        path = parsed.path.rstrip('/')
        normalized = f'{hostname.lower()}{port}{path}'
        if parsed.query:
            normalized += f'?{parsed.query}'
        if parsed.fragment:
            normalized += f'#{parsed.fragment}'
        return normalized
    except ValueError:
        return value.lower().rstrip('/')


def _label_from_value(value):
    label = str(value).strip().lower().replace(' ', '_')
    if label in PHISHING_LABELS:
        return 1
    if label in LEGITIMATE_LABELS:
        return 0
    return None


def read_csv_urls(file_path, default_label=None):
    """Read URL entries from a CSV, using recognized labels or a chosen default."""
    with open(file_path, 'r', encoding='utf-8-sig', errors='replace', newline='') as csv_file:
        rows = csv.reader(line for line in csv_file if not line.lstrip().startswith('#'))
        first_row = next(rows, None)
        if not first_row:
            raise ValueError('The selected CSV file is empty.')

        headings = [cell.strip().lower() for cell in first_row]
        has_header = any(
            heading in URL_COLUMNS or heading in LABEL_COLUMNS
            for heading in headings
        )

        if has_header:
            url_index = next(
                (i for i, heading in enumerate(headings) if heading in URL_COLUMNS),
                None,
            )
            label_index = next(
                (i for i, heading in enumerate(headings) if heading in LABEL_COLUMNS),
                None,
            )
            if url_index is None:
                raise ValueError(
                    'Could not find a URL column. Use a column named url, domain, website, link, or host.'
                )
            data_rows = rows
        else:
            url_index = next(
                (i for i, cell in enumerate(first_row) if '.' in cell and not cell.isdigit()),
                0,
            )
            label_index = None
            data_rows = iter([first_row])
            data_rows = _chain_rows(data_rows, rows)

        entries = []
        found_label = False
        for row in data_rows:
            if url_index >= len(row):
                continue
            url = row[url_index].strip()
            if not url or url.lower() in URL_COLUMNS:
                continue
            label = _label_from_value(row[label_index]) if label_index is not None and label_index < len(row) else None
            if label is not None:
                found_label = True
            elif default_label is not None:
                label = default_label
            entries.append((url, label))

    if not entries:
        if found_label:
            raise ValueError('No URL rows with recognizable phishing or legitimate labels were found.')
        raise ValueError('No URLs could be read from this CSV file.')
    return entries, found_label


def _chain_rows(first_rows, remaining_rows):
    yield from first_rows
    yield from remaining_rows


class UrlCheckerApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Experiment 1 - URL Checker')
        self.root.geometry('1100x760')
        self.root.minsize(850, 500)
        self.url_labels = {}
        self.sources = []
        self.rule_engine = StaticRuleEngine()
        self.scan_models = None

        content_frame = ttk.Frame(root)
        content_frame.pack(fill='both', expand=True)
        content_canvas = tk.Canvas(content_frame, highlightthickness=0)
        vertical_scrollbar = ttk.Scrollbar(
            content_frame,
            orient='vertical',
            command=content_canvas.yview,
        )
        content_canvas.configure(yscrollcommand=vertical_scrollbar.set)
        vertical_scrollbar.pack(side='right', fill='y')
        content_canvas.pack(side='left', fill='both', expand=True)

        main = ttk.Frame(content_canvas, padding=12)
        canvas_window = content_canvas.create_window(
            (0, 0),
            window=main,
            anchor='nw',
        )

        def update_scroll_region(_event=None):
            content_canvas.configure(scrollregion=content_canvas.bbox('all'))

        def resize_content(event):
            content_canvas.itemconfigure(canvas_window, width=event.width)

        main.bind('<Configure>', update_scroll_region)
        content_canvas.bind('<Configure>', resize_content)

        ttk.Label(
            main,
            text='URL Phishing Checker',
            font=('Segoe UI', 18, 'bold'),
        ).pack(anchor='w')
        ttk.Label(
            main,
            text='Check an individual URL or scan a CSV file for suspicious links using 12 URL features.',
            wraplength=710,
        ).pack(anchor='w', pady=(4, 14))

        input_row = ttk.Frame(main)
        input_row.pack(fill='x')
        self.url_entry = ttk.Entry(input_row)
        self.url_entry.pack(side='left', fill='x', expand=True)
        self.url_entry.bind('<Return>', lambda _event: self.check_url())
        self.check_button = ttk.Button(input_row, text='Check URL', command=self.check_url)
        self.check_button.pack(side='left', padx=(8, 0))

        single_results = ttk.LabelFrame(main, text='Single URL predictions', padding=8)
        single_results.pack(fill='x', pady=(12, 14))
        self.single_result_labels = {}
        for model_name in ('Rule-based', 'XGBoost', 'LightGBM'):
            row = ttk.Frame(single_results)
            row.pack(fill='x', pady=2)
            ttk.Label(row, text=f'{model_name}:', width=12).pack(side='left')
            result = ttk.Label(
                row,
                text='Enter a URL and click Check URL.',
                wraplength=900,
                justify='left',
            )
            result.pack(side='left', fill='x', expand=True)
            self.single_result_labels[model_name] = result
        self.single_details_label = ttk.Label(
            single_results,
            text='',
            wraplength=1000,
            justify='left',
        )
        self.single_details_label.pack(anchor='w', pady=(4, 0))

        scan_header = ttk.Frame(main)
        scan_header.pack(fill='x')
        ttk.Label(scan_header, text='CSV scan results', font=('Segoe UI', 11, 'bold')).pack(side='left')
        self.scan_button = ttk.Button(scan_header, text='Scan CSV...', command=self.scan_csv)
        self.scan_button.pack(side='right')

        self.scan_summary = ttk.Label(main, text='Choose a CSV file to scan its URLs.')
        self.scan_summary.pack(anchor='w', pady=(6, 4))

        results_frame = ttk.Frame(main)
        results_frame.pack(fill='both', expand=True)
        self.results_table = ttk.Treeview(
            results_frame,
            columns=('url', 'rules', 'xgboost', 'lightgbm', 'details'),
            show='headings',
            height=10,
        )
        self.results_table.heading('url', text='URL')
        self.results_table.heading('rules', text='Rule-based')
        self.results_table.heading('xgboost', text='XGBoost')
        self.results_table.heading('lightgbm', text='LightGBM')
        self.results_table.heading('details', text='Rule details')
        self.results_table.column('url', width=300, minwidth=180)
        self.results_table.column('rules', width=150, minwidth=130, stretch=False)
        self.results_table.column('xgboost', width=120, minwidth=110, stretch=False)
        self.results_table.column('lightgbm', width=120, minwidth=110, stretch=False)
        self.results_table.column('details', width=350, minwidth=200)
        self.results_table.tag_configure('phishing', foreground='#b00020')
        self.results_table.tag_configure('legitimate', foreground='#176b35')
        scrollbar = ttk.Scrollbar(results_frame, orient='vertical', command=self.results_table.yview)
        self.results_table.configure(yscrollcommand=scrollbar.set)
        self.results_table.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        horizontal_scrollbar = ttk.Scrollbar(
            main,
            orient='horizontal',
            command=self.results_table.xview,
        )
        self.results_table.configure(xscrollcommand=horizontal_scrollbar.set)
        horizontal_scrollbar.pack(fill='x')

        ttk.Label(
            main,
            text=(
                'Scan CSV runs the five-fold benchmark directly on the uploaded file. '
                'Labeled URLs are required for objective classification metrics; '
                'unlabeled files use the majority vote of the three scan results '
                'as the comparison reference.'
            ),
            wraplength=710,
        ).pack(anchor='w', pady=(10, 0))

        benchmark_results = ttk.LabelFrame(
            main,
            text='Five-fold benchmark results',
            padding=8,
        )
        benchmark_results.pack(fill='x', pady=(8, 0))
        benchmark_columns = ('metric', 'rule_based', 'xgboost', 'lightgbm')
        self.benchmark_table = ttk.Treeview(
            benchmark_results,
            columns=benchmark_columns,
            show='headings',
            height=8,
        )
        benchmark_headings = {
            'metric': 'Metric',
            'rule_based': 'Rule-based',
            'xgboost': 'XGBoost',
            'lightgbm': 'LightGBM',
        }
        for column in benchmark_columns:
            self.benchmark_table.heading(column, text=benchmark_headings[column])
            self.benchmark_table.column(column, width=125, minwidth=95, stretch=False)
        self.benchmark_table.column('metric', width=220, minwidth=180)
        self.benchmark_table.column('rule_based', width=250, minwidth=200)
        self.benchmark_table.column('xgboost', width=250, minwidth=200)
        self.benchmark_table.column('lightgbm', width=250, minwidth=200)
        self.benchmark_table.pack(fill='x', expand=True)
        benchmark_scrollbar = ttk.Scrollbar(
            benchmark_results,
            orient='horizontal',
            command=self.benchmark_table.xview,
        )
        self.benchmark_table.configure(xscrollcommand=benchmark_scrollbar.set)
        benchmark_scrollbar.pack(fill='x', pady=(4, 0))
        ttk.Label(
            benchmark_results,
            text=(
                'Accuracy = (TP + TN) / Total | '
                'Precision = TP / (TP + FP) | '
                'Recall = TP / (TP + FN) | '
                'F1-score = 2 x (P x R) / (P + R)\n'
                'FPR = FP / (FN + TP) | '
                'FNR = FN / (FN + TP) | '
                    'Latency (ms/URL) = ((T_end - T_start) / N_test) x 1000\n'
                    'Unlabeled CSV reference: majority vote of Rule-based, XGBoost, and LightGBM.'
            ),
            justify='left',
            wraplength=1000,
        ).pack(anchor='w', pady=(6, 0))

        actions = ttk.Frame(main)
        actions.pack(fill='x', pady=(10, 0))
        self.benchmark_status_label = ttk.Label(
            actions,
            text='Five-fold benchmark runs automatically after Scan CSV.',
            wraplength=800,
        )
        self.benchmark_status_label.pack(side='right', padx=(12, 0))

        content_canvas.bind_all(
            '<MouseWheel>',
            lambda event: content_canvas.yview_scroll(
                int(-event.delta / 120),
                'units',
            ),
        )

    def _get_scan_models(self):
        if self.scan_models is None:
            import lightgbm as lgb
            import xgboost as xgb

            X_train, y_train = load_and_preprocess_data()
            models = {
                'XGBoost': xgb.XGBClassifier(
                    random_state=RANDOM_STATE,
                    eval_metric='logloss',
                ),
                'LightGBM': lgb.LGBMClassifier(
                    random_state=RANDOM_STATE,
                    verbose=-1,
                ),
            }
            for model in models.values():
                model.fit(X_train, y_train)
            self.scan_models = models
        return self.scan_models

    def classify_url(self, url):
        label = self.url_labels.get(normalize_url(url))
        if label is not None:
            verdict = 'PHISHING' if label else 'LEGITIMATE'
            return verdict, 'Exact match in labeled reference CSV'

        score, reasons = self.rule_engine.explain_single(extract_11_features(url))
        verdict = 'LIKELY PHISHING' if score >= 2 else 'LIKELY LEGITIMATE'
        details = '; '.join(reasons) if reasons else 'No experiment risk rules triggered'
        return verdict, f'Heuristic ({score}/5): {details}'

    def _predict_urls(self, urls):
        import pandas as pd

        models = self._get_scan_models()
        features = pd.DataFrame(
            [extract_11_features(url) for url in urls]
        )
        predictions = {
            name: model.predict(features)
            for name, model in models.items()
        }
        results = []
        for index, url in enumerate(urls):
            rule_verdict, rule_details = self.classify_url(url)
            results.append((
                rule_verdict,
                'PHISHING' if predictions['XGBoost'][index] == 1 else 'LEGITIMATE',
                'PHISHING' if predictions['LightGBM'][index] == 1 else 'LEGITIMATE',
                rule_details,
            ))
        return results

    def scan_csv(self):
        file_path = filedialog.askopenfilename(
            title='Choose a CSV file to scan',
            filetypes=[('CSV files', '*.csv'), ('All files', '*.*')],
        )
        if not file_path:
            return

        self.scan_button.config(state='disabled')
        self.benchmark_status_label.config(text='Running five-fold evaluation...')
        self.scan_summary.config(text='Loading training data and scanning URLs...')
        threading.Thread(
            target=self._scan_csv_worker,
            args=(file_path,),
            daemon=True,
        ).start()

    def _scan_csv_worker(self, file_path):
        try:
            entries, _ = read_csv_urls(file_path)
            valid_entries = []
            rejected_count = 0
            for url, label in entries:
                try:
                    _parse_url(url)
                except ValueError:
                    rejected_count += 1
                    continue
                valid_entries.append((url, label))
            valid_urls = [url for url, _label in valid_entries]
            if not valid_urls:
                raise ValueError('No valid URLs were found in this CSV file.')
            benchmark_results = self._benchmark_scanned_csv(valid_entries)
            predictions = self._predict_urls(valid_urls)
            results = [(url, *prediction) for url, prediction in zip(valid_urls, predictions)]
        except (
            OSError,
            csv.Error,
            ValueError,
            ImportError,
            RuntimeError,
            TypeError,
        ) as error:
            self.root.after(0, self._scan_failed, file_path, error)
            return
        except Exception as error:
            self.root.after(0, self._scan_failed, file_path, error)
            return

        self.root.after(
            0,
            self._show_benchmark_results,
            self.benchmark_status_label,
            None,
            benchmark_results,
        )
        self.root.after(
            0,
            self._show_scan_results,
            file_path,
            results,
            rejected_count,
        )

    def _scan_failed(self, file_path, error):
        self.scan_button.config(state='normal')
        if isinstance(error, ImportError):
            message = (
                'The three-model scan needs pandas, LightGBM, and XGBoost installed '
                'in the Python environment running this app.\n\n'
                f'Interpreter:\n{sys.executable}\n\n'
                f'Import error:\n{error}\n\n'
                'Install them with:\n'
                f'"{sys.executable}" -m pip install pandas lightgbm xgboost'
            )
        else:
            message = str(error)
        messagebox.showerror(
            'Could not scan CSV',
            f'{os.path.basename(file_path)}\n\n{message}',
            parent=self.root,
        )
        self.scan_summary.config(text='CSV scan failed.')

    def _show_scan_results(self, file_path, results, rejected_count=0):
        self.scan_button.config(state='normal')
        self.results_table.delete(*self.results_table.get_children())
        counts = {'Rule-based': 0, 'XGBoost': 0, 'LightGBM': 0}
        for url, rule_verdict, xgb_verdict, lgb_verdict, details in results:
            for name, verdict in (
                ('Rule-based', rule_verdict),
                ('XGBoost', xgb_verdict),
                ('LightGBM', lgb_verdict),
            ):
                counts[name] += int('PHISHING' in verdict)
            self.results_table.insert(
                '',
                'end',
                values=(
                    url,
                    rule_verdict,
                    xgb_verdict,
                    lgb_verdict,
                    details,
                ),
                tags=('phishing' if 'PHISHING' in rule_verdict else 'legitimate',),
            )

        self.scan_summary.config(
            text=(
                f'{os.path.basename(file_path)} — scanned {len(results):,} URLs. '
                f'Predictions: Rule-based {counts["Rule-based"]:,}, '
                f'XGBoost {counts["XGBoost"]:,}, LightGBM {counts["LightGBM"]:,}. '
                f'Skipped {rejected_count:,} malformed rows.'
            )
        )

    def check_url(self):
        url = self.url_entry.get().strip()
        if not url:
            messagebox.showwarning('URL required', 'Enter a URL to check.', parent=self.root)
            return

        self.check_button.config(state='disabled')
        self.benchmark_status_label.config(text='Running five-fold evaluation...')
        for label in self.single_result_labels.values():
            label.config(text='Checking...', foreground='')
        self.single_details_label.config(text='')
        threading.Thread(
            target=self._check_url_worker,
            args=(url,),
            daemon=True,
        ).start()

    def _check_url_worker(self, url):
        try:
            prediction = self._predict_urls([url])[0]
        except Exception as error:
            self.root.after(0, self._single_check_failed, error)
            return
        self.root.after(0, self._show_single_result, prediction)

    def _single_check_failed(self, error):
        self.check_button.config(state='normal')
        if isinstance(error, ImportError):
            message = (
                'The three-model checker needs pandas, LightGBM, and XGBoost installed '
                'in the Python environment running this app.\n\n'
                f'Interpreter:\n{sys.executable}\n\n'
                f'Import error:\n{error}\n\n'
                'Install them with:\n'
                f'"{sys.executable}" -m pip install pandas lightgbm xgboost'
            )
        else:
            message = str(error)
        messagebox.showerror(
            'Could not check URL',
            message,
            parent=self.root,
        )
        for label in self.single_result_labels.values():
            label.config(text='Check failed.', foreground='#b00020')

    def _show_single_result(self, prediction):
        self.check_button.config(state='normal')
        rule_verdict, xgb_verdict, lgb_verdict, details = prediction
        for model_name, verdict in (
            ('Rule-based', rule_verdict),
            ('XGBoost', xgb_verdict),
            ('LightGBM', lgb_verdict),
        ):
            self.single_result_labels[model_name].config(
                text=verdict,
                foreground='#b00020' if 'PHISHING' in verdict else '#176b35',
            )
        self.single_details_label.config(text=f'Rule details: {details}')

    def _benchmark_scanned_csv(self, entries):
        import numpy as np
        import pandas as pd
        import lightgbm as lgb
        import xgboost as xgb
        from sklearn.model_selection import KFold

        labels = [label for _url, label in entries]
        urls = [url for url, _label in entries]
        features = pd.DataFrame([extract_11_features(url) for url in urls])
        has_complete_labels = all(label is not None for label in labels)

        metric_names = (
            ('Accuracy', 0),
            ('Precision', 1),
            ('Recall', 2),
            ('F1-score', 3),
            ('False Positive Rate', 4),
            ('False Negative Rate', 5),
        )
        model_names = ('Rule-based', 'XGBoost', 'LightGBM')
        if not has_complete_labels:
            scan_models = self._get_scan_models()
            predictions = {}
            latency_values = {}
            rule_start = time.perf_counter()
            predictions['Rule-based'] = np.asarray(
                self.rule_engine.predict(features),
                dtype=int,
            )
            latency_values['Rule-based'] = (
                (time.perf_counter() - rule_start) / len(features) * 1000
            )
            for name in ('XGBoost', 'LightGBM'):
                model = scan_models[name]
                start = time.perf_counter()
                predictions[name] = np.asarray(model.predict(features), dtype=int)
                latency_values[name] = (
                    (time.perf_counter() - start) / len(features) * 1000
                )
            reference = (
                predictions['Rule-based']
                + predictions['XGBoost']
                + predictions['LightGBM']
                >= 2
            ).astype(int)
            metric_labels = (
                'Accuracy',
                'Precision',
                'Recall',
                'F1-score',
                'False Positive Rate (FPR)',
                'False Negative Rate (FNR)',
            )
            metric_values = {
                model_name: calculate_metrics(reference, predictions[model_name])
                for model_name in model_names
            }
            rows = [
                {
                    'metric': metric_name,
                    **{
                        model_name.lower().replace('-', '_'): (
                            f'{metric_values[model_name][index] * 100:.2f}%'
                        )
                        for model_name in model_names
                    },
                }
                for index, metric_name in enumerate(metric_labels)
            ]
            rows.append({
                'metric': 'Inference Latency',
                'rule_based': f'{latency_values["Rule-based"]:.6f} ms/URL',
                'xgboost': f'{latency_values["XGBoost"]:.6f} ms/URL',
                'lightgbm': f'{latency_values["LightGBM"]:.6f} ms/URL',
            })
            return rows

        y_true = np.asarray(labels, dtype=int)
        kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        fold_values = {
            model_name: {name: [] for name, _ in metric_names}
            for model_name in model_names
        }
        latency_values = {model_name: [] for model_name in model_names}
        for train_index, test_index in kfold.split(features):
            X_train, X_test = features.iloc[train_index], features.iloc[test_index]
            y_train, y_test = y_true[train_index], y_true[test_index]
            fold_models = [
                self.rule_engine,
                xgb.XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss'),
                lgb.LGBMClassifier(random_state=RANDOM_STATE, verbose=-1),
            ]
            predictions = []
            for model_name, model in zip(model_names, fold_models):
                if model_name != 'Rule-based':
                    model.fit(X_train, y_train)
                start = time.perf_counter()
                predictions.append(model.predict(X_test))
                latency_values[model_name].append(
                    (time.perf_counter() - start) / len(X_test) * 1000
                )
            for model_name, predictions_for_model in zip(model_names, predictions):
                metric_values = calculate_metrics(y_test, predictions_for_model)
                for name, metric_index in metric_names:
                    fold_values[model_name][name].append(metric_values[metric_index])
        rows = []
        for name, _ in metric_names:
            rows.append({
                'metric': name,
                **{
                    model_name.lower().replace('-', '_'): (
                        f'{np.mean(fold_values[model_name][name]) * 100:.2f}% ± '
                        f'{np.std(fold_values[model_name][name]) * 100:.2f}%'
                    )
                    for model_name in model_names
                },
            })
        rows.append({
            'metric': 'Inference Latency',
            **{
                model_name.lower().replace('-', '_'): (
                    f'{np.mean(latency_values[model_name]):.6f} ± '
                    f'{np.std(latency_values[model_name]):.6f} ms/URL'
                )
                for model_name in model_names
            },
        })
        return rows

    def _benchmark_failed(self, status, error):
        status.config(text='Benchmark failed.')
        if isinstance(error, ImportError):
            message = (
                'The benchmark needs pandas, NumPy, scikit-learn, LightGBM, '
                'and XGBoost installed in the Python environment running this app.'
                '\n\nInstall them with:\npip install numpy pandas scikit-learn lightgbm xgboost'
            )
        else:
            message = str(error)
        messagebox.showerror(
            'Could not run benchmark',
            message,
            parent=self.root,
        )

    def _show_benchmark_results(self, status, table, results):
        self.benchmark_table.delete(*self.benchmark_table.get_children())
        for row in results:
            self.benchmark_table.insert(
                '',
                'end',
                values=(
                    row['metric'],
                    row['rule_based'],
                    row['xgboost'],
                    row['lightgbm'],
                ),
            )
        status.config(
            text='Benchmark metrics calculated from the selected CSV scan.'
        )


if __name__ == '__main__':
    root = tk.Tk()
    app = UrlCheckerApp(root)
    root.mainloop()