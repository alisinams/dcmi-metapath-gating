"""Download the Amazon Reviews 2023 Video Games ratings (5-core) and metadata into data/amazon/.

TLS certificates are verified; set DOWNLOAD_INSECURE=1 only behind a TLS-intercepting proxy.
"""
import os
import requests

VERIFY = os.environ.get('DOWNLOAD_INSECURE', '0') != '1'
if not VERIFY:
    import urllib3
    urllib3.disable_warnings()
D = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'amazon'))
os.makedirs(D, exist_ok=True)
for u in ['https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/rating_only/Video_Games.csv.gz',
          'https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Video_Games.jsonl.gz']:
    f = os.path.join(D, u.split('/')[-1])
    if os.path.exists(f) and os.path.getsize(f) > 1000:
        print('skip', f); continue
    with requests.get(u, stream=True, timeout=600, verify=VERIFY) as r:
        r.raise_for_status()
        with open(f, 'wb') as fh:
            for ch in r.iter_content(1 << 20):
                fh.write(ch)
    print('done', f, os.path.getsize(f))
