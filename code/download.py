"""Download MovieLens-100K, MovieLens-1M and HetRec 2011 Last.FM from GroupLens into data/.

TLS certificates are verified.  Behind a proxy that intercepts TLS, set DOWNLOAD_INSECURE=1
to skip verification; tools/verify_splits.py still checks the rebuilt splits end to end.
"""
import os, io, zipfile
import requests

VERIFY = os.environ.get('DOWNLOAD_INSECURE', '0') != '1'
if not VERIFY:
    import urllib3
    urllib3.disable_warnings()
D = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
urls = {
    'ml-100k': 'https://files.grouplens.org/datasets/movielens/ml-100k.zip',
    'ml-1m':   'https://files.grouplens.org/datasets/movielens/ml-1m.zip',
    'lastfm':  'https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip',
}
for k, u in urls.items():
    out = os.path.join(D, k)
    if os.path.isdir(out) and os.listdir(out):
        print('skip', k); continue
    try:
        r = requests.get(u, timeout=180, verify=VERIFY)
        r.raise_for_status()
        print(k, r.status_code, len(r.content))
        z = zipfile.ZipFile(io.BytesIO(r.content))
        os.makedirs(out, exist_ok=True)
        z.extractall(out)
        print('  extracted ->', out)
    except Exception as e:
        print('ERR', k, repr(e)[:200])
