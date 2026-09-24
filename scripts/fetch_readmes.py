"""Fetch READMEs for a frozen selection, gradually and resumably, with provenance.

For each repository in the selection CSV (first column 'owner/name'), calls
GET https://api.github.com/repos/{owner}/{name}/readme and writes

    <out_dir>/<owner>,<name>.txt     the decoded README (same naming as repo_readme/)
    <out_dir>/manifest.tsv           repo, status, http, blob_sha, readme_path, fetched_at_utc

Re-running skips every repository already in the manifest with status 'ok' or 'missing'
(HTTP 404: no README), so the fetch can be stopped and resumed at any time. It honours the
primary rate limit (sleeps until X-RateLimit-Reset when X-RateLimit-Remaining hits 0) and
secondary limits (Retry-After, or 60 s on a 403/429 without it).

The token is read from GITHUB_TOKEN (environment, or the git-ignored .env); nothing is written anywhere except out_dir.

Usage: GITHUB_TOKEN=... python scripts/fetch_readmes.py enrich/top10k.csv enrich/readme_raw [--limit N]
"""

import argparse
import base64
import csv
import datetime
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("selection")
ap.add_argument("out_dir")
ap.add_argument("--limit", type=int, default=None, help="fetch at most this many this session")
a = ap.parse_args()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from env_file import load_env  # noqa: E402
load_env()
token = os.environ.get("GITHUB_TOKEN")
if not token:
    sys.exit("GITHUB_TOKEN is not set (put it in .env)")

out = Path(a.out_dir)
out.mkdir(parents=True, exist_ok=True)
manifest = out / "manifest.tsv"
done = set()
if manifest.exists():
    for line in manifest.read_text(encoding="utf-8").splitlines()[1:]:
        repo, status = line.split("\t")[:2]
        if status in ("ok", "missing"):
            done.add(repo)
else:
    manifest.write_text("repo\tstatus\thttp\tblob_sha\treadme_path\tfetched_at_utc\n")

with open(a.selection, newline="", encoding="utf-8") as f:
    todo = [row[0] for row in csv.reader(f) if row and row[0] not in done]
if a.limit is not None:
    todo = todo[:a.limit]
print(f"{len(done)} already fetched, {len(todo)} to fetch this session", flush=True)


def get(repo):
    req = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/readme",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                 "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "github-recommender-research"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


with open(manifest, "a", encoding="utf-8") as log:
    for i, repo in enumerate(todo, 1):
        while True:
            try:
                code, headers, body = get(repo)
            except (urllib.error.URLError, TimeoutError) as e:
                print(f"network error on {repo}: {e}; retrying in 30 s", flush=True)
                time.sleep(30)
                continue
            if code in (403, 429) and (headers.get("Retry-After")
                                       or headers.get("X-RateLimit-Remaining") == "0"):
                if headers.get("Retry-After"):
                    wait = int(headers["Retry-After"])
                else:
                    wait = max(0, int(headers.get("X-RateLimit-Reset", time.time() + 60)) - time.time()) + 5
                print(f"rate limited; sleeping {wait:.0f} s", flush=True)
                time.sleep(wait)
                continue
            break
        now = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"
        if code == 200:
            meta = json.loads(body)
            text = base64.b64decode(meta.get("content", "")).decode("utf-8", errors="replace")
            (out / f"{repo.replace('/', ',')}.txt").write_text(text, encoding="utf-8")
            status, sha, path = "ok", meta.get("sha", ""), meta.get("path", "")
        elif code == 404:
            status, sha, path = "missing", "", ""
        else:
            status, sha, path = "error", "", ""
        log.write(f"{repo}\t{status}\t{code}\t{sha}\t{path}\t{now}\n")
        log.flush()
        if i % 100 == 0:
            print(f"{i}/{len(todo)} (remaining quota {headers.get('X-RateLimit-Remaining')})", flush=True)
        if headers.get("X-RateLimit-Remaining") == "0":
            wait = max(0, int(headers.get("X-RateLimit-Reset", time.time())) - time.time()) + 5
            print(f"quota exhausted; sleeping {wait:.0f} s", flush=True)
            time.sleep(wait)
print("done", flush=True)
