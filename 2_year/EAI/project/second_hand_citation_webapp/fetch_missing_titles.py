"""Backfills paper titles/authors/venues for graph nodes not covered by
the local Semantic Scholar detail cache (detail_cache_compact.sqlite3) --
typically citing/cited neighbors reached via the graph view that were
never fetched into that cache. Fetches only the missing ids from the
public Semantic Scholar API and caches the result locally; the only step
in the app that makes outbound network calls (rate-limited to ~1 req/s
unauthenticated). Run `python build_graph_index.py` first if needed, then
`python fetch_missing_titles.py` (optionally --api-key for a higher rate
limit)."""
import argparse
import json
import os
import pickle
import sqlite3
import time

import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

GRAPH_INDEX_PATH = os.path.join(BASE_DIR, "data", "test_graph_index.json")
SQLITE_PATH = os.path.join(PROJECT_ROOT, "graphs", "cache", "detail_cache_compact.sqlite3")
FALLBACK_CACHE_PATH = os.path.join(BASE_DIR, "data", "title_fallback_cache.json")

BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
FIELDS = "title,authors,venue,year"
BATCH_SIZE = 500


def load_fallback_cache():
    if not os.path.exists(FALLBACK_CACHE_PATH):
        return {}
    with open(FALLBACK_CACHE_PATH) as f:
        return json.load(f)


def save_fallback_cache(cache):
    tmp_path = FALLBACK_CACHE_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(cache, f)
    os.replace(tmp_path, FALLBACK_CACHE_PATH)


def ids_covered_by_sqlite(ids):
    if not os.path.exists(SQLITE_PATH):
        return set()
    conn = sqlite3.connect(SQLITE_PATH)
    covered = set()
    ids = list(ids)
    for i in range(0, len(ids), 900):  # SQLite has a default ~999-variable limit per query
        chunk = ids[i:i + 900]
        placeholders = ",".join("?" for _ in chunk)
        rows = conn.execute(
            f"SELECT paper_id, payload FROM details WHERE paper_id IN ({placeholders})", chunk
        ).fetchall()
        for pid, payload in rows:
            try:
                title = pickle.loads(payload).get("title")
            except Exception:
                title = None
            if title:
                covered.add(pid)
    conn.close()
    return covered


def fetch_batch(session, ids, api_key):
    headers = {"x-api-key": api_key} if api_key else {}
    resp = session.post(BATCH_URL, params={"fields": FIELDS}, json={"ids": ids}, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-key", type=str, default=os.environ.get("S2_API_KEY"))
    parser.add_argument("--sleep", type=float, default=1.0)
    args = parser.parse_args()

    if not os.path.exists(GRAPH_INDEX_PATH):
        print(f"No graph index at {GRAPH_INDEX_PATH}. Run `python build_graph_index.py` first.")
        return

    with open(GRAPH_INDEX_PATH) as f:
        index = json.load(f)
    all_ids = set(index["nodes"].keys())
    print(f"{len(all_ids)} distinct paper_ids in the graph index")

    fallback_cache = load_fallback_cache()
    already_titled_in_sqlite = ids_covered_by_sqlite(all_ids)
    todo = [pid for pid in all_ids if pid not in already_titled_in_sqlite and pid not in fallback_cache]
    print(f"{len(already_titled_in_sqlite)} already have a title in detail_cache_compact.sqlite3")
    print(f"{len(fallback_cache)} already in the fallback cache from a previous run")
    print(f"{len(todo)} left to fetch from the Semantic Scholar API")

    if not todo:
        print("Nothing to do.")
        return

    session = requests.Session()
    for i in range(0, len(todo), BATCH_SIZE):
        batch_ids = todo[i:i + BATCH_SIZE]
        try:
            results = fetch_batch(session, batch_ids, args.api_key)
        except requests.RequestException as e:
            print(f"batch {i // BATCH_SIZE} failed ({e}); backing off 10s and retrying once...")
            time.sleep(10)
            try:
                results = fetch_batch(session, batch_ids, args.api_key)
            except requests.RequestException as e2:
                print(f"batch {i // BATCH_SIZE} failed again ({e2}); skipping this batch for now")
                continue

        for pid, r in zip(batch_ids, results):
            if r is None:
                fallback_cache[pid] = {"title": None, "authors": [], "venue": None, "year": None}
            else:
                fallback_cache[pid] = {
                    "title": r.get("title"),
                    "authors": [a.get("name") for a in (r.get("authors") or []) if a.get("name")],
                    "venue": r.get("venue") or None,
                    "year": r.get("year"),
                }

        save_fallback_cache(fallback_cache)
        print(f"[{i + len(batch_ids)}/{len(todo)}] checkpointed ({len(fallback_cache)} total cached)")
        time.sleep(args.sleep)

    print(f"Done. {len(fallback_cache)} papers in {FALLBACK_CACHE_PATH}")


if __name__ == "__main__":
    main()
