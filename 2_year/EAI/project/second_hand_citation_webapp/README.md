# Second-Hand Citation Detector — local web app

A small Flask app that browses the output of `scripts/second_hand_citation.py`
as a filterable, explained list, instead of a CSV. Runs entirely on your
machine and makes no network calls during normal use — the one exception is
the optional one-time title-backfill step described below, which does call
the public Semantic Scholar API.

## Run it

From this `webapp/` folder:

```bash
pip install -r requirements.txt
python app.py
```

Then open **http://localhost:5050** in your browser.

> Runs on port 5050, not the more common 5000, because macOS's AirPlay
> Receiver (Monterey+) claims port 5000 by default and returns a
> confusing 403 "access denied" for anything sent to `localhost:5000` —
> using a different port sidesteps that entirely rather than requiring
> you to go disable AirPlay Receiver in System Settings.

## What it needs

- `../GNNs/results/second_hand_citation_test.csv` — the results file produced by
  `python scripts/second_hand_citation.py --split test` (already present in
  your repo, generated on 2026-07-23; delete/regenerate it if you want fresh
  numbers).
- `../graphs/cache/detail_cache_compact.sqlite3` — the Semantic Scholar detail
  cache used to resolve paper titles/authors/venues. If it's missing, the app
  still runs, just showing raw paper IDs instead of titles.

## What it shows

Each card is one flagged "second-hand citation" candidate: paper **U** cites
paper **X**, but U also cites **W**, which itself cites X — and U's citing
sentence for X reads suspiciously like W's citing sentence for X, more so
than other independent citers of X. The percentile score is how unusual that
similarity is against a background of X's other citers; the citing sentences
themselves are shown side by side as the evidence, so the flag is a **local
explanation** rather than a black-box score.

Filter by minimum percentile, sort by percentile/similarity/date, and search
across paper titles and citing-sentence text.

## Citation graph view (optional)

Each card also has "View graph around X / W / U" buttons that open a
ResearchRabbit-style node-link graph: the selected paper in the center, its
citing/cited neighbors around it, click any node to re-center on it.

This needs a one-time preprocessing step, because the full citation graph
(`graphs/cache/final_graphs.pkl`) is ~3.6GB and carries a 768-dim embedding
per node that the graph view doesn't need — loading it directly in the Flask
process isn't practical. Instead, run the indexer once, from a machine with
enough RAM to open `final_graphs.pkl` (the same place you trained
GraphSAGE/SEAL — this sandbox only has 3.8GB RAM and can't do it):

```bash
cd webapp
python build_graph_index.py
```

This writes a much smaller `webapp/data/test_graph_index.json` (structure
only: paper_id, year, and edges — no embeddings). Once that file exists,
the "View graph" buttons work; until then, they show a message telling you
to run this step, rather than failing silently.

Neighbors are capped at 15 per direction for the centered paper and 5 per
direction for second-hop papers (configurable via `MAX_NEIGHBORS_PER_DIRECTION`
/ `MAX_NEIGHBORS_PER_DIRECTION_HOP2` / `MAX_TOTAL_NODES` in `app.py`) so a
heavily-cited hub paper doesn't produce an unreadable graph — the modal tells
you how many were shown vs. how many exist. The view also draws edges
directly between any two shown papers that cite each other, not just
edges back to the centered paper, so it reads as an actual interconnected
graph rather than a plain hub-and-spoke star.

### Missing paper titles in the graph view ("Untitled" nodes)

The citation-list view's titles come from `detail_cache_compact.sqlite3`,
which covers the core SciCite corpus (every paper that appears as U/W/X in
the results CSV) — so those always resolve. The graph view, though, walks
outward into citing/cited *neighbors* of those papers, and some neighbors
were never added to that cache. Those show up as "no title cached" nodes.

To backfill them, run (after `build_graph_index.py`):

```bash
cd webapp
python fetch_missing_titles.py
```

This fetches titles/authors/venues **only** for the specific paper_ids that
actually appear in your built graph index and are missing a title — not
the whole corpus — from the public Semantic Scholar API (unauthenticated
calls are rate-limited to ~1/second; pass `--api-key` if you have one to
speed this up). Results are cached to `webapp/data/title_fallback_cache.json`
and checked automatically by the app from then on, so this only needs to
run once (safe to re-run later if you rebuild the graph index with a
larger/different neighborhood and want to backfill the new nodes too).

This app covers second-hand citation detection only. The link-prediction
"papers you might want to cite" recommender is a separate app — see
`../link_predictor_webapp/README.md`.
