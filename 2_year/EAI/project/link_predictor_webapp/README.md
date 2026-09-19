# Link Predictor — local web app

A standalone Flask app, separate from `second_hand_citation_webapp/` (the second-hand-citation
detector). Insert a paper and see a ResearchRabbit-style view: its
citation subgraph, with the trained GraphSAGE link predictor's own
predicted-but-not-yet-existing links drawn into the same graph (dashed,
distinct color from real citations), plus those predictions split into
three lenses on the side instead of one flat ranked list — and a "Why?"
explanation for every suggestion, not a black-box score.

Runs entirely locally. At request time this app never touches
`final_graphs.pkl`, `torch`, `sklearn`, or the sqlite metadata cache
directly — only the small precomputed files described below.

## 1. Build the recommender index (one-time, offline)

Because computing embeddings requires loading the full citation graph
(`graphs/cache/final_graphs.pkl`, ~3.6GB) and the model checkpoint with
`torch`/`torch_geometric`, this step needs to run once on a machine with
enough RAM — the same place you trained GraphSAGE/SEAL — in the same
conda environment used for training (needs `torch`, `torch_geometric`,
`igraph`, `scikit-learn`):

```bash
conda activate ethics
cd link_predictor_webapp
python build_recommender_index.py
```

This writes three small files to `link_predictor_webapp/data/`:

- `recommender_embeddings.npz` — a 256-dim GraphSAGE embedding for every
  paper in the test split, plus its title/authors/venue/year resolved
  once from the local Semantic Scholar detail cache (so the app itself
  never needs that cache, or the full graph, at runtime).
- `recommender_predictor.npz` — the trained `LinkPredictor` MLP's weights
  (under 1MB), re-implemented at request time with plain `numpy`
  (concat → Linear → ReLU → Linear → sigmoid) so the Flask app itself
  never needs `torch` installed.
- `recommender_explainer.npz` — a shallow CART decision tree (Symbolic
  Knowledge Extraction, course Module 2 / L4-Transparency) fit offline to
  approximate the real model's predicted probability from four
  interpretable features (cosine similarity, shared-neighbor count, year
  gap, candidate citation-graph degree), exported as plain arrays so the
  app can walk it with numpy only. Also stores its held-out fidelity
  (R² against the real model).
- `recommender_examples.json` — a curated list of query papers known to
  have at least one genuinely confident predicted link, found by scanning
  each candidate query's own 2-hop structural neighborhood (not a random
  pair anywhere in the corpus) for the real model's own top score. Used
  as landing-page starting points -- see "Why example papers?" below.

It also reuses `second_hand_citation_webapp/data/test_graph_index.json` (built by the sibling
`second_hand_citation_webapp/build_graph_index.py`) for citation-neighborhood adjacency — run
that too if you haven't already, so real edges / shared neighborhood /
buckets aren't empty. Not required for the app to run otherwise.

## 2. Run the app

```bash
pip install -r requirements.txt
python app.py
```

Then open **http://localhost:5051** in your browser. Runs on a different
port than `second_hand_citation_webapp/` (5050) so both apps can run at the same time.

## What it shows

Search for a paper by title (or paste an exact paper ID). The page then
shows two things side by side:

**The subgraph** (left) — the query paper centered, its real 1-hop
citation neighbors (solid lines), and the model's top predicted links
(dashed lines, one per bucket item currently shown) drawn into the same
graph. Click any node to recenter the whole view on it.

**Three buckets** (right) — the model's candidate suggestions, split by
citation profile rather than shown as one flat list, to avoid always
surfacing the same most-cited papers (the "Matthew effect"):

- **All-time greats** — the top third of that query's own candidate
  shortlist by citation-graph degree.
- **Promising stars** — of the remaining candidates, published within N
  years of the real current date, where N is `RECENT_YEARS_WINDOWS`
  (`[3, 5, 7, 10, 15]`) tried smallest-first: the tightest window that
  finds at least one genuinely recent candidate for THIS query wins.
- **Hidden gems** — the rest: not recent even at the widest window tried,
  lower-degree, but still flagged relevant by the model despite never
  catching on.

"All-time greats" is split by RANK within each query's own shortlist, not
by a fixed threshold against the whole corpus — the indexed citation
subgraph is a training/eval subset, not the full literature, so its degree
distribution is extremely sparse (median degree across the whole ~256K
indexed papers is 1). A fixed percentile threshold is degenerate on data
that skewed and would dump almost everything into one bucket; ranking
within the shortlist guarantees a real 3-way split regardless.

"Promising stars" vs "hidden gems," on the other hand, is judged against
the actual calendar date (`datetime.date.today()`), not rank — recency has
an objective meaning independent of what else is in the shortlist, unlike
degree. But the indexed dataset is a fixed, sparse subset, and a given
query paper's own topical neighborhood can skew old (an older subfield's
nearest candidates may have nothing published in the last 3 years even
though the corpus as a whole does) — so instead of either a hardcoded
narrow window (frequently empty) or a rank-based fallback (silently
mislabels an old paper as "recent" just because it's the least-old option),
the window adapts per query: try 3 years, then 5, 7, 10, 15, stopping at
the first one that actually finds a candidate. It's still always an
absolute date cutoff, never relative rank — if even 15 years back finds
nothing, "promising stars" legitimately comes back empty for that query.

Each bucket is sorted by the model's own predicted probability. Every
item has a **Why?** toggle showing the explanation for that specific
prediction:

- **Decision tree (SKE)** — when the CART surrogate's estimate agrees
  closely with the real model's score for this instance (and the tree's
  overall fidelity is high enough to trust), you get the actual decision
  path: which feature thresholds this prediction fell on either side of.
- **LIME (local surrogate)** — when the tree doesn't fit this specific
  prediction well, the app falls back to a live, from-scratch LIME
  explanation instead: the candidate's real embedding is perturbed with
  Gaussian noise, the real model scores every perturbed sample, and a
  local weighted linear surrogate is fit on the same four interpretable
  features — its coefficients are shown as the explanation, along with
  the local fit's own fidelity.

Already-cited papers are excluded from suggestions throughout.

## Why example papers?

The indexed citation subgraph is a citation-extraction training/eval
subset, not the full literature — its degree distribution is extremely
sparse (median degree across the whole ~256K indexed papers is 1). That
means most randomly-chosen query papers genuinely have no candidate the
model is confident about, and a naive demo would often show a near-zero
P(link) as if it were a real prediction. Two things follow from this:

- The graph view only draws a "predicted" edge for a candidate whose
  P(link) clears `MIN_CONFIDENT_PREDICTED_PROBABILITY` (0.5 by default,
  in `app.py`) — a dashed edge always means the model actually endorses
  it, not just "whatever ranked highest." When nothing clears the bar,
  no predicted edges are drawn and the status line says so; the bucket
  rankings still show real (if low) scores underneath.
- The landing page surfaces curated **example papers** (from
  `recommender_examples.json`) known to have a confident predicted link,
  so a new user's first interaction demonstrates the intended behavior
  clearly, rather than requiring them to stumble onto a good query paper
  by luck. Free search still works for any paper — the examples are a
  starting point, not the only thing you can explore.
