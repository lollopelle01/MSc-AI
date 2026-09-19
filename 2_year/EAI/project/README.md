# PellegrinoScavello2425

A project on detecting second hand citations in scientific papers and predicting new citation links (link prediction), with attention to model transparency (local explanations, not just a score).

For an in depth description see `report.pdf` in the project root.

## Main phases

**1. Data preparation and graph building**
Papers and their citations are collected and turned into a citation graph, with text embeddings (SciBERT) on the citing sentences. This phase is handled by the `graphs/graphs_generation.ipynb` notebook and produces the cache in `graphs/cache/` (including `final_graphs.pkl`, the full graph with embeddings).

**2. Second hand citation detection**
The script `scripts/second_hand_citation.py` looks for triads u, w, x where u cites w, w cites x, and u also cites x directly, then measures how similar the sentence u uses to cite x is to the one w uses to cite x. A high percentile score suggests a possible case of a citation copied from an intermediate source rather than read directly.

**3. Training the link prediction models**
The `scripts/` folder contains the scripts to train several models that predict new links in the citation graph: GraphSAGE, SEAL, and a few MLP baselines (on raw or encoded embeddings), both in their standard version and with hard negatives. The `GNNs/train_and_test.ipynb` notebook collects training and evaluation, with weights saved in `GNNs/weights/` and results in `GNNs/results/`.

**4. Web apps for exploring the results**
Two local Flask apps let you explore the results without reading raw CSVs.
   `second_hand_citation_webapp/` shows the detected second hand citations, with the citing sentences shown side by side as evidence.
   `link_predictor_webapp/` shows the link prediction model's suggestions on an interactive graph, with an explanation for every suggestion (decision tree or LIME).

## How to run the main things

**Environment**
The project uses a conda environment called `ethics`.
```bash
conda env create -f environment.yml
conda activate ethics
pip install -r requirements.txt
```

**Detect second hand citations**
```bash
python scripts/second_hand_citation.py --split test --top-k 20
```
Requires `transformers` to be installed (for SciBERT). The output is a CSV in `GNNs/results/`.

**Train a link prediction model**
Example with GraphSAGE:
```bash
python scripts/train_graphsage.py
```
The other models are run the same way with the corresponding script, e.g. `train_seal.py`, `train_mlp_baseline_raw.py`, or the `_hard` variants for hard negatives.

**Run the second hand citation web app**
```bash
cd second_hand_citation_webapp
pip install -r requirements.txt
python app.py
```
Open `http://localhost:5050`. Requires the CSV generated in the previous step. For the graph view, first run `python build_graph_index.py` (one time only).

**Run the link predictor web app**
```bash
cd link_predictor_webapp
python build_recommender_index.py
```
This must be run once, in the same conda environment used for training (it needs `torch`), since it loads the full graph and the trained model to precompute the embeddings.
```bash
pip install -r requirements.txt
python app.py
```
Open `http://localhost:5051`.