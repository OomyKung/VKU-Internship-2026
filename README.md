# Fair Influence Maximization Hybrid Prototype

This repository contains a modular Python research prototype for Fair Influence Maximization (FIM) on social networks.

The codebase focuses on a unified hybrid optimizer that combines:

- Swarm Intelligence:
  leader guidance, collective search, and movement toward promising regions
- Evolutionary Algorithms:
  crossover, mutation, population diversity, and survival selection

The project supports:

- graph loading from built-in datasets, pickled NetworkX graphs, or edge lists with optional attribute CSV files
- community detection with Louvain, Leiden, Label Propagation, and Infomap
- structural and community-aware feature extraction
- Independent Cascade Monte Carlo diffusion
- fairness evaluation with MF, DCV, and the combined score `F(S)`
- ML-guided candidate pool reduction before hybrid optimization
- experiment scripts for baseline and hybrid comparisons

## Project Structure

```text
VKU-Internship-2026/
|-- fim_hybrid/
|   |-- __init__.py
|   |-- baselines.py
|   |-- community_detection.py
|   |-- config.py
|   |-- data_loader.py
|   |-- diffusion.py
|   |-- experiment_runner.py
|   |-- fairness.py
|   |-- feature_extraction.py
|   |-- hybrid_optimizer.py
|   |-- label_generation.py
|   `-- ml_training.py
|-- networks/
|-- results/
|-- scripts/
|   `-- run_experiment.py
|-- README.md
`-- requirements.txt
```

## Setup

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Optional:

```powershell
python -m pip install xgboost
python -m pip install torch torch_geometric
```

## Graph Embedding Benchmark

The repository now includes a separate modular graph embedding benchmark under `fim_hybrid/embeddings/`.

Implemented method families:

- shallow / random-walk / proximity: `deepwalk`, `node2vec`, `line`
- autoencoder / generative: `vgae`
- message-passing GNN: `gcn`, `graphsage`
- self-supervised / contrastive GNN: `dgi`, `graphcl`
- heterogeneous placeholder: `metapath2vec` is stubbed and will skip cleanly on homogeneous graphs

Design choices:

- all methods share one interface and return a normalized `pandas.DataFrame` with `node_id` and `embedding_*` columns
- directed graphs are symmetrized by default for cross-method comparability
- GNN-based methods use structural features plus encoded node attributes when available
- missing optional dependencies fail per method without breaking unrelated benchmark runs
- downstream evaluation reuses the same normalized embedding outputs and shared seeded splits across methods

Downstream evaluation tasks:

- `node_classification`: `StandardScaler + LogisticRegression`, reporting `accuracy`, `macro_f1`, and `micro_f1`
- `link_prediction`: seeded positive/negative edge splits with Logistic Regression and default Hadamard edge features, reporting `roc_auc` and `average_precision`
- `node_clustering`: `KMeans`, reporting `nmi` and `ari`

Built-in label availability:

- `twitter`, `synth2`, `synth3`, `rice_subset`: `color`
- `graph_spa_500_0`: `region`, `ethnicity`, `age`, `gender`, `status`

Run the benchmark CLI:

```powershell
python scripts\run_embedding_benchmark.py --dataset synth3 --methods deepwalk
python scripts\run_embedding_benchmark.py --dataset synth3 --methods node2vec
python scripts\run_embedding_benchmark.py --dataset synth3 --methods line
python scripts\run_embedding_benchmark.py --dataset synth3 --methods vgae
python scripts\run_embedding_benchmark.py --dataset synth3 --methods gcn
python scripts\run_embedding_benchmark.py --dataset synth3 --methods graphsage
python scripts\run_embedding_benchmark.py --dataset synth3 --methods dgi
python scripts\run_embedding_benchmark.py --dataset synth3 --methods graphcl
python scripts\run_embedding_benchmark.py --dataset graph_spa_500_0 --methods gcn graphsage dgi --label-column gender
python scripts\run_embedding_benchmark.py --dataset synth2 --methods all --max-workers 0 --label-column color --evaluation-tasks node_classification
```

Parallel execution:

- `--max-workers 0` runs all selected embedding methods concurrently
- `--max-workers 1` forces serial execution
- use a small positive value like `--max-workers 2` or `--max-workers 4` to cap concurrency if GNN methods are heavy on your machine

Run embedding evaluation on top of the generated embeddings:

```powershell
python scripts\run_embedding_benchmark.py --dataset synth2 --methods deepwalk node2vec line --label-column color --evaluation-tasks node_classification link_prediction
python scripts\run_embedding_benchmark.py --dataset synth3 --methods deepwalk dgi graphcl --label-column color --evaluation-tasks node_classification link_prediction node_clustering
python scripts\run_embedding_benchmark.py --dataset graph_spa_500_0 --methods gcn graphsage dgi --classification-label-column gender --clustering-label-column ethnicity --evaluation-tasks node_classification link_prediction node_clustering
python scripts\run_embedding_benchmark.py --dataset graph_spa_500_0 --methods line --classification-label-column status --evaluation-tasks node_classification
```

Benchmark outputs are written under:

```text
results/<dataset>/embeddings/
```

Each run can export embeddings as CSV, pickle, and NumPy plus node-order metadata, and writes:

- an embedding summary CSV
- an evaluation summary CSV when `--evaluation-tasks` is requested
- per-method embedding exports in the same directory

## Example Commands

Run a baseline + hybrid experiment on Twitter:

```powershell
python scripts\run_experiment.py --dataset twitter --budget 40 --protected-attribute color --community-methods leiden louvain --mc-runs 50 --population-size 12 --generations 20
```

Run with ML-guided candidate restriction:

```powershell
python scripts\run_experiment.py --dataset synth3 --budget 20 --protected-attribute color --community-methods leiden --mc-runs 30 --population-size 10 --generations 15 --ml --ml-top-fraction 0.25 --ml-singleton-runs 15 --ml-max-nodes 150
```

Run side-by-side tabular and GNN-guided node scoring:

```powershell
python scripts\run_experiment.py --dataset synth3 --budget 20 --protected-attribute color --community-methods leiden --mc-runs 30 --population-size 10 --generations 15 --ml --ml-backend both --gnn-model-type graphsage --gnn-epochs 100
```

Run the GCN + RIS + Node2Vec-guided variant with final recheck on `synth2`:

```powershell
python scripts\run_experiment.py --dataset synth2 --protected-attribute color --community-methods leiden --baseline-methods degree --budget 40 --mc-runs-search 30 --mc-runs-eval 1000 --enable-final-recheck --final-recheck-mc-runs 1000 --final-recheck-top-k 2 --population-size 12 --generations 10 --random-seed 42 --ml --ml-backend gnn_ris --gnn-model-type gcn --gnn-node2vec-mode compare --ris-num-rr-sets 256 --ris-mode weak_group_weighted --fairness-urgency-weight 0.25 --diversity-weight 0.15 --ml-guidance-mode two_tier --only-methods cea_fim hybrid_siea_ml_gnn_ris_node2vec_two_tier_tuned_swap_local_search --no-ablations
```

Run the actual GCN + RIS compare mode side-by-side for plain GNN+RIS vs GNN+RIS+Node2Vec:

```powershell
python scripts\run_experiment.py --dataset synth2 --protected-attribute color --community-methods leiden --baseline-methods degree --budget 40 --mc-runs-search 30 --mc-runs-eval 1000 --enable-final-recheck --final-recheck-mc-runs 1000 --final-recheck-top-k 2 --population-size 12 --generations 10 --random-seed 42 --ml --ml-backend gnn_ris --gnn-model-type gcn --gnn-node2vec-mode compare --ris-num-rr-sets 256 --ris-mode weak_group_weighted --fairness-urgency-weight 0.25 --diversity-weight 0.15 --ml-guidance-mode two_tier --only-methods cea_fim hybrid_siea_ml_gnn_ris_two_tier_tuned_swap_local_search hybrid_siea_ml_gnn_ris_node2vec_two_tier_tuned_swap_local_search --no-ablations
```

Run a custom edge list and attribute CSV:

```powershell
python scripts\run_experiment.py --dataset custom_graph --edge-path data\custom_edges.txt --attribute-path data\custom_attributes.csv --protected-attribute group --budget 10 --community-methods label_propagation
```

## Output

Each run saves:

- a CSV result table under `results/<dataset>/<protected_attribute>/`
- optional PNG plots for `f_score`, `total_spread`, and `runtime_seconds`

## Notes

- The modular research package lives in `fim_hybrid/`.
- Run commands from the `VKU-Internship-2026` repository root so built-in datasets and default outputs stay local to this project.
- The CLI resolves relative dataset paths and `--output-dir` against the repository root.
- Leiden and Infomap use `igraph`.
- The default ML backend remains the existing tabular model; GNN scoring is optional via `--ml-backend gnn` or `--ml-backend both`.
- The GNN backend reuses the handcrafted node features and trains a node-regression model for ranking only; it does not replace the downstream hybrid optimizer.
- If `torch` or `torch_geometric` is not installed, GNN-enabled runs fail fast with a clear error while non-GNN runs continue to work unchanged.
