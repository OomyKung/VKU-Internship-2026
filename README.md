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
```

## Example Commands

Run a baseline + hybrid experiment on Twitter:

```powershell
python scripts\run_experiment.py --dataset twitter --budget 40 --protected-attribute color --community-methods leiden louvain --mc-runs 50 --population-size 12 --generations 20
```

Run with ML-guided candidate restriction:

```powershell
python scripts\run_experiment.py --dataset synth3 --budget 20 --protected-attribute color --community-methods leiden --mc-runs 30 --population-size 10 --generations 15 --ml --ml-top-fraction 0.25 --ml-singleton-runs 15 --ml-max-nodes 150
```

Run a custom edge list and attribute CSV:

```powershell
python scripts\run_experiment.py --dataset custom_graph --edge-path data\custom_edges.txt --attribute-path data\custom_attributes.csv --protected-attribute group --budget 10 --community-methods label_propagation
```

## Output

Each run saves:

- a CSV result table under `results/`
- optional PNG plots for `f_score`, `total_spread`, and `runtime_seconds`

## Notes

- The modular research package lives in `fim_hybrid/`.
- Run commands from the `VKU-Internship-2026` repository root so built-in datasets and default outputs stay local to this project.
- The CLI resolves relative dataset paths and `--output-dir` against the repository root.
- Leiden and Infomap use `igraph`.
