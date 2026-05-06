# Fair Influence Maximization

This repository contains a Python research prototype for Fair Influence Maximization on social networks.

The active framework stack is:

- Community Detection: Leiden
- Graph Embedding: GraphSAGE
- Search-time Simulation: RIS / Fair RIS
- Diffusion Model: Independent Cascade
- Optimizer: Evolutionary Algorithm + Memetic Algorithm
- Final Evaluator: Monte Carlo

The retired optimizer family is no longer part of the active framework. Removed optimizer and stack names fail with a clear deprecation error.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Optional GraphSAGE dependencies:

```powershell
python -m pip install torch torch_geometric
```

## Primary FIM Run

```powershell
python scripts\run_fim_stack.py --dataset graph_spa_500_0 --protected-attribute ethnicity --budget 40 --embedding-method graphsage --community-method leiden --optimizer ea_memetic --ris-num-rr-sets 512 --mc-runs-eval 300 --population-size 12 --generations 5 --random-seed 42 --output-dir results\fim_smoke_test --save-json
```

## Benchmark Run

```powershell
python scripts\run_ml_fim_benchmark.py --dataset graph_spa_500_0 --protected-attribute region --budget 50 --ml-stacks graphsage_leiden_ris_ic_ea_memetic graphsage_community_ea_memetic node2vec_xgboost_community_ea_memetic --include-baseline --optimizer ea_memetic --ranking-policy professor_priority --spread-estimator-search fairness_aware_ris --spread-estimator-final monte_carlo --use-fair-ris --force-ris-for-all-stacks --require-ris --ris-num-rr-sets 512 --mc-runs-search 20 --mc-runs-eval 1000 --population-size 20 --generations 12 --random-seed 42 --output-dir results\shortfall_dcv_strong_confirm --save-json
```

## Built-In Protected Attributes

- `twitter`, `synth2`, `synth3`, `rice_subset`: `color`
- `graph_spa_500_0`: `region`, `ethnicity`, `age`, `gender`, `status`

## Outputs

Runs save CSV and optional JSON output under the selected `--output-dir`. Result rows report `MF`, `DCV`, `F-score`, final Monte Carlo spread, `optimizer_mode=ea_memetic`, and `method_type=graphsage_leiden_ris_ic_ea_memetic` for the primary stack.
