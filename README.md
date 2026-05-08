# Fair Influence Maximization

This repository contains a Python research prototype for Fair Influence Maximization (FIM) on social networks.

The goal is to choose a fixed-size seed set that spreads influence through a graph while avoiding outcomes where protected groups are left below their influence targets.

## Active Algorithm Stack

The active framework stack is:

- Community detection: Leiden
- Graph embedding and node ranking: GraphSAGE
- Search-time spread estimator: RIS / Fair RIS
- Diffusion model: Independent Cascade
- Optimizer: Evolutionary Algorithm + Memetic Algorithm
- Repair strategy: Target-shortfall-first fairness repair
- Final evaluator: Monte Carlo

The retired optimizer family is no longer part of the active framework. Removed optimizer and stack names fail with a clear deprecation error.

## What The Algorithm Returns

A run returns a seed set of `budget` nodes plus diagnostics explaining the quality of that seed set:

- `selected_seed_set.csv`: ranked selected seed nodes.
- `result.csv`: one-row summary of spread, fairness, runtime, and diagnostics.
- `summary.json` / `fim_stack_summary.json`: machine-readable full summary.
- `report.txt`: human-readable report with the algorithm result, fairness interpretation, and recommendations.
- `candidate_score_table.csv`: candidate-level scoring components when the score table is available.

The important result fields are:

- `Spread`: expected number of activated nodes from Monte Carlo simulation.
- `MF`: the minimum normalized influence across protected groups. Higher is better because the weakest protected group is doing better.
- `DCV_shortfall`: target-shortfall violation. `0.0` means every protected group met or exceeded its target.
- `DCV_disparity`: remaining normalized influence imbalance between groups.
- `Target Coverage Ratio`: fraction of protected groups that met their target.
- `F-score`: shortfall-first fairness score, computed as `MF - DCV_shortfall - 0.25 * DCV_disparity`.
- `fairness_gate_status`: whether the professor-priority gates passed.

## Representative Result

The saved run in `results/target_alpha_08_full` uses the active stack on the `twitter` dataset with protected attribute `color`, budget `120`, `target_alpha=0.8`, `2048` RR sets, and `1000` final Monte Carlo runs.

| Metric | Value |
| --- | ---: |
| Stack | `graphsage_leiden_ris_ic_ea_memetic` |
| Selected seeds | `120 / 120` |
| Duplicate seeds | `0` |
| Spread | `125.189` |
| MF | `0.031717` |
| DCV_shortfall | `0.000000` |
| DCV_disparity | `0.016394` |
| Target Coverage Ratio | `1.000000` |
| F-score | `0.027618` |
| Runtime | `1142.817s` |
| Fairness gate | `fail: scalability_required` |

Group-level result:

| Group | Group Size | Seeds | Influence | Target | Normalized Influence | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `0` | `812` | `28` | `30.609` | `24.560` | `0.037696` | met |
| `1` | `2755` | `85` | `87.379` | `83.320` | `0.031717` | met |
| `2` | `186` | `7` | `7.201` | `5.960` | `0.038715` | met |

What happened:

The algorithm selected all `120` required seeds with no duplicates. Every protected group exceeded its target influence, so `DCV_shortfall` is `0.0` and `Target Coverage Ratio` is `1.0`. Group `1` still has the lowest normalized influence because it is much larger than the other groups: it receives the largest absolute influence (`87.379`) but that influence is divided by `2755` group members.

Why the fairness gate still fails:

The shortfall fairness target is satisfied, but the professor-priority gate also requires the run to pass the scalability check. This saved result reports `scalability_pass=False` because the protected groups are highly imbalanced (`2755 / 186 = 14.81`) and the report warns that the budget may be too small for stable fairness on this graph. In other words, the target-shortfall result is good, but the run is still flagged as not scalable under the strict gate.

GraphSAGE quality in this run:

- Training target: `combined_fairness_gain`
- Epochs run: `100`
- Best epoch: `91`
- Validation Spearman: `0.979353`
- Precision@Budget: `0.741667`
- GraphSAGE effective score weight: `0.8`

The high validation correlation means GraphSAGE learned the fairness-guided node ranking signal well for this saved run. The optimizer still uses Fair RIS and repair rules because GraphSAGE is guidance, not final truth.

## Why The Algorithm Behaves This Way

The pipeline is target-shortfall-first:

1. Load the graph and protected groups.
   The loader verifies that each graph node has a protected-group assignment. This matters because every spread and fairness calculation is group-aware.

2. Detect Leiden communities.
   Communities give the ranker and optimizer information about graph structure. The optimizer can avoid over-concentrating seeds in one community when other communities still have useful candidates.

3. Compute protected-group targets.
   Each protected group receives an ideal influence target based on the budget and the group subgraph. `target_alpha` scales those targets. A value of `0.8` asks the algorithm to reach 80% of the original ideal target, which helps avoid impossible or unstable targets on sparse or imbalanced graphs.

4. Build RIS and Fair RIS guidance.
   RIS creates reverse reachable sets to estimate which nodes can cover many possible diffusion outcomes quickly. Fair RIS weights RR sets by protected-group need, so candidates that can help weaker or under-covered groups receive more score.

5. Train GraphSAGE as a candidate scorer.
   GraphSAGE learns a node score from structural features, communities, RIS scores, Fair RIS scores, shortfall gain, weakest-group gain, community diversity, and spread proxy signals. Protected attributes are excluded from ML features by default; group information is used in fairness targets and diagnostics.

6. Score candidates.
   The combined score blends GraphSAGE, RIS, Fair RIS, shortfall gain, community diversity, protected-group coverage, weak-group bonus, and spread proxy components. This narrows the search to promising nodes without running expensive Monte Carlo simulation for every possible seed set.

7. Run EA + Memetic search.
   The optimizer creates quota-aware initial seed sets, keeps elites, uses crossover and mutation, repairs children that lose target coverage, and applies local swap search. A move is accepted only when it preserves or improves the shortfall-first priority: target coverage first, then lower shortfall, then better MF/F-score/spread.

8. Run MF-lift.
   After shortfall targets are satisfied, the optional MF-lift phase tries swaps that improve the weakest normalized group influence while preserving `DCV_shortfall=0` and target coverage. This is why reports can say "Shortfall fairness target satisfied. Optimizing MF is now the next objective."

9. Evaluate with Monte Carlo.
   Search uses RIS/Fair RIS because it is fast. The final metrics are reported by Monte Carlo Independent Cascade simulation because it gives a more direct estimate of spread and group influence for the final seed set.

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
python scripts\run_fim_stack.py --dataset twitter --protected-attribute color --budget 120 --embedding-method graphsage --community-method leiden --optimizer ea_memetic --ris-num-rr-sets 2048 --mc-runs-eval 1000 --population-size 60 --generations 50 --target-alpha 0.8 --random-seed 42 --output-dir results\target_alpha_08_full --save-json
```

For a faster smoke run, lower `--ris-num-rr-sets`, `--mc-runs-eval`, `--population-size`, and `--generations`.

## Benchmark Run

```powershell
python scripts\run_ml_fim_benchmark.py --dataset graph_spa_500_0 --protected-attribute region --budget 50 --ml-stacks graphsage_leiden_ris_ic_ea_memetic graphsage_community_ea_memetic node2vec_xgboost_community_ea_memetic --include-baseline --optimizer ea_memetic --ranking-policy professor_priority --spread-estimator-search fairness_aware_ris --spread-estimator-final monte_carlo --use-fair-ris --force-ris-for-all-stacks --require-ris --ris-num-rr-sets 512 --mc-runs-search 20 --mc-runs-eval 1000 --population-size 20 --generations 12 --random-seed 42 --output-dir results\shortfall_dcv_strong_confirm --save-json
```

## Built-In Protected Attributes

- `twitter`, `synth2`, `synth3`, `rice_subset`: `color`
- `graph_spa_500_0`: `region`, `ethnicity`, `age`, `gender`, `status`

## Output Interpretation

Use `DCV_shortfall` and `Target Coverage Ratio` to answer whether protected groups met their targets. Use `MF` and `DCV_disparity` to understand remaining inequality after the shortfall target is met. Use `fairness_gate_status` to know whether the stricter professor-priority policy accepted the run.

For the representative Twitter result, the algorithm met all group influence targets but did not pass the strict scalability gate. The practical next experiments are a budget sweep, higher `fair_ris_score_weight`, higher `weak_group_bonus_weight`, or support-aware fairness settings for strongly imbalanced protected groups.
