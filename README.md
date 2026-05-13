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
- `DCV`: target-shortfall violation. `0.0` means every protected group met or exceeded its effective ideal target.
- `DCV_shortfall`: same value as `DCV`, retained in detailed outputs for clarity.
- `Target Coverage Ratio`: fraction of protected groups that met their target.
- `F-score`: shortfall-first fairness score, computed as `MF - DCV`.
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
| DCV | `0.000000` |
| Target Coverage Ratio | `1.000000` |
| F-score | `0.031717` |
| Runtime | `1142.817s` |
| Fairness gate | `fail: scalability_required` |

Group-level result:

| Group | Group Size | Seeds | Influence | Target | Normalized Influence | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `0` | `812` | `28` | `30.609` | `24.560` | `0.037696` | met |
| `1` | `2755` | `85` | `87.379` | `83.320` | `0.031717` | met |
| `2` | `186` | `7` | `7.201` | `5.960` | `0.038715` | met |

What happened:

The algorithm selected all `120` required seeds with no duplicates. Every protected group exceeded its target influence, so `DCV` is `0.0` and `Target Coverage Ratio` is `1.0`. Group `1` still has the lowest normalized influence because it is much larger than the other groups: it receives the largest absolute influence (`87.379`) but that influence is divided by `2755` group members.

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

## Saved Dataset Results

The table below summarizes the current top-level saved dataset runs under `results/*/summary.json`.

| Run | Dataset | Attr | Budget | RR | MC | F-score | MF | DCV | Coverage | Spread | Runtime(s) | Gate |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| graph_spa_age_fast_test | graph_spa_500_0 | age | 50 | 512 | 300 | 0.0952 | 0.0952 | 0.0000 | 1.0000 | 51.513 | 1.96 | fail: scalability_required |
| graph_spa_ethnicity_fast_test | graph_spa_500_0 | ethnicity | 50 | 512 | 300 | 0.0978 | 0.0978 | 0.0000 | 1.0000 | 51.683 | 1.70 | fail: scalability_required |
| graph_spa_gender_fast_test | graph_spa_500_0 | gender | 50 | 512 | 300 | 0.1018 | 0.1018 | 0.0000 | 1.0000 | 51.703 | 1.68 | fail: scalability_required |
| graph_spa_region_fast_confirm | graph_spa_500_0 | region | 50 | 768 | 1000 | 0.0890 | 0.0890 | 0.0000 | 1.0000 | 51.936 | 52.14 | fail: scalability_required |
| graph_spa_region_fast_test | graph_spa_500_0 | region | 50 | 512 | 300 | 0.0887 | 0.0887 | 0.0000 | 1.0000 | 51.650 | 40.51 | fail: scalability_required |
| scalability_rice_balanced | rice_subset | color | 70 | 1024 | 1000 | 0.2304 | 0.2304 | 0.0000 | 1.0000 | 117.408 | 44.46 | pass |
| scalability_synth2_balanced | synth2 | color | 40 | 1024 | 1000 | 0.0830 | 0.0830 | 0.0000 | 1.0000 | 42.439 | 8.77 | pass |
| scalability_synth3_balanced | synth3 | color | 40 | 1024 | 1000 | 0.0815 | 0.0815 | 0.0000 | 1.0000 | 42.378 | 90.42 | pass |
| target_alpha_08_full | twitter | color | 120 | 2048 | 1000 | 0.0316 | 0.0316 | 0.0000 | 1.0000 | 123.125 | 1117.05 | fail: scalability_required |
| twitter_balanced_seed_21 | twitter | color | 120 | 1024 | 1000 | 0.0334 | 0.0334 | 0.0000 | 1.0000 | 125.783 | 144.75 | fail: scalability_required |
| twitter_balanced_seed_42 | twitter | color | 120 | 1024 | 1000 | 0.0316 | 0.0316 | 0.0000 | 1.0000 | 122.973 | 148.26 | fail: scalability_required |
| twitter_balanced_seed_7 | twitter | color | 120 | 1024 | 1000 | 0.0313 | 0.0313 | 0.0000 | 1.0000 | 124.703 | 180.78 | fail: scalability_required |
| twitter_balanced_seed_77 | twitter | color | 120 | 1024 | 1000 | 0.0320 | 0.0320 | 0.0000 | 1.0000 | 123.297 | 161.88 | fail: scalability_required |
| twitter_balanced_seed_99 | twitter | color | 120 | 1024 | 1000 | 0.0316 | 0.0316 | 0.0000 | 1.0000 | 122.798 | 162.39 | fail: scalability_required |
| twitter_scalability_balanced_confirm | twitter | color | 120 | 1024 | 1000 | 0.0316 | 0.0316 | 0.0000 | 1.0000 | 122.973 | 157.24 | fail: scalability_required |

Runs with `DCV = 0.0000` and `Coverage = 1.0000` met the target-shortfall fairness objective. A `scalability_required` gate failure means the stricter professor-priority scalability check still rejected the run.

## CEA-FIM Comparison Evaluation Setting

The following experimental setting is used when evaluating the proposed GraphSAGE + Leiden + Fair RIS + EA/Memetic framework for manual comparison with the CEA-FIM paper.

| Evaluation Requirement | Value |
|---|---:|
| Budget | 40 |
| Monte Carlo Simulations | 1000 |
| Population Size | 10 |
| Generations | 150 |

### Reproduction Command

```bash
python scripts/run_fim_stack.py --dataset twitter --protected-attribute color --budget 40 --embedding-method graphsage --graphsage-training-target combined_fairness_gain --community-method leiden --optimizer ea_memetic --ris-num-rr-sets 768 --mc-runs-eval 1000 --population-size 10 --generations 150 --graphsage-hidden-dim 16 --graphsage-embedding-dim 32 --graphsage-epochs 30 --target-alpha 0.5 --candidate-pool-size 120 --shortfall-repair-rounds 2 --shortfall-repair-candidate-limit 60 --include-mf-gain-in-graphsage-label --random-seed 42 --output-dir results/twitter_fast_compared_to_CEA_FIM --save-json --output-mode compact
```

This configuration is used as the proposed-method evaluation setting for manual comparison against the CEA-FIM paper.

## Proposed-Method Results for CEA-FIM Comparison Setting

The following table collects all saved proposed-method experiment results that use the evaluation setting selected for manual comparison with the CEA-FIM paper.

### Fixed Comparison Setting

| Requirement | Value |
|---|---:|
| Budget | 40 |
| Monte Carlo Simulations | 1000 |
| Population Size | 10 |
| Generations | 150 |

### Matching Proposed-Method Results

| Dataset | Protected Attribute | Target Alpha | RR Sets | Spread | MF | DCV | Target Coverage | F-score | Runtime (s) | Seed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| graph_spa_500_0 | age | 0.5 | 768 | 41.3440 | 0.0742 | 0.0000 | 1.0000 | 0.0742 | 24.844 | N/A |
| graph_spa_500_0 | ethnicity | 0.5 | 768 | 41.6340 | 0.0731 | 0.0000 | 1.0000 | 0.0731 | 17.393 | N/A |
| graph_spa_500_0 | gender | 0.5 | 768 | 41.7400 | 0.0820 | 0.0000 | 1.0000 | 0.0820 | 71.671 | N/A |
| graph_spa_500_0 | region | 0.5 | 768 | 41.3950 | 0.0628 | 0.0000 | 1.0000 | 0.0628 | 23.916 | N/A |
| rice_subset | color | 0.5 | 768 | 72.8330 | 0.1238 | 0.0000 | 1.0000 | 0.1238 | 63.361 | N/A |
| synth2 | color | 0.5 | 768 | 42.8850 | 0.0840 | 0.0000 | 1.0000 | 0.0840 | 19.366 | N/A |
| synth3 | color | 0.5 | 768 | 42.2110 | 0.0823 | 0.0000 | 1.0000 | 0.0823 | 18.518 | N/A |
| twitter | color | 0.5 | 768 | 40.9320 | 0.0100 | 0.0000 | 1.0000 | 0.0100 | 132.352 | N/A |

All rows have `DCV = 0.0000` and `Target Coverage = 1.0000`, meaning every protected group met its shortfall-based fairness target in each run. The random seed is not stored in `summary.json` and is therefore listed as N/A.

### Saved Result Folders

- `results/graph_spa_age_fast_compared_to_CEA_FIM`
- `results/graph_spa_ethnicity_fast_compared_to_CEA_FIM`
- `results/graph_spa_gender_fast_compared_to_CEA_FIM`
- `results/graph_spa_region_fast_compared_to_CEA_FIM`
- `results/rice_subset_fast_compared_to_CEA_FIM`
- `results/synth2_fast_compared_to_CEA_FIM`
- `results/synth3_fast_compared_to_CEA_FIM`
- `results/twitter_fast_compared_to_CEA_FIM`

These runs are intended to be compared manually against the CEA-FIM paper using the same chosen high-level evaluation requirements.

Do not add any CEA-FIM scores or claims here.

## Detailed Algorithm Explanation

### Problem Being Solved

The input is a graph `G = (V, E)`, a seed budget `k`, and a protected attribute such as `color`, `region`, or `ethnicity`. The algorithm must choose a seed set `S` where `|S| = k`.

The selected seed set should satisfy two goals at the same time:

1. Spread influence to many nodes.
2. Make sure protected groups receive enough influence, especially groups that would otherwise be under-served.

This is harder than ordinary influence maximization because a node with high total spread may mainly activate the already well-served group. The active stack therefore ranks nodes by both influence potential and fairness contribution.

### Diffusion Model

Final influence is measured with the Independent Cascade model.

The process is:

1. The selected seeds start active.
2. Each newly active node gets one chance to activate each inactive neighbor.
3. Activation succeeds with the edge probability `p`, or the fallback `--propagation-prob`.
4. The cascade stops when no new nodes activate.
5. The process is repeated many times, and the result is the mean spread.

The representative run uses `--mc-runs-eval 1000`, so the reported `Spread`, `MF`, `DCV_shortfall`, and group influence values are Monte Carlo averages over 1000 final simulations.

### Fairness Metrics

For each protected group `g`, the final evaluator computes:

```text
group_influence[g] = expected activated nodes in group g
normalized_influence[g] = group_influence[g] / group_size[g]
MF = min(normalized_influence[g])
```

`MF` is strict: it only looks at the weakest protected group. If one group receives low normalized influence, `MF` is low even when total spread is high.

The target-shortfall metric compares each group against an effective influence target:

```text
effective_target[g] = target_alpha * original_ideal_target[g]
shortfall[g] = max(0, effective_target[g] - group_influence[g])
shortfall_ratio[g] = shortfall[g] / effective_target[g]
DCV = DCV_shortfall = average(shortfall_ratio[g])
Target Coverage Ratio = groups_with_zero_shortfall / number_of_groups
```

`DCV = 0.0` means no protected group is below its effective ideal target. `Target Coverage Ratio = 1.0` means every protected group satisfies the fairness target.

The framework does not require exact equality of normalized influence across protected groups. `MF` remains useful after shortfall fairness is satisfied because it identifies the weakest normalized group influence.

The shortfall-first F-score used by the active stack is:

```text
F-score = MF - DCV
```

For the representative result:

```text
F-score = 0.031717 - 0.000000
        = 0.031717
```

### Step 1: Loading And Protected-Group Validation

The loader reads the graph and protected-group attribute file. It verifies that every graph node belongs to exactly one protected group for the selected attribute.

This happens because all fairness metrics require a complete node-to-group mapping. If a node could activate without a known group, the final group spread totals would be incomplete and fairness values would be misleading.

### Step 2: Leiden Community Detection

The stack detects Leiden communities before scoring candidates.

Leiden is used because influence usually travels through graph structure. If all seeds are placed inside one dense region, total spread may look good locally but the seed set can miss other useful regions. Community information gives the model and optimizer a structural signal for seed diversity.

In the representative Twitter result:

- Number of communities: `118`
- Communities covered by selected seeds: `46`
- Community modularity: `0.655835`

This means the graph has strong community structure, so community-aware scoring is useful.

### Step 3: Computing Group Influence Targets

Each group receives an ideal influence target. The code first estimates an original target for each group using the budget and the group's induced subgraph. Then `target_alpha` scales it:

```text
effective_target[g] = target_alpha * original_ideal_target[g]
```

The representative run uses `target_alpha=0.8`.

| Group | Original Ideal | Effective Target |
| --- | ---: | ---: |
| `0` | `30.700` | `24.560` |
| `1` | `104.150` | `83.320` |
| `2` | `7.450` | `5.960` |

This happens because full ideal targets can be too strict when groups are sparse, disconnected, or highly imbalanced. Scaling by `target_alpha` keeps the fairness target meaningful while making it feasible for the chosen budget.

### Step 4: RIS And Fair RIS Guidance

RIS means Reverse Influence Sampling. Instead of simulating diffusion from every possible seed set, RIS samples reverse reachable sets:

1. Pick a random root node.
2. Traverse edges in reverse using the same activation probability.
3. Store the set of nodes that could have activated that root.
4. A candidate node gets score when it appears in many reverse reachable sets.

A high RIS score means the node can likely influence many possible cascade endpoints.

Fair RIS adds protected-group weighting. Reverse reachable sets rooted in under-covered or weak groups receive more weight. This changes the node score from "covers many endpoints" to "covers many useful endpoints, especially for fairness."

In the representative run:

- RR sets used: `2048`
- RIS active: `true`
- Fair RIS active: `true`
- Fair RIS mode: `weak_group_weighted`
- Nonzero RIS node scores: `1647`
- Nonzero Fair RIS node scores: `1647`

This happens so the optimizer can evaluate and repair seed sets quickly without running expensive Monte Carlo simulations during every candidate swap.

### Step 5: GraphSAGE Candidate Scoring

GraphSAGE learns a node ranking model. It does not directly choose the final seed set. It produces a guidance score that is combined with RIS and fairness components.

The GraphSAGE target is `combined_fairness_gain`, built from signals such as:

- Fair RIS score
- Standard RIS score
- Shortfall gain
- Weakest-group gain / MF gain
- Community diversity
- Spread proxy based on degree, PageRank, and RIS coverage

Protected attributes are not used as raw ML features by default. They are used to build fairness labels and diagnostics, while the model learns from structural and diffusion-guidance features.

For the representative run:

- Validation Spearman: `0.979353`
- Precision@Budget: `0.741667`
- NDCG@Budget: `0.960400`

This means the learned ranking closely matches the generated fairness-aware labels. The optimizer still combines it with Fair RIS because a learned score is only an approximation.

### Step 6: Building The Candidate Pool

The optimizer cannot test every possible seed set. It first creates a candidate pool.

Candidate scoring combines weighted components:

```text
combined_score ~= GraphSAGE
                + RIS
                + Fair RIS
                + shortfall gain
                + weak-group / MF gain
                + community diversity
                + protected-group coverage
                + spread proxy
```

The default active run weights include:

| Component | Weight |
| --- | ---: |
| GraphSAGE score | `0.8` |
| RIS score | `1.0` |
| Fair RIS score | `2.0` |
| Shortfall gain | `3.0` |
| Community diversity | `0.5` |
| Spread proxy | `0.3` |

For highly imbalanced data, the stack can use group-stratified candidate selection. In the representative Twitter result, the group sizes are very imbalanced, so the candidate pool is stratified:

| Group | Candidate Pool Count |
| --- | ---: |
| `0` | `481` |
| `1` | `1585` |
| `2` | `186` |

This happens because a purely global top-score pool could exclude smaller or harder-to-reach groups before the optimizer gets a chance to repair fairness.

### Step 7: EA + Memetic Optimization

The optimizer works over complete seed sets, not individual nodes.

It starts with an initial population using several strategies:

- `quota_balanced`: starts from approximate protected-group seed quotas.
- `shortfall_target_heavy`: emphasizes groups with target gaps.
- `combined_score_topk`: uses the strongest candidate scores.
- `random_diverse`: adds diversity so search does not collapse into one local pattern.

Then each generation applies:

1. Elitism: keep the best seed sets.
2. Parent selection: choose good seed sets for reproduction.
3. Crossover: combine parts of two seed sets.
4. Mutation: replace some seeds with candidate nodes.
5. Repair: fix children that lose target coverage or worsen shortfall.
6. Local search: try swaps around strong seed sets.

The target-shortfall priority is:

1. Prefer higher `Target Coverage Ratio`.
2. Then prefer lower `DCV`.
3. Then prefer better `MF`.
4. Then prefer higher F-score.
5. Then prefer higher spread, with lower runtime only as a final tie-breaker.

This ordering explains why the algorithm may reject a high-spread seed if it causes a group to fall below target.

### Step 8: Repair

Repair is the fairness-first correction step.

If a seed set leaves a group below target, repair tries to:

1. Identify groups with shortfall.
2. Rank candidate additions by Fair RIS marginal gain for those groups.
3. Rank removable seeds from over-covered or less useful groups.
4. Swap one or more seeds.
5. Reject the swap if it loses target coverage or worsens shortfall beyond tolerance.

In the representative Twitter result, the initial quota-aware seed sets already met the target-shortfall objective, so there were no successful shortfall repairs needed. The optimizer instead spent most effort on F-score and MF-safe swaps.

### Step 9: MF-Lift

Once `DCV_shortfall = 0.0`, the algorithm changes focus. At that point, every group met its target, so the next useful improvement is raising `MF`, the weakest normalized influence.

MF-lift tries swaps that:

- Improve the weakest group's normalized influence.
- Preserve `DCV = 0.0`.
- Preserve `Target Coverage Ratio = 1.0`.
- Avoid unacceptable spread degradation.

The final implementation evaluates pre-lift, post-lift, and accepted lift-swap candidates with Monte Carlo, then chooses the best one under the same shortfall-first priority.

When MF-lift is enabled, the report includes fields such as:

- `mf_lift_started`
- `mf_lift_attempts`
- `mf_lift_successful_swaps`
- `mf_lift_rejected_target_loss`
- `mf_lift_rejected_dcv_shortfall_worsen`
- `mf_lift_final_mc_mf`
- `mf_lift_target_coverage_preserved`

The representative `results/target_alpha_08_full` row reports final Monte Carlo `MF = 0.031717`. More recent saved runs, such as `results/twitter_mf_lift`, include explicit MF-lift diagnostics for comparing pre-lift, post-lift, and accepted swap candidates.

This happens because approximate RIS/proxy MF and final Monte Carlo MF can differ. The algorithm therefore uses Monte Carlo at the end to avoid trusting the proxy blindly.

### Step 10: Final Monte Carlo Evaluation

The final seed set is evaluated with Monte Carlo Independent Cascade simulation. This is the result reported in `result.csv`, `summary.json`, and `report.txt`.

Search-time RIS is fast but approximate. Final Monte Carlo is slower but more direct. That is why reports can show both:

- `approx_final_search_spread`: search-time proxy value.
- `final_mc_spread`: final Monte Carlo value.

The README table uses final Monte Carlo values.

## Why The Representative Result Looks Like It Does

The final selected seed distribution is:

```text
group 0: 28 seeds
group 1: 85 seeds
group 2: 7 seeds
```

The expected quota estimate was close to:

```text
group 0: 25.96 seeds
group 1: 88.09 seeds
group 2: 5.95 seeds
```

The selected seed counts are therefore broadly proportional to protected-group support, with enough seeds in the small group to avoid zero or near-zero influence.

The final influence values are:

```text
group 0: 30.609 influence vs 24.560 target
group 1: 87.379 influence vs 83.320 target
group 2: 7.201 influence vs 5.960 target
```

Every group beats its target:

```text
shortfall group 0 = 0
shortfall group 1 = 0
shortfall group 2 = 0
DCV_shortfall = 0
Target Coverage Ratio = 3 / 3 = 1.0
```

Group `1` is still the weakest group by normalized influence:

```text
group 0 normalized = 30.609 / 812  = 0.037696
group 1 normalized = 87.379 / 2755 = 0.031717
group 2 normalized = 7.201 / 186   = 0.038715
MF = min(...) = 0.031717
```

This is why the report says the shortfall target is satisfied but MF remains low. The largest group receives the most total influence, but because the group is very large, its per-member normalized influence is still the minimum.

The strict fairness gate fails only because of `scalability_required`, not because of target shortfall:

```text
DCV_shortfall = 0.0
Target Coverage Ratio = 1.0
zero_covered_groups = 0
scalability_pass = false
fairness_gate_status = fail
fairness_gate_failures = scalability_required
```

The graph has strong protected-group imbalance:

```text
largest group / smallest group = 2755 / 186 = 14.81
```

That imbalance makes the run fail the strict scalability gate, even though all groups are covered and all effective targets are met.

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

# Evaluation Command Summary

This section summarizes the most useful commands for debugging, balanced evaluation, scalability testing, multi-seed stability, ablation studies, result summarization, and result cleanup.

---

## 1. Quick Single-Dataset Debug Runs

### Fast synth2 test

```bash
python scripts/run_fim_stack.py --dataset synth2 --protected-attribute color --budget 40 --embedding-method graphsage --graphsage-training-target combined_fairness_gain --community-method leiden --optimizer ea_memetic --ris-num-rr-sets 1024 --mc-runs-eval 500 --population-size 40 --generations 30 --graphsage-hidden-dim 16 --graphsage-embedding-dim 32 --graphsage-epochs 50 --target-alpha 0.8 --candidate-pool-size 250 --shortfall-repair-rounds 8 --shortfall-repair-candidate-limit 300 --use-mf-lift --mf-lift-rounds 4 --mf-lift-candidate-limit 200 --include-mf-gain-in-graphsage-label --random-seed 42 --output-dir results/synth2_mf_lift_fast --save-json --output-mode compact
```

### Fast graph_spa_500_0 region test

```bash
python scripts/run_fim_stack.py --dataset graph_spa_500_0 --protected-attribute region --budget 50 --embedding-method graphsage --graphsage-training-target combined_fairness_gain --community-method leiden --optimizer ea_memetic --ris-num-rr-sets 512 --mc-runs-eval 300 --population-size 12 --generations 8 --graphsage-hidden-dim 16 --graphsage-embedding-dim 32 --graphsage-epochs 30 --target-alpha 0.8 --candidate-pool-size 120 --shortfall-repair-rounds 2 --shortfall-repair-candidate-limit 60 --include-mf-gain-in-graphsage-label --random-seed 42 --output-dir results/graph_spa_region_fast_test --save-json --output-mode compact
```

### Fast Twitter scalability test

```bash
python scripts/run_fim_stack.py --dataset twitter --protected-attribute color --budget 120 --embedding-method graphsage --graphsage-training-target combined_fairness_gain --community-method leiden --optimizer ea_memetic --ris-num-rr-sets 512 --mc-runs-eval 300 --population-size 30 --generations 20 --graphsage-hidden-dim 16 --graphsage-embedding-dim 32 --graphsage-epochs 50 --target-alpha 0.8 --candidate-pool-size 500 --shortfall-repair-rounds 6 --shortfall-repair-candidate-limit 250 --use-mf-lift --mf-lift-rounds 3 --mf-lift-candidate-limit 150 --include-mf-gain-in-graphsage-label --random-seed 42 --output-dir results/twitter_scalability_smoke --save-json --output-mode compact
```

---

## 2. Balanced Report-Quality Runs

### Balanced synth2

```bash
python scripts/run_fim_stack.py --dataset synth2 --protected-attribute color --budget 40 --embedding-method graphsage --graphsage-training-target combined_fairness_gain --community-method leiden --optimizer ea_memetic --ris-num-rr-sets 1024 --mc-runs-eval 1000 --population-size 30 --generations 20 --graphsage-hidden-dim 16 --graphsage-embedding-dim 32 --graphsage-epochs 50 --target-alpha 0.8 --candidate-pool-size 250 --shortfall-repair-rounds 8 --shortfall-repair-candidate-limit 300 --use-mf-lift --mf-lift-rounds 4 --mf-lift-candidate-limit 200 --include-mf-gain-in-graphsage-label --random-seed 42 --output-dir results/scalability_synth2_balanced --save-json --output-mode compact
```

### Balanced synth3

```bash
python scripts/run_fim_stack.py --dataset synth3 --protected-attribute color --budget 40 --embedding-method graphsage --graphsage-training-target combined_fairness_gain --community-method leiden --optimizer ea_memetic --ris-num-rr-sets 1024 --mc-runs-eval 1000 --population-size 30 --generations 20 --graphsage-hidden-dim 16 --graphsage-embedding-dim 32 --graphsage-epochs 50 --target-alpha 0.8 --candidate-pool-size 300 --shortfall-repair-rounds 8 --shortfall-repair-candidate-limit 300 --use-mf-lift --mf-lift-rounds 4 --mf-lift-candidate-limit 200 --include-mf-gain-in-graphsage-label --random-seed 42 --output-dir results/scalability_synth3_balanced --save-json --output-mode compact
```

### Balanced rice_subset

```bash
python scripts/run_fim_stack.py --dataset rice_subset --protected-attribute color --budget 70 --embedding-method graphsage --graphsage-training-target combined_fairness_gain --community-method leiden --optimizer ea_memetic --ris-num-rr-sets 1024 --mc-runs-eval 1000 --population-size 30 --generations 20 --graphsage-hidden-dim 16 --graphsage-embedding-dim 32 --graphsage-epochs 50 --target-alpha 0.8 --candidate-pool-size 400 --shortfall-repair-rounds 8 --shortfall-repair-candidate-limit 350 --use-mf-lift --mf-lift-rounds 4 --mf-lift-candidate-limit 250 --include-mf-gain-in-graphsage-label --random-seed 42 --output-dir results/scalability_rice_balanced --save-json --output-mode compact
```

### Balanced Twitter

```bash
python scripts/run_fim_stack.py --dataset twitter --protected-attribute color --budget 120 --embedding-method graphsage --graphsage-training-target combined_fairness_gain --community-method leiden --optimizer ea_memetic --ris-num-rr-sets 1024 --mc-runs-eval 1000 --population-size 30 --generations 20 --graphsage-hidden-dim 16 --graphsage-embedding-dim 32 --graphsage-epochs 50 --target-alpha 0.8 --candidate-pool-size 500 --shortfall-repair-rounds 6 --shortfall-repair-candidate-limit 250 --use-mf-lift --mf-lift-rounds 3 --mf-lift-candidate-limit 150 --include-mf-gain-in-graphsage-label --random-seed 42 --output-dir results/twitter_scalability_balanced_confirm --save-json --output-mode compact
```

### Balanced graph_spa_500_0 region confirmation

```bash
python scripts/run_fim_stack.py --dataset graph_spa_500_0 --protected-attribute region --budget 50 --embedding-method graphsage --graphsage-training-target combined_fairness_gain --community-method leiden --optimizer ea_memetic --ris-num-rr-sets 768 --mc-runs-eval 1000 --population-size 12 --generations 8 --graphsage-hidden-dim 16 --graphsage-embedding-dim 32 --graphsage-epochs 30 --target-alpha 0.8 --candidate-pool-size 120 --shortfall-repair-rounds 2 --shortfall-repair-candidate-limit 60 --include-mf-gain-in-graphsage-label --random-seed 42 --output-dir results/graph_spa_region_fast_confirm --save-json --output-mode compact
```

---

## 3. All-Dataset Evaluation Runner

### Run all supported dataset/protected-attribute pairs

```bash
python scripts/run_all_fim_evaluations.py
```

### Fast mode

```bash
python scripts/run_all_fim_evaluations.py --mode fast
```

### Balanced mode

```bash
python scripts/run_all_fim_evaluations.py --mode balanced
```

### Quality mode

```bash
python scripts/run_all_fim_evaluations.py --mode quality
```

### Preview commands without executing

```bash
python scripts/run_all_fim_evaluations.py --dry-run
```

### Skip completed outputs

```bash
python scripts/run_all_fim_evaluations.py --skip-existing
```

### Run selected datasets only

```bash
python scripts/run_all_fim_evaluations.py --datasets synth2,twitter,graph_spa_500_0
```

### Run selected protected attributes only

```bash
python scripts/run_all_fim_evaluations.py --protected-attributes color,region,ethnicity
```

### Run all datasets with multiple seeds

```bash
python scripts/run_all_fim_evaluations.py --seeds 7,21,42
```

### Run all datasets once, but Twitter with multiple seeds

```bash
python scripts/run_all_fim_evaluations.py --multi-seed-twitter
```

---

## 4. Multi-Seed Stability Runs

### Twitter multi-seed stability

```bash
python scripts/run_all_fim_evaluations.py --datasets twitter --multi-seed-twitter
```

### All datasets with three seeds

```bash
python scripts/run_all_fim_evaluations.py --seeds 7,21,42
```

---

## 5. Ablation Tests

### Full stack baseline

```bash
python scripts/run_fim_stack.py --dataset twitter --protected-attribute color --budget 120 --embedding-method graphsage --graphsage-training-target combined_fairness_gain --community-method leiden --optimizer ea_memetic --ris-num-rr-sets 1024 --mc-runs-eval 1000 --population-size 30 --generations 20 --graphsage-hidden-dim 16 --graphsage-embedding-dim 32 --graphsage-epochs 50 --target-alpha 0.8 --candidate-pool-size 500 --shortfall-repair-rounds 6 --shortfall-repair-candidate-limit 250 --use-mf-lift --mf-lift-rounds 3 --mf-lift-candidate-limit 150 --include-mf-gain-in-graphsage-label --random-seed 42 --output-dir results/ablation_full_stack --save-json --output-mode compact
```

### No GraphSAGE guidance

```bash
python scripts/run_fim_stack.py --dataset twitter --protected-attribute color --budget 120 --embedding-method graphsage --graphsage-training-target combined_fairness_gain --community-method leiden --optimizer ea_memetic --ris-num-rr-sets 1024 --mc-runs-eval 1000 --population-size 30 --generations 20 --graphsage-hidden-dim 16 --graphsage-embedding-dim 32 --graphsage-epochs 50 --target-alpha 0.8 --candidate-pool-size 500 --shortfall-repair-rounds 6 --shortfall-repair-candidate-limit 250 --use-mf-lift --mf-lift-rounds 3 --mf-lift-candidate-limit 150 --include-mf-gain-in-graphsage-label --disable-graphsage-guidance --random-seed 42 --output-dir results/ablation_no_graphsage --save-json --output-mode compact
```

### No MF-Lift

```bash
python scripts/run_fim_stack.py --dataset twitter --protected-attribute color --budget 120 --embedding-method graphsage --graphsage-training-target combined_fairness_gain --community-method leiden --optimizer ea_memetic --ris-num-rr-sets 1024 --mc-runs-eval 1000 --population-size 30 --generations 20 --graphsage-hidden-dim 16 --graphsage-embedding-dim 32 --graphsage-epochs 50 --target-alpha 0.8 --candidate-pool-size 500 --shortfall-repair-rounds 6 --shortfall-repair-candidate-limit 250 --random-seed 42 --output-dir results/ablation_no_mf_lift --save-json --output-mode compact
```

### No Fair RIS guidance

```bash
python scripts/run_fim_stack.py --dataset twitter --protected-attribute color --budget 120 --embedding-method graphsage --graphsage-training-target combined_fairness_gain --community-method leiden --optimizer ea_memetic --ris-num-rr-sets 1024 --mc-runs-eval 1000 --population-size 30 --generations 20 --graphsage-hidden-dim 16 --graphsage-embedding-dim 32 --graphsage-epochs 50 --target-alpha 0.8 --candidate-pool-size 500 --shortfall-repair-rounds 6 --shortfall-repair-candidate-limit 250 --use-mf-lift --mf-lift-rounds 3 --mf-lift-candidate-limit 150 --include-mf-gain-in-graphsage-label --disable-fair-ris-guidance --random-seed 42 --output-dir results/ablation_no_fair_ris --save-json --output-mode compact
```

---

## 6. Result Summary Tables

### All-dataset evaluation summary

```bash
python scripts/summarize_scalability_results.py --results-root results --include-pattern all_eval_ --output-dir results/all_dataset_eval_summary --print-table --save-csv --save-json --save-markdown --table-mode compact
```

### Scalability-only summary

```bash
python scripts/summarize_scalability_results.py --results-root results --include-pattern scalability_ --output-dir results/scalability_summary --print-table --save-csv --save-json --save-markdown --table-mode compact
```

### All-results summary

```bash
python scripts/summarize_scalability_results.py --results-root results --output-dir results/all_results_summary --print-table --save-csv --save-json --save-markdown --table-mode compact
```

### Vertical readable summary

```bash
python scripts/summarize_scalability_results.py --results-root results --output-dir results/all_results_summary --print-table --save-csv --save-json --save-markdown --table-mode vertical
```

---

## 7. Cleanup Commands

### Preview cleanup that keeps only today's runs

```bash
python scripts/cleanup_results.py --results-root results --keep-only-today
```

### Delete older-than-today runs

```bash
python scripts/cleanup_results.py --results-root results --keep-only-today --apply --delete --confirm-delete
```

### Regenerate summary after cleanup

```bash
python scripts/summarize_scalability_results.py --results-root results --output-dir results/all_results_summary --print-table --save-csv --save-json --save-markdown --table-mode compact
```

---

## 8. Evaluation Checklist

- Fast runs: use while debugging.
- Balanced runs: use for report tables.
- Quality runs: use only for final confirmation.
- Multi-seed runs: use to prove stability.
- Ablation runs: use to prove each component matters.
- Summary commands: use to generate final CSV/Markdown tables.
- Cleanup commands: use after important outputs are saved.
- Always check: F-score, MF, DCV, Target Coverage, Spread, Runtime.

## Built-In Protected Attributes

- `twitter`, `synth2`, `synth3`, `rice_subset`: `color`
- `graph_spa_500_0`: `region`, `ethnicity`, `age`, `gender`, `status`

## Output Interpretation

Use `DCV` and `Target Coverage Ratio` to answer whether protected groups met their targets. Use `MF` to understand the weakest normalized group influence after the shortfall target is met. Use `fairness_gate_status` to know whether the stricter professor-priority policy accepted the run.

For the representative Twitter result, the algorithm met all group influence targets but did not pass the strict scalability gate. The practical next experiments are a budget sweep, higher `fair_ris_score_weight`, higher `weak_group_bonus_weight`, or support-aware fairness settings for strongly imbalanced protected groups.
