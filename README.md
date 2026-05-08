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
DCV_shortfall = average(shortfall_ratio[g])
Target Coverage Ratio = groups_with_zero_shortfall / number_of_groups
```

`DCV_shortfall = 0.0` means no protected group is below its target. This is the primary fairness objective in the active stack.

`DCV_disparity` is different. It measures remaining population-proportional under-service after spread is allocated across groups. A run can have `DCV_shortfall = 0.0` and still have nonzero `DCV_disparity` if some groups receive more influence per population share than others.

The shortfall-first F-score used in the README result is:

```text
F-score = MF - DCV_shortfall - 0.25 * DCV_disparity
```

For the representative result:

```text
F-score = 0.031717 - 0.000000 - 0.25 * 0.016394
        = 0.027618
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

1. Prefer seed sets where all groups meet their targets.
2. If not all groups meet targets, prefer higher target coverage.
3. Then prefer lower `DCV_shortfall`.
4. Then prefer better `MF`, lower disparity, higher F-score, and higher spread.

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
- Preserve `DCV_shortfall = 0.0`.
- Preserve `Target Coverage Ratio = 1.0`.
- Avoid unacceptable spread or disparity degradation.

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

## Built-In Protected Attributes

- `twitter`, `synth2`, `synth3`, `rice_subset`: `color`
- `graph_spa_500_0`: `region`, `ethnicity`, `age`, `gender`, `status`

## Output Interpretation

Use `DCV_shortfall` and `Target Coverage Ratio` to answer whether protected groups met their targets. Use `MF` and `DCV_disparity` to understand remaining inequality after the shortfall target is met. Use `fairness_gate_status` to know whether the stricter professor-priority policy accepted the run.

For the representative Twitter result, the algorithm met all group influence targets but did not pass the strict scalability gate. The practical next experiments are a budget sweep, higher `fair_ris_score_weight`, higher `weak_group_bonus_weight`, or support-aware fairness settings for strongly imbalanced protected groups.
