# ML-FIM Sweep Report

- Report: smoke_skip_budget
- Dataset: graph_spa_500_0
- Protected attributes: region
- Budgets: 1000
- Random seeds: 7
- Stacks: line_fast_ml
- Raw rows: 1

## Aggregate Ranking

```text
dataset          protected_attribute  budget  rank  stack_name                    mean_f_score  mean_mf  mean_dcv  mean_spread  mean_runtime  f_wins  runtime_wins  ok  failed
graph_spa_500_0  region               1000    1     line_fast_ml                  -             -        -        -            -             0       0             0   1
```

## Instant Insights

- No successful runs were available for insight generation.

## Final Recommendation

- best_overall_stack=n/a
- best_ml_only_stack=n/a
- best_fairness_first_stack=n/a
- best_spread_first_stack=n/a
- best_runtime_first_stack=n/a
- best_quality_runtime_tradeoff=n/a
- final_recommendation=n/a

## Existing Evaluation Report

FIM Result Evaluation
====================================================================================================
Report: smoke_skip_budget
Inputs: ml_fim_sweep_raw_runs
Rows loaded: 1
Ranking mode: fim_default

Global Warnings
------------------------------------------------------------------------
- Expected metrics missing or empty: total_spread, extra_spread, mf, dcv, f_score, runtime_seconds

Group 1: dataset=graph_spa_500_0 | protected_attribute=region | budget=1000
------------------------------------------------------------------------
Rows: 0 methods | ranking=fim_default | aggregated_runs=no

No successful methods to rank.

Instant Insights
- No successful methods were available for evaluation.

Per-Method Notes
- No per-method notes were generated.

Warnings
- No successful methods were available for ranking in this group.

Skipped / Partial Rows
- line_fast_ml: budget 1000 exceeds graph node count 500.

Recommendation
- best_overall_method=n/a
- best_fairness_first_method=n/a
- best_spread_first_method=n/a
- best_runtime_first_method=n/a
- best_interpretable_baseline=n/a
- best_exploratory_comparator=n/a
- best_practical_choice=n/a
