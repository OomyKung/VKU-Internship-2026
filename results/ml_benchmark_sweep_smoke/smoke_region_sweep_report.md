# ML-FIM Sweep Report

- Report: smoke_region_sweep
- Dataset: graph_spa_500_0
- Protected attributes: region
- Budgets: 10, 20
- Random seeds: 1, 7
- Stacks: line_fast_ml
- Raw rows: 4

## Aggregate Ranking

```text
dataset          protected_attribute  budget  rank  stack_name                    mean_f_score  mean_mf  mean_dcv  mean_spread  mean_runtime  f_wins  runtime_wins  ok  failed
graph_spa_500_0  region               10      1     line_fast_ml                  -0.3069       0.0000   0.6139   11.000       2.124         2       2             2   0
graph_spa_500_0  region               20      1     line_fast_ml                  -0.2618       0.0000   0.5237   20.667       3.725         2       2             2   0
```

## Instant Insights

- Best overall stack: line_fast_ml
- Best ML-only stack: line_fast_ml
- Best fairness-first stack: line_fast_ml
- Best spread-first stack: line_fast_ml
- Best runtime-first stack: line_fast_ml
- Best quality-runtime trade-off: line_fast_ml
- line_fast_ml wins most often by F-score (4 run group win(s)).
- line_fast_ml wins most often by runtime (4 run group win(s)).
- Shallow ML fairness collapse warning: line_fast_ml
- Some winner differences are seed-sensitive based on F-score standard deviation.

## Final Recommendation

- best_overall_stack=line_fast_ml
- best_ml_only_stack=line_fast_ml
- best_fairness_first_stack=line_fast_ml
- best_spread_first_stack=line_fast_ml
- best_runtime_first_stack=line_fast_ml
- best_quality_runtime_tradeoff=line_fast_ml
- final_recommendation=line_fast_ml

## Existing Evaluation Report

FIM Result Evaluation
====================================================================================================
Report: smoke_region_sweep
Inputs: ml_fim_sweep_raw_runs
Rows loaded: 4
Ranking mode: fim_default

Sweep Summary
------------------------------------------------------------------------
- line_fast_ml: 2 overall win(s)
- Most stable by mean F-score std: line_fast_ml

Group 1: dataset=graph_spa_500_0 | protected_attribute=region | budget=10
------------------------------------------------------------------------
Rows: 1 methods | ranking=fim_default | aggregated_runs=yes

Rank  Method                         n   F-score   MF      DCV     Spread   Extra   Runtime
1     line_fast_ml                   2   -0.3069  0.0000  0.6139  11.000   1.000   2.124s

Instant Insights
- Overall winner: line_fast_ml
- Best spread: line_fast_ml
- Best fairness profile: line_fast_ml
- Fastest method: line_fast_ml
- Warning: line_fast_ml shows a fairness collapse signal.
- Best practical choice: line_fast_ml

Per-Method Notes
- line_fast_ml: Best overall fairness-adjusted result; Fastest evaluated method; Fairness collapse warning: MF near zero; Fairness collapse warning: DCV extremely high; Negative F-score; not recommended as default

Recommendation
- best_overall_method=line_fast_ml
- best_fairness_first_method=line_fast_ml
- best_spread_first_method=line_fast_ml
- best_runtime_first_method=line_fast_ml
- best_interpretable_baseline=n/a
- best_exploratory_comparator=n/a
- best_practical_choice=line_fast_ml

Group 2: dataset=graph_spa_500_0 | protected_attribute=region | budget=20
------------------------------------------------------------------------
Rows: 1 methods | ranking=fim_default | aggregated_runs=yes

Rank  Method                         n   F-score   MF      DCV     Spread   Extra   Runtime
1     line_fast_ml                   2   -0.2618  0.0000  0.5237  20.667   0.667   3.725s

Instant Insights
- Overall winner: line_fast_ml
- Best spread: line_fast_ml
- Best fairness profile: line_fast_ml
- Fastest method: line_fast_ml
- Warning: line_fast_ml shows a fairness collapse signal.
- Best practical choice: line_fast_ml

Per-Method Notes
- line_fast_ml: Best overall fairness-adjusted result; Fastest evaluated method; Fairness collapse warning: MF near zero; Fairness collapse warning: DCV extremely high; Negative F-score; not recommended as default

Recommendation
- best_overall_method=line_fast_ml
- best_fairness_first_method=line_fast_ml
- best_spread_first_method=line_fast_ml
- best_runtime_first_method=line_fast_ml
- best_interpretable_baseline=n/a
- best_exploratory_comparator=n/a
- best_practical_choice=line_fast_ml
