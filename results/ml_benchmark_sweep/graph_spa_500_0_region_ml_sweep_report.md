# ML-FIM Sweep Report

- Report: graph_spa_500_0_region_ml_sweep
- Dataset: graph_spa_500_0
- Protected attributes: region
- Budgets: 10, 20, 40, 100, 200
- Random seeds: 1, 7, 21, 42, 99
- Stacks: community_aware_fair_greedy, graphsage_fair_ris_hybrid, gcn_fair_ris_hybrid, node2vec_xgboost, deepwalk_mlp, line_fast_ml
- Raw rows: 150

## Aggregate Ranking

```text
dataset          protected_attribute  budget  rank  stack_name                    mean_f_score  mean_mf  mean_dcv  mean_spread  mean_runtime  f_wins  runtime_wins  ok  failed
graph_spa_500_0  region               10      1     gcn_fair_ris_hybrid           -0.1373       0.0001   0.2746   10.347       11.084        3       0             5   0
graph_spa_500_0  region               10      2     graphsage_fair_ris_hybrid     -0.1488       0.0001   0.2976   10.506       11.148        2       0             5   0
graph_spa_500_0  region               10      3     node2vec_xgboost              -0.1756       0.0000   0.3512   10.899       7.013         0       0             5   0
graph_spa_500_0  region               10      4     community_aware_fair_greedy   -0.1921       0.0000   0.3841   10.618       6.893         0       0             5   0
graph_spa_500_0  region               10      5     line_fast_ml                  -0.3408       0.0000   0.6817   10.577       4.124         0       5             5   0
graph_spa_500_0  region               10      6     deepwalk_mlp                  -0.3531       0.0000   0.7061   10.579       8.778         0       0             5   0
graph_spa_500_0  region               20      1     gcn_fair_ris_hybrid           -0.0216       0.0225   0.0658   21.262       14.676        2       0             5   0
graph_spa_500_0  region               20      2     graphsage_fair_ris_hybrid     -0.0218       0.0220   0.0657   21.191       14.352        1       0             5   0
graph_spa_500_0  region               20      3     node2vec_xgboost              -0.0315       0.0178   0.0807   21.440       8.238         2       5             5   0
graph_spa_500_0  region               20      4     community_aware_fair_greedy   -0.0501       0.0144   0.1147   20.991       17.869        0       0             5   0
graph_spa_500_0  region               20      5     line_fast_ml                  -0.2828       0.0000   0.5657   21.126       9.953         0       0             5   0
graph_spa_500_0  region               20      6     deepwalk_mlp                  -0.3108       0.0000   0.6215   21.132       14.184        0       0             5   0
graph_spa_500_0  region               40      1     graphsage_fair_ris_hybrid     0.0160        0.0670   0.0350   42.241       20.120        1       0             5   0
graph_spa_500_0  region               40      2     community_aware_fair_greedy   0.0144        0.0683   0.0396   41.494       48.317        1       0             5   0
graph_spa_500_0  region               40      3     gcn_fair_ris_hybrid           0.0140        0.0662   0.0383   42.168       20.379        1       0             5   0
graph_spa_500_0  region               40      4     node2vec_xgboost              0.0020        0.0548   0.0508   42.471       10.575        2       5             5   0
graph_spa_500_0  region               40      5     line_fast_ml                  -0.1966       0.0000   0.3933   42.082       26.787        0       0             5   0
graph_spa_500_0  region               40      6     deepwalk_mlp                  -0.2536       0.0000   0.5072   41.662       28.907        0       0             5   0
graph_spa_500_0  region               100     1     community_aware_fair_greedy   0.0887        0.1902   0.0129   103.063      210.832       3       0             5   0
graph_spa_500_0  region               100     2     gcn_fair_ris_hybrid           0.0860        0.1871   0.0150   103.678      34.224        1       0             5   0
... 10 more row(s)
```

## Instant Insights

- Best overall stack: gcn_fair_ris_hybrid
- Best ML-only stack: gcn_fair_ris_hybrid
- Best fairness-first stack: community_aware_fair_greedy
- Best spread-first stack: node2vec_xgboost
- Best runtime-first stack: node2vec_xgboost
- Best quality-runtime trade-off: gcn_fair_ris_hybrid
- gcn_fair_ris_hybrid wins most often by F-score (8 run group win(s)).
- node2vec_xgboost wins most often by runtime (20 run group win(s)).
- Close result; not decisive. Top two mean F-scores differ by 0.0023.
- GraphSAGE+RIS is consistently close to the best baseline on mean F-score.
- Shallow ML fairness collapse warning: node2vec_xgboost, deepwalk_mlp, line_fast_ml
- Budget size changes the winner: budget 10 -> gcn_fair_ris_hybrid; budget 20 -> gcn_fair_ris_hybrid; budget 40 -> graphsage_fair_ris_hybrid; budget 100 -> community_aware_fair_greedy; budget 200 -> community_aware_fair_greedy
- community_aware_fair_greedy wins only at larger budgets in this sweep.
- Some winner differences are seed-sensitive based on F-score standard deviation.

## Final Recommendation

- best_overall_stack=gcn_fair_ris_hybrid
- best_ml_only_stack=gcn_fair_ris_hybrid
- best_fairness_first_stack=community_aware_fair_greedy
- best_spread_first_stack=node2vec_xgboost
- best_runtime_first_stack=node2vec_xgboost
- best_quality_runtime_tradeoff=gcn_fair_ris_hybrid
- final_recommendation=gcn_fair_ris_hybrid

## Existing Evaluation Report

FIM Result Evaluation
================================================================================================================
Report: graph_spa_500_0_region_ml_sweep
Inputs: ml_fim_sweep_raw_runs
Rows loaded: 150
Ranking mode: fim_default

Sweep Summary
------------------------------------------------------------------------
- gcn_fair_ris_hybrid: 2 overall win(s)
- community_aware_fair_greedy: 2 overall win(s)
- graphsage_fair_ris_hybrid: 1 overall win(s)
- Most stable by mean F-score std: gcn_fair_ris_hybrid

Group 1: dataset=graph_spa_500_0 | protected_attribute=region | budget=10
------------------------------------------------------------------------
Rows: 6 methods | ranking=fim_default | aggregated_runs=yes

Rank  Method                         n   F-score   MF      DCV     Spread   Extra   Runtime
1     gcn_fair_ris_hybrid            5   -0.1373  0.0001  0.2746  10.347   0.347   11.084s
2     graphsage_fair_ris_hybrid      5   -0.1488  0.0001  0.2976  10.506   0.506   11.148s
3     node2vec_xgboost               5   -0.1756  0.0000  0.3512  10.899   0.899   7.013s
4     community_aware_fair_greedy    5   -0.1921  0.0000  0.3841  10.618   0.618   6.893s
5     line_fast_ml                   5   -0.3408  0.0000  0.6817  10.577   0.577   4.124s
6     deepwalk_mlp                   5   -0.3531  0.0000  0.7061  10.579   0.579   8.778s

Instant Insights
- Overall winner: gcn_fair_ris_hybrid
- Best spread: node2vec_xgboost
- Best fairness profile: gcn_fair_ris_hybrid
- Fastest method: line_fast_ml
- Result closeness: decisive overall result; top f_score gap is 0.0115
- Trade-off: node2vec_xgboost delivers the best spread, but gcn_fair_ris_hybrid keeps the stronger fairness-adjusted score.
- Warning: gcn_fair_ris_hybrid, graphsage_fair_ris_hybrid, node2vec_xgboost, community_aware_fair_greedy, line_fast_ml, deepwalk_mlp shows a fairness collapse signal.
- Best practical choice: gcn_fair_ris_hybrid
- Best interpretable baseline: community_aware_fair_greedy

Per-Method Notes
- gcn_fair_ris_hybrid: Best overall fairness-adjusted result; Fairness collapse warning: MF near zero; Fairness collapse warning: DCV extremely high; Negative F-score; not recommended as default
- graphsage_fair_ris_hybrid: Fairness collapse warning: MF near zero; Fairness collapse warning: DCV extremely high; Negative F-score; not recommended as default
- node2vec_xgboost: Best spread-oriented method, but not best fairness-adjusted method; Fairness collapse warning: MF near zero; Fairness collapse warning: DCV extremely high; Negative F-score; not recommended as default
- community_aware_fair_greedy: Best interpretable baseline; Fairness collapse warning: MF near zero; Fairness collapse warning: DCV extremely high; Negative F-score; not recommended as default
- line_fast_ml: Fastest evaluated method; Fairness collapse warning: MF near zero; Fairness collapse warning: DCV extremely high; Negative F-score; not recommended as default
- deepwalk_mlp: Fairness collapse warning: MF near zero; Fairness collapse warning: DCV extremely high; Negative F-score; not recommended as default

Recommendation
- best_overall_method=gcn_fair_ris_hybrid
- best_fairness_first_method=gcn_fair_ris_hybrid
- best_spread_first_method=node2vec_xgboost
- best_runtime_first_method=line_fast_ml
- best_interpretable_baseline=community_aware_fair_greedy
- best_exploratory_comparator=n/a
- best_practical_choice=gcn_fair_ris_hybrid

Group 2: dataset=graph_spa_500_0 | protected_attribute=region | budget=20
------------------------------------------------------------------------
Rows: 6 methods | ranking=fim_default | aggregated_runs=yes

Rank  Method                         n   F-score   MF      DCV     Spread   Extra   Runtime
1     gcn_fair_ris_hybrid            5   -0.0216  0.0225  0.0658  21.262   1.262   14.676s
2     graphsage_fair_ris_hybrid      5   -0.0218  0.0220  0.0657  21.191   1.191   14.352s
3     node2vec_xgboost               5   -0.0315  0.0178  0.0807  21.440   1.440   8.238s
4     community_aware_fair_greedy    5   -0.0501  0.0144  0.1147  20.991   0.991   17.869s
5     line_fast_ml                   5   -0.2828  0.0000  0.5657  21.126   1.126   9.953s
6     deepwalk_mlp                   5   -0.3108  0.0000  0.6215  21.132   1.132   14.184s

Instant Insights
- Overall winner: gcn_fair_ris_hybrid
- Best spread: node2vec_xgboost
- Best fairness profile: gcn_fair_ris_hybrid
- Fastest method: node2vec_xgboost
- Result closeness: close overall result; top f_score gap is 0.0002
- Trade-off: node2vec_xgboost delivers the best spread, but gcn_fair_ris_hybrid keeps the stronger fairness-adjusted score.
- Warning: gcn_fair_ris_hybrid, graphsage_fair_ris_hybrid, node2vec_xgboost, community_aware_fair_greedy, line_fast_ml, deepwalk_mlp shows a fairness collapse signal.
- Best practical choice: gcn_fair_ris_hybrid
- Best interpretable baseline: community_aware_fair_greedy

Per-Method Notes
- gcn_fair_ris_hybrid: Best overall fairness-adjusted result; Negative F-score; not recommended as default
- graphsage_fair_ris_hybrid: Negative F-score; not recommended as default
- node2vec_xgboost: Best spread-oriented method, but not best fairness-adjusted method; Fastest evaluated method; Negative F-score; not recommended as default
- community_aware_fair_greedy: Best interpretable baseline; Negative F-score; not recommended as default
- line_fast_ml: Fairness collapse warning: MF near zero; Fairness collapse warning: DCV extremely high; Negative F-score; not recommended as default
- deepwalk_mlp: Fairness collapse warning: MF near zero; Fairness collapse warning: DCV extremely high; Negative F-score; not recommended as default

Recommendation
- best_overall_method=gcn_fair_ris_hybrid
- best_fairness_first_method=gcn_fair_ris_hybrid
- best_spread_first_method=node2vec_xgboost
- best_runtime_first_method=node2vec_xgboost
- best_interpretable_baseline=community_aware_fair_greedy
- best_exploratory_comparator=n/a
- best_practical_choice=gcn_fair_ris_hybrid

Group 3: dataset=graph_spa_500_0 | protected_attribute=region | budget=40
------------------------------------------------------------------------
Rows: 6 methods | ranking=fim_default | aggregated_runs=yes

Rank  Method                         n   F-score   MF      DCV     Spread   Extra   Runtime
1     graphsage_fair_ris_hybrid      5   0.0160   0.0670  0.0350  42.241   2.241   20.120s
2     community_aware_fair_greedy    5   0.0144   0.0683  0.0396  41.494   1.494   48.317s
3     gcn_fair_ris_hybrid            5   0.0140   0.0662  0.0383  42.168   2.168   20.379s
4     node2vec_xgboost               5   0.0020   0.0548  0.0508  42.471   2.471   10.575s
5     line_fast_ml                   5   -0.1966  0.0000  0.3933  42.082   2.082   26.787s
6     deepwalk_mlp                   5   -0.2536  0.0000  0.5072  41.662   1.662   28.907s

Instant Insights
- Overall winner: graphsage_fair_ris_hybrid
- Best spread: node2vec_xgboost
- Best fairness profile: community_aware_fair_greedy
- Fastest method: node2vec_xgboost
- Result closeness: close overall result; top f_score gap is 0.0016
- Trade-off: node2vec_xgboost delivers the best spread, but graphsage_fair_ris_hybrid keeps the stronger fairness-adjusted score.
- Warning: line_fast_ml, deepwalk_mlp shows a fairness collapse signal.
- Best practical choice: graphsage_fair_ris_hybrid
- Best interpretable baseline: community_aware_fair_greedy

Per-Method Notes
- graphsage_fair_ris_hybrid: Best overall fairness-adjusted result
- community_aware_fair_greedy: Best fairness-first method; Best interpretable baseline
- gcn_fair_ris_hybrid: Competitive comparison method
- node2vec_xgboost: Best spread-oriented method, but not best fairness-adjusted method; Fastest evaluated method
- line_fast_ml: Fairness collapse warning: MF near zero; Fairness collapse warning: DCV extremely high; Negative F-score; not recommended as default
- deepwalk_mlp: Fairness collapse warning: MF near zero; Fairness collapse warning: DCV extremely high; Negative F-score; not recommended as default

Recommendation
- best_overall_method=graphsage_fair_ris_hybrid
- best_fairness_first_method=community_aware_fair_greedy
- best_spread_first_method=node2vec_xgboost
- best_runtime_first_method=node2vec_xgboost
- best_interpretable_baseline=community_aware_fair_greedy
- best_exploratory_comparator=n/a
- best_practical_choice=graphsage_fair_ris_hybrid

Group 4: dataset=graph_spa_500_0 | protected_attribute=region | budget=100
------------------------------------------------------------------------
Rows: 6 methods | ranking=fim_default | aggregated_runs=yes

Rank  Method                         n   F-score   MF      DCV     Spread   Extra   Runtime
1     community_aware_fair_greedy    5   0.0887   0.1902  0.0129  103.063  3.063   210.832s
2     gcn_fair_ris_hybrid            5   0.0860   0.1871  0.0150  103.678  3.678   34.224s
3     graphsage_fair_ris_hybrid      5   0.0830   0.1855  0.0195  103.390  3.390   33.508s
4     node2vec_xgboost               5   0.0595   0.1508  0.0318  103.898  3.898   13.441s
5     line_fast_ml                   5   -0.0824  0.0006  0.1655  103.664  3.664   101.858s
6     deepwalk_mlp                   5   -0.1327  0.0004  0.2658  103.123  3.123   94.111s

Instant Insights
- Overall winner: community_aware_fair_greedy
- Best spread: node2vec_xgboost
- Best fairness profile: community_aware_fair_greedy
- Fastest method: node2vec_xgboost
- Result closeness: moderate overall result; top f_score gap is 0.0026
- Trade-off: node2vec_xgboost delivers the best spread, but community_aware_fair_greedy keeps the stronger fairness-adjusted score.
- Warning: line_fast_ml, deepwalk_mlp shows a fairness collapse signal.
- Best practical choice: community_aware_fair_greedy
- Best interpretable baseline: community_aware_fair_greedy

Per-Method Notes
- community_aware_fair_greedy: Best overall fairness-adjusted result; Best interpretable baseline
- gcn_fair_ris_hybrid: Competitive comparison method
- graphsage_fair_ris_hybrid: Competitive comparison method
- node2vec_xgboost: Best spread-oriented method, but not best fairness-adjusted method; Fastest evaluated method
- line_fast_ml: Fairness collapse warning: MF near zero; Negative F-score; not recommended as default
- deepwalk_mlp: Fairness collapse warning: MF near zero; Fairness collapse warning: DCV extremely high; Negative F-score; not recommended as default

Recommendation
- best_overall_method=community_aware_fair_greedy
- best_fairness_first_method=community_aware_fair_greedy
- best_spread_first_method=node2vec_xgboost
- best_runtime_first_method=node2vec_xgboost
- best_interpretable_baseline=community_aware_fair_greedy
- best_exploratory_comparator=n/a
- best_practical_choice=community_aware_fair_greedy

Group 5: dataset=graph_spa_500_0 | protected_attribute=region | budget=200
------------------------------------------------------------------------
Rows: 6 methods | ranking=fim_default | aggregated_runs=yes

Rank  Method                         n   F-score   MF      DCV     Spread   Extra   Runtime
1     community_aware_fair_greedy    5   0.1917   0.3911  0.0077  204.188  4.188   615.897s
2     graphsage_fair_ris_hybrid      5   0.1825   0.3763  0.0113  203.882  3.882   50.450s
3     gcn_fair_ris_hybrid            5   0.1813   0.3739  0.0112  203.703  3.703   50.448s
4     node2vec_xgboost               5   0.1017   0.2511  0.0477  203.911  3.911   16.210s
5     deepwalk_mlp                   5   0.0473   0.2010  0.1063  204.127  4.127   193.205s
6     line_fast_ml                   5   -0.0753  0.0396  0.1902  203.733  3.733   207.242s

Instant Insights
- Overall winner: community_aware_fair_greedy
- Best spread: community_aware_fair_greedy
- Best fairness profile: community_aware_fair_greedy
- Fastest method: node2vec_xgboost
- Result closeness: decisive overall result; top f_score gap is 0.0092
- Warning: line_fast_ml shows a fairness collapse signal.
- Best practical choice: community_aware_fair_greedy
- Best interpretable baseline: community_aware_fair_greedy

Per-Method Notes
- community_aware_fair_greedy: Best overall fairness-adjusted result; Best interpretable baseline
- graphsage_fair_ris_hybrid: Competitive comparison method
- gcn_fair_ris_hybrid: Competitive comparison method
- node2vec_xgboost: Fastest evaluated method
- deepwalk_mlp: Competitive comparison method
- line_fast_ml: Negative F-score; not recommended as default

Recommendation
- best_overall_method=community_aware_fair_greedy
- best_fairness_first_method=community_aware_fair_greedy
- best_spread_first_method=community_aware_fair_greedy
- best_runtime_first_method=node2vec_xgboost
- best_interpretable_baseline=community_aware_fair_greedy
- best_exploratory_comparator=n/a
- best_practical_choice=community_aware_fair_greedy
