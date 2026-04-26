# ML-FIM Sweep Report

- Report: graph_spa_500_0_all_attrs_b40_seed42
- Dataset: graph_spa_500_0
- Protected attributes: region, ethnicity, gender, age, status
- Budgets: 40
- Random seeds: 42
- Stacks: community_aware_fair_greedy, graphsage_fair_ris_hybrid, gcn_fair_ris_hybrid, node2vec_xgboost, deepwalk_mlp, line_fast_ml
- Raw rows: 30

## Aggregate Ranking

```text
dataset          protected_attribute  budget  rank  stack_name                    mean_f_score  mean_mf  mean_dcv  mean_spread  mean_runtime  f_wins  runtime_wins  ok  failed
graph_spa_500_0  region               40      1     community_aware_fair_greedy   0.0170        0.0647   0.0307   41.345       41.715        1       0             1   0
graph_spa_500_0  region               40      2     node2vec_xgboost              0.0139        0.0657   0.0379   42.635       6.519         0       0             1   0
graph_spa_500_0  region               40      3     graphsage_fair_ris_hybrid     0.0086        0.0683   0.0512   42.115       5.085         0       1             1   0
graph_spa_500_0  region               40      4     gcn_fair_ris_hybrid           0.0034        0.0610   0.0542   42.120       5.121         0       0             1   0
graph_spa_500_0  region               40      5     line_fast_ml                  -0.1324       0.0000   0.2648   41.925       26.010        0       0             1   0
graph_spa_500_0  region               40      6     deepwalk_mlp                  -0.2190       0.0000   0.4380   41.825       30.249        0       0             1   0
graph_spa_500_0  ethnicity            40      1     gcn_fair_ris_hybrid           0.0305        0.0787   0.0177   42.400       4.599         1       0             1   0
graph_spa_500_0  ethnicity            40      2     graphsage_fair_ris_hybrid     0.0282        0.0783   0.0219   42.160       4.589         0       1             1   0
graph_spa_500_0  ethnicity            40      3     node2vec_xgboost              0.0242        0.0779   0.0296   42.335       5.021         0       0             1   0
graph_spa_500_0  ethnicity            40      4     deepwalk_mlp                  0.0171        0.0690   0.0348   41.775       26.292        0       0             1   0
graph_spa_500_0  ethnicity            40      5     community_aware_fair_greedy   0.0117        0.0681   0.0448   41.435       37.073        0       0             1   0
graph_spa_500_0  ethnicity            40      6     line_fast_ml                  -0.0192       0.0478   0.0861   41.955       26.457        0       0             1   0
graph_spa_500_0  gender               40      1     node2vec_xgboost              0.0390        0.0843   0.0062   42.690       4.851         1       0             1   0
graph_spa_500_0  gender               40      2     gcn_fair_ris_hybrid           0.0387        0.0833   0.0059   42.155       4.459         0       1             1   0
graph_spa_500_0  gender               40      3     graphsage_fair_ris_hybrid     0.0383        0.0833   0.0067   42.205       4.527         0       0             1   0
graph_spa_500_0  gender               40      4     community_aware_fair_greedy   0.0339        0.0826   0.0148   42.545       37.412        0       0             1   0
graph_spa_500_0  gender               40      5     line_fast_ml                  0.0320        0.0811   0.0170   41.955       23.305        0       0             1   0
graph_spa_500_0  gender               40      6     deepwalk_mlp                  0.0179        0.0769   0.0411   41.915       26.834        0       0             1   0
graph_spa_500_0  age                  40      1     community_aware_fair_greedy   0.0293        0.0761   0.0175   41.405       37.528        1       0             1   0
graph_spa_500_0  age                  40      2     node2vec_xgboost              0.0236        0.0735   0.0262   42.645       5.167         0       0             1   0
... 10 more row(s)
```

## Instant Insights

- Best overall stack: node2vec_xgboost
- Best ML-only stack: node2vec_xgboost
- Best fairness-first stack: graphsage_fair_ris_hybrid
- Best spread-first stack: node2vec_xgboost
- Best runtime-first stack: graphsage_fair_ris_hybrid
- Best quality-runtime trade-off: node2vec_xgboost
- community_aware_fair_greedy wins most often by F-score (2 run group win(s)).
- graphsage_fair_ris_hybrid wins most often by runtime (4 run group win(s)).
- Close result; not decisive. Top two mean F-scores differ by 0.0004.
- GraphSAGE+RIS is consistently close to the best baseline on mean F-score.
- Shallow ML fairness collapse warning: deepwalk_mlp, line_fast_ml
- Budget size changes the winner: budget 40 -> community_aware_fair_greedy; budget 40 -> gcn_fair_ris_hybrid; budget 40 -> node2vec_xgboost; budget 40 -> community_aware_fair_greedy; budget 40 -> graphsage_fair_ris_hybrid
- Winner differences look stable under the configured seeds.

## Final Recommendation

- best_overall_stack=node2vec_xgboost
- best_ml_only_stack=node2vec_xgboost
- best_fairness_first_stack=graphsage_fair_ris_hybrid
- best_spread_first_stack=node2vec_xgboost
- best_runtime_first_stack=graphsage_fair_ris_hybrid
- best_quality_runtime_tradeoff=node2vec_xgboost
- final_recommendation=node2vec_xgboost

## Existing Evaluation Report

FIM Result Evaluation
=================================================================================================================
Report: graph_spa_500_0_all_attrs_b40_seed42
Inputs: ml_fim_sweep_raw_runs
Rows loaded: 30
Ranking mode: fim_default

Sweep Summary
------------------------------------------------------------------------
- community_aware_fair_greedy: 2 overall win(s)
- gcn_fair_ris_hybrid: 1 overall win(s)
- node2vec_xgboost: 1 overall win(s)
- graphsage_fair_ris_hybrid: 1 overall win(s)

Group 1: dataset=graph_spa_500_0 | protected_attribute=region | budget=40
------------------------------------------------------------------------
Rows: 6 methods | ranking=fim_default | aggregated_runs=no

Rank  Method                         F-score   MF      DCV     Spread   Extra   Runtime
1     community_aware_fair_greedy    0.0170   0.0647  0.0307  41.345   1.345   41.715s
2     node2vec_xgboost               0.0139   0.0657  0.0379  42.635   2.635   6.519s
3     graphsage_fair_ris_hybrid      0.0086   0.0683  0.0512  42.115   2.115   5.085s
4     gcn_fair_ris_hybrid            0.0034   0.0610  0.0542  42.120   2.120   5.121s
5     line_fast_ml                   -0.1324  0.0000  0.2648  41.925   1.925   26.010s
6     deepwalk_mlp                   -0.2190  0.0000  0.4380  41.825   1.825   30.249s

Instant Insights
- Overall winner: community_aware_fair_greedy
- Best spread: node2vec_xgboost
- Best fairness profile: graphsage_fair_ris_hybrid
- Fastest method: graphsage_fair_ris_hybrid
- Result closeness: moderate overall result; top f_score gap is 0.0031
- Trade-off: node2vec_xgboost delivers the best spread, but community_aware_fair_greedy keeps the stronger fairness-adjusted score.
- Warning: line_fast_ml, deepwalk_mlp shows a fairness collapse signal.
- Best practical choice: community_aware_fair_greedy
- Best interpretable baseline: community_aware_fair_greedy

Per-Method Notes
- community_aware_fair_greedy: Best overall fairness-adjusted result; Best interpretable baseline
- node2vec_xgboost: Best spread-oriented method, but not best fairness-adjusted method
- graphsage_fair_ris_hybrid: Best fairness-first method; Fastest evaluated method
- gcn_fair_ris_hybrid: Competitive comparison method
- line_fast_ml: Fairness collapse warning: MF near zero; Fairness collapse warning: DCV extremely high; Negative F-score; not recommended as default
- deepwalk_mlp: Fairness collapse warning: MF near zero; Fairness collapse warning: DCV extremely high; Negative F-score; not recommended as default

Recommendation
- best_overall_method=community_aware_fair_greedy
- best_fairness_first_method=graphsage_fair_ris_hybrid
- best_spread_first_method=node2vec_xgboost
- best_runtime_first_method=graphsage_fair_ris_hybrid
- best_interpretable_baseline=community_aware_fair_greedy
- best_exploratory_comparator=n/a
- best_practical_choice=community_aware_fair_greedy

Group 2: dataset=graph_spa_500_0 | protected_attribute=ethnicity | budget=40
------------------------------------------------------------------------
Rows: 6 methods | ranking=fim_default | aggregated_runs=no

Rank  Method                         F-score   MF      DCV     Spread   Extra   Runtime
1     gcn_fair_ris_hybrid            0.0305   0.0787  0.0177  42.400   2.400   4.599s
2     graphsage_fair_ris_hybrid      0.0282   0.0783  0.0219  42.160   2.160   4.589s
3     node2vec_xgboost               0.0242   0.0779  0.0296  42.335   2.335   5.021s
4     deepwalk_mlp                   0.0171   0.0690  0.0348  41.775   1.775   26.292s
5     community_aware_fair_greedy    0.0117   0.0681  0.0448  41.435   1.435   37.073s
6     line_fast_ml                   -0.0192  0.0478  0.0861  41.955   1.955   26.457s

Instant Insights
- Overall winner: gcn_fair_ris_hybrid
- Best spread: gcn_fair_ris_hybrid
- Best fairness profile: gcn_fair_ris_hybrid
- Fastest method: graphsage_fair_ris_hybrid
- Result closeness: moderate overall result; top f_score gap is 0.0023
- Warning: line_fast_ml shows a fairness collapse signal.
- Best practical choice: gcn_fair_ris_hybrid
- Best interpretable baseline: community_aware_fair_greedy

Per-Method Notes
- gcn_fair_ris_hybrid: Best overall fairness-adjusted result
- graphsage_fair_ris_hybrid: Fastest evaluated method
- node2vec_xgboost: Competitive comparison method
- deepwalk_mlp: Competitive comparison method
- community_aware_fair_greedy: Best interpretable baseline
- line_fast_ml: Negative F-score; not recommended as default

Recommendation
- best_overall_method=gcn_fair_ris_hybrid
- best_fairness_first_method=gcn_fair_ris_hybrid
- best_spread_first_method=gcn_fair_ris_hybrid
- best_runtime_first_method=graphsage_fair_ris_hybrid
- best_interpretable_baseline=community_aware_fair_greedy
- best_exploratory_comparator=n/a
- best_practical_choice=gcn_fair_ris_hybrid

Group 3: dataset=graph_spa_500_0 | protected_attribute=gender | budget=40
------------------------------------------------------------------------
Rows: 6 methods | ranking=fim_default | aggregated_runs=no

Rank  Method                         F-score   MF      DCV     Spread   Extra   Runtime
1     node2vec_xgboost               0.0390   0.0843  0.0062  42.690   2.690   4.851s
2     gcn_fair_ris_hybrid            0.0387   0.0833  0.0059  42.155   2.155   4.459s
3     graphsage_fair_ris_hybrid      0.0383   0.0833  0.0067  42.205   2.205   4.527s
4     community_aware_fair_greedy    0.0339   0.0826  0.0148  42.545   2.545   37.412s
5     line_fast_ml                   0.0320   0.0811  0.0170  41.955   1.955   23.305s
6     deepwalk_mlp                   0.0179   0.0769  0.0411  41.915   1.915   26.834s

Instant Insights
- Overall winner: node2vec_xgboost
- Best spread: node2vec_xgboost
- Best fairness profile: node2vec_xgboost
- Fastest method: gcn_fair_ris_hybrid
- Result closeness: close overall result; top f_score gap is 0.0003
- Best practical choice: node2vec_xgboost
- Best interpretable baseline: community_aware_fair_greedy

Per-Method Notes
- node2vec_xgboost: Best overall fairness-adjusted result
- gcn_fair_ris_hybrid: Fastest evaluated method
- graphsage_fair_ris_hybrid: Competitive comparison method
- community_aware_fair_greedy: Best interpretable baseline
- line_fast_ml: Competitive comparison method
- deepwalk_mlp: Competitive comparison method

Recommendation
- best_overall_method=node2vec_xgboost
- best_fairness_first_method=node2vec_xgboost
- best_spread_first_method=node2vec_xgboost
- best_runtime_first_method=gcn_fair_ris_hybrid
- best_interpretable_baseline=community_aware_fair_greedy
- best_exploratory_comparator=n/a
- best_practical_choice=node2vec_xgboost

Group 4: dataset=graph_spa_500_0 | protected_attribute=age | budget=40
------------------------------------------------------------------------
Rows: 6 methods | ranking=fim_default | aggregated_runs=no

Rank  Method                         F-score   MF      DCV     Spread   Extra   Runtime
1     community_aware_fair_greedy    0.0293   0.0761  0.0175  41.405   1.405   37.528s
2     node2vec_xgboost               0.0236   0.0735  0.0262  42.645   2.645   5.167s
3     graphsage_fair_ris_hybrid      0.0200   0.0744  0.0343  42.005   2.005   4.564s
4     gcn_fair_ris_hybrid            0.0149   0.0719  0.0421  42.310   2.310   4.715s
5     deepwalk_mlp                   -0.0166  0.0506  0.0839  41.975   1.975   27.968s
6     line_fast_ml                   -0.0301  0.0549  0.1152  42.110   2.110   25.338s

Instant Insights
- Overall winner: community_aware_fair_greedy
- Best spread: node2vec_xgboost
- Best fairness profile: community_aware_fair_greedy
- Fastest method: graphsage_fair_ris_hybrid
- Result closeness: moderate overall result; top f_score gap is 0.0056
- Trade-off: node2vec_xgboost delivers the best spread, but community_aware_fair_greedy keeps the stronger fairness-adjusted score.
- Warning: deepwalk_mlp, line_fast_ml shows a fairness collapse signal.
- Best practical choice: community_aware_fair_greedy
- Best interpretable baseline: community_aware_fair_greedy

Per-Method Notes
- community_aware_fair_greedy: Best overall fairness-adjusted result; Best interpretable baseline
- node2vec_xgboost: Best spread-oriented method, but not best fairness-adjusted method
- graphsage_fair_ris_hybrid: Fastest evaluated method
- gcn_fair_ris_hybrid: Competitive comparison method
- deepwalk_mlp: Negative F-score; not recommended as default
- line_fast_ml: Negative F-score; not recommended as default

Recommendation
- best_overall_method=community_aware_fair_greedy
- best_fairness_first_method=community_aware_fair_greedy
- best_spread_first_method=node2vec_xgboost
- best_runtime_first_method=graphsage_fair_ris_hybrid
- best_interpretable_baseline=community_aware_fair_greedy
- best_exploratory_comparator=n/a
- best_practical_choice=community_aware_fair_greedy

Group 5: dataset=graph_spa_500_0 | protected_attribute=status | budget=40
------------------------------------------------------------------------
Rows: 6 methods | ranking=fim_default | aggregated_runs=no

Rank  Method                         F-score   MF      DCV     Spread   Extra   Runtime
1     graphsage_fair_ris_hybrid      0.0357   0.0822  0.0108  42.500   2.500   4.607s
2     gcn_fair_ris_hybrid            0.0342   0.0819  0.0135  42.675   2.675   4.865s
3     deepwalk_mlp                   0.0338   0.0804  0.0128  41.820   1.820   28.269s
4     community_aware_fair_greedy    0.0333   0.0796  0.0130  41.420   1.420   34.783s
5     node2vec_xgboost               0.0321   0.0808  0.0165  42.475   2.475   5.050s
6     line_fast_ml                   -0.0852  0.0516  0.2220  42.020   2.020   24.776s

Instant Insights
- Overall winner: graphsage_fair_ris_hybrid
- Best spread: gcn_fair_ris_hybrid
- Best fairness profile: graphsage_fair_ris_hybrid
- Fastest method: graphsage_fair_ris_hybrid
- Result closeness: close overall result; top f_score gap is 0.0015
- Trade-off: gcn_fair_ris_hybrid delivers the best spread, but graphsage_fair_ris_hybrid keeps the stronger fairness-adjusted score.
- Warning: line_fast_ml shows a fairness collapse signal.
- Best practical choice: graphsage_fair_ris_hybrid
- Best interpretable baseline: community_aware_fair_greedy

Per-Method Notes
- graphsage_fair_ris_hybrid: Best overall fairness-adjusted result; Fastest evaluated method
- gcn_fair_ris_hybrid: Best spread-oriented method, but not best fairness-adjusted method
- deepwalk_mlp: Competitive comparison method
- community_aware_fair_greedy: Best interpretable baseline
- node2vec_xgboost: Competitive comparison method
- line_fast_ml: Negative F-score; not recommended as default

Recommendation
- best_overall_method=graphsage_fair_ris_hybrid
- best_fairness_first_method=graphsage_fair_ris_hybrid
- best_spread_first_method=gcn_fair_ris_hybrid
- best_runtime_first_method=graphsage_fair_ris_hybrid
- best_interpretable_baseline=community_aware_fair_greedy
- best_exploratory_comparator=n/a
- best_practical_choice=graphsage_fair_ris_hybrid
