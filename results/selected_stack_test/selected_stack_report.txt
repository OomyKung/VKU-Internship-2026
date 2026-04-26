Selected FIM Stack Report
========================================================================

Selection stage
------------------------------------------------------------------------
report=selected_fim_stack
selected_stack_from_benchmark=graphsage_fair_ris_hybrid
benchmark_f_score=0.016024
benchmark_mf=0.067049
benchmark_dcv=0.035002
benchmark_runtime=20.120
benchmark_quality_runtime_score=1.000000
selection_policy=fairness_runtime_tradeoff
selection_reason=within 0.003000 F-score of best valid stack (0.016024), within DCV delta 0.010000, MF ratio >= 0.950000, and lowest runtime among close rows

Fallback
------------------------------------------------------------------------
fallback_stack=community_aware_fair_greedy
fallback_benchmark_f_score=0.014380
fallback_benchmark_mf=0.068337
fallback_benchmark_dcv=0.039576
fallback_benchmark_runtime=48.317
fallback_reason=forced fairness-first fallback stack from --fallback-stack=community_aware_fair_greedy

Validation stage
------------------------------------------------------------------------
candidate_stack=graphsage_fair_ris_hybrid
fallback_stack=community_aware_fair_greedy
validation_seeds=[7, 21, 42]
candidate_mean_f_score=0.008603
candidate_std_f_score=0.002916
candidate_mean_mf=0.063741
candidate_mean_dcv=0.046536
candidate_mean_spread=42.101667
candidate_mean_runtime=6.091
fallback_mean_f_score=0.014551
fallback_std_f_score=0.002185
fallback_mean_mf=0.066302
fallback_mean_dcv=0.037200
fallback_mean_spread=41.435000
fallback_mean_runtime=56.574
f_score_gap=-0.005949
dcv_delta=0.009336
mf_ratio=0.961380
runtime_speedup=9.288826
accepted=False
final_selected_stack=community_aware_fair_greedy
final_reason=rejected_after_validation: F-score dropped beyond validation threshold; fallback_selected=community_aware_fair_greedy

Selected pipeline
------------------------------------------------------------------------
decision_status=fallback_selected
community_method=leiden
embedding_method=none
ranking_model=fairness_weighted_greedy
optimizer_mode=local_search
spread_estimator_search=monte_carlo
spread_estimator_final=monte_carlo
use_repair=False
use_swap_local_search=True

Benchmark row used for selection
------------------------------------------------------------------------
{
  "dataset": "graph_spa_500_0",
  "protected_attribute": "region",
  "budget": 40,
  "stack_name": "graphsage_fair_ris_hybrid",
  "status": "ok",
  "f_score": 0.0160236816325902,
  "std_f_score": 0.0010606914634351,
  "mf": 0.0670494039925319,
  "std_mf": 0.0020890806138757,
  "dcv": 0.0350020407273515,
  "std_dcv": 0.00351385738399,
  "total_spread": 42.241,
  "std_spread": 0.2596728711282725,
  "mean_extra": 2.2410000000000023,
  "std_extra": 0.2596728711282718,
  "runtime_seconds": 20.12001303999996,
  "std_runtime": 1.165959980568858,
  "win_count_by_f_score": 1,
  "win_count_by_runtime": 0,
  "number_of_successful_runs": 5,
  "number_of_failed_or_skipped_runs": 0,
  "variant_type": "ml_guided",
  "embedding_method": "graphsage",
  "ranking_model": "graphsage",
  "community_method": "leiden",
  "clustering_method": "none",
  "optimizer_mode": "hybrid_si_ea",
  "debias_mode": "worst_group_boost",
  "spread_estimator_search": "fairness_aware_ris",
  "spread_estimator_final": "monte_carlo",
  "key_enabled_modules": "diffusion=ic | community=leiden | embedding=graphsage | clustering=none | clustering_input_mode=none | ranking=graphsage | optimizer=hybrid_si_ea | debias_mode=worst_group_boost | search_estimator=fairness_aware_ris | final_estimator=monte_carlo",
  "normalized_f_score": 1.0,
  "normalized_runtime": 0.2528962363729004,
  "quality_runtime_score": 1.0
}

Final result
------------------------------------------------------------------------
final_selected_stack=community_aware_fair_greedy
total_spread=41.345
extra_spread=1.3449999999999989
MF=0.064739336492891
DCV=0.030735419040623373
F-score=0.017001958726133814
runtime_seconds=50.36499499999991
final_spread_estimator=monte_carlo
final_validation_warning=False
final_validation_warning_reasons=[]

Stack execution report
------------------------------------------------------------------------
FIM Algorithm-Stack Permutation Comparison
------------------------------------------------------------------------
Protected attribute: region
Budget: 40
Final estimator: monte_carlo | mc_runs_eval=200

1. community_aware_fair_greedy [ok]
   spread=41.3450 | extra=1.3450 | MF=0.0647 | DCV=0.0307 | F-score=0.0170 | runtime=50.365s
   modules: diffusion=ic | community=leiden | embedding=none | clustering=none | ranking=fairness_weighted_greedy | optimizer=local_search | debias=none
   notes=initial_rr=[12, 13, 14, 15, 17, 18, 21, 223, 229, 238, 24, 241, 264, 265, 267, 268, 269, 271, 278, 281, 284, 287, 30, 303, 312, 326, 33, 331, 38, 39, 40, 44, 464, 465, 481, 485, 5, 53, 59, 6]; greedy=[106, 116, 164, 178, 2, 21, 212, 227, 229, 23, 243, 250, 260, 261, 289, 311, 315, 34, 354, 364, 380, 391, 41, 419, 420, 423, 426, 428, 431, 44, 461, 468, 475, 480, 488, 495, 53, 73, 75, 9]; communities=19; modularity=0.802180

Recommendations
------------------------------------------------------------------------
best_interpretable_baseline=community_aware_fair_greedy
best_practical_ml_default=n/a
best_exploratory_fairness_heavy=n/a
