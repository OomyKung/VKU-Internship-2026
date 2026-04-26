Selected FIM Stack Report
========================================================================

Selection stage
------------------------------------------------------------------------
report=path_resolution_smoke
selected_stack_from_benchmark=graphsage_fair_ris_hybrid
benchmark_f_score=0.016024
benchmark_mf=0.067049
benchmark_dcv=0.035002
benchmark_runtime=20.120
benchmark_quality_runtime_score=1.000000
selection_policy=fairness_runtime_tradeoff
selection_reason=within 0.003000 F-score of best valid stack (0.016024), within DCV delta 0.010000, MF ratio >= 0.950000, and lowest runtime among close rows

Validation stage
------------------------------------------------------------------------
candidate_stack=graphsage_fair_ris_hybrid
fallback_stack=None
validation_seeds=[]
candidate_mean_f_score=n/a
candidate_std_f_score=n/a
candidate_mean_mf=n/a
candidate_mean_dcv=n/a
candidate_mean_runtime=n/a
fallback_mean_f_score=n/a
fallback_std_f_score=n/a
fallback_mean_mf=n/a
fallback_mean_dcv=n/a
fallback_mean_runtime=n/a
f_score_gap=n/a
dcv_delta=n/a
runtime_speedup=n/a
accepted=None
final_selected_stack=graphsage_fair_ris_hybrid
final_reason=selected_from_benchmark_not_validated

Selected pipeline
------------------------------------------------------------------------
decision_status=selected_from_benchmark
community_method=leiden
embedding_method=graphsage
ranking_model=graphsage_plus_fair_ris
optimizer_mode=hybrid_si_ea
spread_estimator_search=fairness_aware_ris
spread_estimator_final=monte_carlo
use_repair=True
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
final_selected_stack=graphsage_fair_ris_hybrid
total_spread=42.0
extra_spread=2.0
MF=0.05723905723905724
DCV=0.06228795486299437
F-score=-0.002524448811968564
runtime_seconds=3.799281699997664
final_spread_estimator=monte_carlo

Stack execution report
------------------------------------------------------------------------
FIM Algorithm-Stack Permutation Comparison
------------------------------------------------------------------------
Protected attribute: region
Budget: 40
Final estimator: monte_carlo | mc_runs_eval=3

1. graphsage_fair_ris_hybrid [ok]
   spread=42.0000 | extra=2.0000 | MF=0.0572 | DCV=0.0623 | F-score=-0.0025 | runtime=3.799s
   modules: diffusion=ic | community=leiden | embedding=graphsage | clustering=none | ranking=graphsage | optimizer=hybrid_si_ea | debias=worst_group_boost
   notes=ML guidance mode=two_tier; full_pool=500; tier_policy=tuned; backend=gnn_ris; weights=(graphsage=1.000, ris=1.000, fair_ris=0.500, fairness=0.200, diversity=0.000); target=label_score; target_variance=0.006410; gnn_model_type=graphsage; node2vec_mode=off; feature_shape=(500, 53); edge_index_shape=(2, 1938); cache=miss; label_variance=0.006410; ris_num_rr_sets=128; ris_mode=weak_group_weighted; fair_ris_enabled=True; ris_reuse_rr_sets=True; history=graph_spa_500_0_budget40_leiden_hybrid_siea_ml_gnn_ris_two_tier_tuned_swap_local_search_history.csv; Delegated to existing gnn_ris hybrid path with shared final Monte Carlo evaluation.

Recommendations
------------------------------------------------------------------------
best_interpretable_baseline=n/a
best_practical_ml_default=graphsage_fair_ris_hybrid
best_exploratory_fairness_heavy=graphsage_fair_ris_hybrid
