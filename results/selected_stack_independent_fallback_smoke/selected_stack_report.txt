Selected FIM Stack Report
========================================================================

Selection stage
------------------------------------------------------------------------
report=independent_fallback_smoke
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
fallback_reason=highest valid benchmark F-score independent fairness-first fallback

Validation stage
------------------------------------------------------------------------
candidate_stack=graphsage_fair_ris_hybrid
fallback_stack=community_aware_fair_greedy
validation_seeds=[7]
candidate_mean_f_score=0.019407
candidate_std_f_score=n/a
candidate_mean_mf=0.067340
candidate_mean_dcv=0.028525
candidate_mean_spread=42.000000
candidate_mean_runtime=5.132
fallback_mean_f_score=0.006944
fallback_std_f_score=n/a
fallback_mean_mf=0.066351
fallback_mean_dcv=0.052463
fallback_mean_spread=41.000000
fallback_mean_runtime=17.356
f_score_gap=0.012463
dcv_delta=-0.023938
mf_ratio=1.014911
runtime_speedup=3.382058
accepted=True
final_selected_stack=graphsage_fair_ris_hybrid
final_reason=accepted_after_validation: candidate met F-score, DCV, and MF validation gates

Selected pipeline
------------------------------------------------------------------------
decision_status=accepted_after_validation
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
runtime_seconds=5.1598494999998366
final_spread_estimator=monte_carlo
final_validation_warning=True
final_validation_warning_reasons=['final F-score dropped below benchmark ratio threshold', 'final DCV increased beyond benchmark delta threshold']

Stack execution report
------------------------------------------------------------------------
FIM Algorithm-Stack Permutation Comparison
------------------------------------------------------------------------
Protected attribute: region
Budget: 40
Final estimator: monte_carlo | mc_runs_eval=3

1. graphsage_fair_ris_hybrid [ok]
   spread=42.0000 | extra=2.0000 | MF=0.0572 | DCV=0.0623 | F-score=-0.0025 | runtime=5.160s
   modules: diffusion=ic | community=leiden | embedding=graphsage | clustering=none | ranking=graphsage | optimizer=hybrid_si_ea | debias=worst_group_boost
   notes=ML guidance mode=two_tier; full_pool=500; tier_policy=tuned; backend=gnn_ris; weights=(graphsage=1.000, ris=1.000, fair_ris=0.500, fairness=0.200, diversity=0.000); target=label_score; target_variance=0.006410; gnn_model_type=graphsage; node2vec_mode=off; feature_shape=(500, 53); edge_index_shape=(2, 1938); cache=miss; label_variance=0.006410; ris_num_rr_sets=128; ris_mode=weak_group_weighted; fair_ris_enabled=True; ris_reuse_rr_sets=True; history=graph_spa_500_0_budget40_leiden_hybrid_siea_ml_gnn_ris_two_tier_tuned_swap_local_search_history.csv; Delegated to existing gnn_ris hybrid path with shared final Monte Carlo evaluation.

Recommendations
------------------------------------------------------------------------
best_interpretable_baseline=n/a
best_practical_ml_default=graphsage_fair_ris_hybrid
best_exploratory_fairness_heavy=graphsage_fair_ris_hybrid
