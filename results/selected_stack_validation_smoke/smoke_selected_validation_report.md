Selected FIM Stack Report
========================================================================

Selection stage
------------------------------------------------------------------------
report=smoke_selected_validation
selected_stack_from_benchmark=line_fast_ml
benchmark_f_score=-0.306950
benchmark_mf=0.000000
benchmark_dcv=0.613899
benchmark_runtime=2.124
benchmark_quality_runtime_score=0.000000
selection_policy=fairness_runtime_tradeoff
selection_reason=within 0.003000 F-score of best valid stack (-0.306950), within DCV delta 0.010000, MF ratio >= 0.950000, and lowest runtime among close rows

Validation stage
------------------------------------------------------------------------
candidate_stack=line_fast_ml
fallback_stack=line_fast_ml
validation_seeds=[7]
candidate_mean_f_score=-0.324225
candidate_std_f_score=n/a
candidate_mean_mf=0.000000
candidate_mean_dcv=0.648451
candidate_mean_runtime=2.680
fallback_mean_f_score=-0.324225
fallback_std_f_score=n/a
fallback_mean_mf=0.000000
fallback_mean_dcv=0.648451
fallback_mean_runtime=2.680
f_score_gap=0.000000
dcv_delta=0.000000
runtime_speedup=1.000000
accepted=True
final_selected_stack=line_fast_ml
final_reason=accepted_after_validation: candidate met F-score closeness/ratio gate and DCV delta gate

Selected pipeline
------------------------------------------------------------------------
decision_status=accepted_after_validation
community_method=leiden
embedding_method=line
ranking_model=logistic_regression
optimizer_mode=greedy
spread_estimator_search=monte_carlo
spread_estimator_final=monte_carlo
use_repair=False
use_swap_local_search=False

Benchmark row used for selection
------------------------------------------------------------------------
{
  "dataset": "graph_spa_500_0",
  "protected_attribute": "region",
  "budget": 10,
  "stack_name": "line_fast_ml",
  "status": "ok",
  "f_score": -0.3069497277633138,
  "std_f_score": 0.0244315810951955,
  "mf": 0.0,
  "std_mf": 0.0,
  "dcv": 0.6138994555266277,
  "std_dcv": 0.0488631621903911,
  "total_spread": 11.0,
  "std_spread": 1.4142135623730951,
  "mean_extra": 1.0,
  "std_extra": 1.4142135623730951,
  "runtime_seconds": 2.123996150000039,
  "std_runtime": 0.0660840077387381,
  "win_count_by_f_score": 2,
  "win_count_by_runtime": 2,
  "number_of_successful_runs": 2,
  "number_of_failed_or_skipped_runs": 0,
  "variant_type": "ml_guided",
  "embedding_method": "line",
  "ranking_model": "logistic_regression",
  "community_method": "leiden",
  "clustering_method": "none",
  "optimizer_mode": "greedy",
  "debias_mode": "none",
  "spread_estimator_search": "monte_carlo",
  "spread_estimator_final": "monte_carlo",
  "key_enabled_modules": "diffusion=ic | community=leiden | embedding=line | clustering=none | clustering_input_mode=none | ranking=logistic_regression | optimizer=greedy | debias_mode=none | search_estimator=monte_carlo | final_estimator=monte_carlo",
  "normalized_f_score": 0.0,
  "normalized_runtime": 0.0,
  "quality_runtime_score": 0.0
}

Final result
------------------------------------------------------------------------
final_selected_stack=line_fast_ml
total_spread=11.0
extra_spread=1.0
MF=0.0
DCV=0.7092329854070484
F-score=-0.3546164927035242
runtime_seconds=2.398876299997937
final_spread_estimator=monte_carlo

Stack execution report
------------------------------------------------------------------------
FIM Algorithm-Stack Permutation Comparison
------------------------------------------------------------------------
Protected attribute: region
Budget: 10
Final estimator: monte_carlo | mc_runs_eval=3

1. line_fast_ml [ok]
   spread=11.0000 | extra=1.0000 | MF=0.0000 | DCV=0.7092 | F-score=-0.3546 | runtime=2.399s
   modules: diffusion=ic | community=leiden | embedding=line | clustering=none | ranking=logistic_regression | optimizer=greedy | debias=none
   notes=Fast shallow-embedding ML baseline.; communities=19; modularity=0.802180; embedding_cache=miss; target=label_score; backend=tabular; candidate_pool=250; validation_spearman=0.4650; validation_precision_at_budget=0.1000

Recommendations
------------------------------------------------------------------------
best_interpretable_baseline=n/a
best_practical_ml_default=line_fast_ml
best_exploratory_fairness_heavy=n/a
