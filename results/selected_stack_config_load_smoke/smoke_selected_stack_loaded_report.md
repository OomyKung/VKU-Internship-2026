# Selected FIM Stack Report

- Report: smoke_selected_stack_loaded
- selected_stack: line_fast_ml
- protected_attribute: region
- budget: 10
- selection_policy: fairness_runtime_tradeoff
- selection_reason: within 0.005000 F-score of best valid stack (-0.306950) and lowest runtime among close rows

## Selected Pipeline

- community_method: leiden
- embedding_method: line
- ranking_model: logistic_regression
- optimizer_mode: greedy
- spread_estimator_search: monte_carlo
- spread_estimator_final: monte_carlo
- use_repair: False
- use_swap_local_search: False

## Benchmark Row Used For Selection

```json
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
  "key_enabled_modules": "diffusion=ic | community=leiden | embedding=line | clustering=none | clustering_input_mode=none | ranking=logistic_regression | optimizer=greedy | debias_mode=none | search_estimator=monte_carlo | final_estimator=monte_carlo"
}
```

## Final Result

- final total_spread: 10.0
- extra_spread: 0.0
- MF: 0.0
- DCV: 0.7327743346700692
- F-score: -0.3663871673350346
- runtime_seconds: 2.389435199998843
- final_spread_estimator: monte_carlo

## Stack Execution Report

```text
FIM Algorithm-Stack Permutation Comparison
------------------------------------------------------------------------
Protected attribute: region
Budget: 10
Final estimator: monte_carlo | mc_runs_eval=3

1. line_fast_ml [ok]
   spread=10.0000 | extra=0.0000 | MF=0.0000 | DCV=0.7328 | F-score=-0.3664 | runtime=2.389s
   modules: diffusion=ic | community=leiden | embedding=line | clustering=none | ranking=logistic_regression | optimizer=greedy | debias=none
   notes=Fast shallow-embedding ML baseline.; communities=18; modularity=0.802426; embedding_cache=miss; target=label_score; backend=tabular; candidate_pool=250; validation_spearman=0.2457; validation_precision_at_budget=0.1000

Recommendations
------------------------------------------------------------------------
best_interpretable_baseline=n/a
best_practical_ml_default=line_fast_ml
best_exploratory_fairness_heavy=n/a
```
