# Symbolic XAI for Logic Games — Results

**Run type:** Quick (smoke)  
**Seeds:** [42]  
**Wall time:** 2.0s  

## Aggregated Summary (mean ± std across seeds)

| game         | model   | xai             |   n_seeds |   nn_accuracy_mean |   nn_accuracy_std |   nn_csr_mean |   nn_csr_std |   fidelity_to_nn_mean |   fidelity_to_nn_std |   comprehensiveness_mean |   comprehensiveness_std |   sufficiency_mean |   sufficiency_std |   max_sensitivity_mean |   max_sensitivity_std |   model_randomization_mean |   model_randomization_std |   data_randomization_mean |   data_randomization_std |   semantic_equivalence_z3_mean |   semantic_equivalence_z3_std |   agreement_gt_mean |   agreement_gt_std |   rule_count_mean |   rule_count_std |   rule_complexity_mean |   rule_complexity_std |
|:-------------|:--------|:----------------|----------:|-------------------:|------------------:|--------------:|-------------:|----------------------:|---------------------:|-------------------------:|------------------------:|-------------------:|------------------:|-----------------------:|----------------------:|---------------------------:|--------------------------:|--------------------------:|-------------------------:|-------------------------------:|------------------------------:|--------------------:|-------------------:|------------------:|-----------------:|-----------------------:|----------------------:|
| minesweeper4 | mlp     | rule_extraction |         1 |             0.8512 |                 0 |         0.375 |            0 |                  0.66 |                    0 |                        0 |                       0 |                  0 |                 0 |                      0 |                     0 |                     0.0142 |                         0 |                    0.0057 |                        0 |                           0.87 |                             0 |              0.9951 |                  0 |                 4 |                0 |                    635 |                     0 |

## Per-Seed Results

| game         | model   | xai             |   seed |   nn_accuracy |   nn_csr |   fidelity_to_nn |   comprehensiveness |   sufficiency |   max_sensitivity |   model_randomization |   data_randomization |   semantic_equivalence_z3 |   agreement_gt |   rule_count |   rule_complexity |
|:-------------|:--------|:----------------|-------:|--------------:|---------:|-----------------:|--------------------:|--------------:|------------------:|----------------------:|---------------------:|--------------------------:|---------------:|-------------:|------------------:|
| minesweeper4 | mlp     | rule_extraction |     42 |        0.8512 |    0.375 |             0.66 |                   0 |             0 |                 0 |                0.0142 |               0.0057 |                      0.87 |         0.9951 |            4 |               635 |

## Key Findings

- **minesweeper4 / mlp / rule_extraction**: NN acc=0.851, fidelity=0.660, GT agreement=0.995

## Done Criteria Check


## Plots

![Fidelity Heatmap](fidelity_heatmap.png)
