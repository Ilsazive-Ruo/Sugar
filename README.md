# Sugar Transfection Prediction v6

This directory contains the v6 workflow used to train, evaluate, interpret, and apply machine-learning models for predicting luciferase expression (RLU) from sugar physicochemical properties and RDKit molecular descriptors.

The manuscript analyses use the scripts with their default parameters unless otherwise noted. Run all commands from this `v6` directory.

```powershell
cd D:\xk\project\xulifeng\sugar\v6
conda activate sugar
```

Alternatively, prefix commands with `conda run -n sugar`.

## Files

| File | Role |
| --- | --- |
| `dataset.csv` | Training data. Contains sugar identity, provided physicochemical features, SMILES, and `Luciferase expression (RLU)`. |
| `ToBeScreened.csv` | Prediction input. Contains training-set rows plus additional candidates to screen; it does not contain RLU labels. |
| `ml_common.py` | Shared utilities for data loading, column normalization, RDKit descriptor generation, model pipelines, metrics, and feature alignment. Not run directly. |
| `model_feature_nested_cv.py` | Main repeated nested-CV model/feature-set comparison and hyperparameter selection. |
| `train_full_models_save_weights.py` | Trains the final selected model on the full training set and saves the model package. |
| `predict_new_samples.py` | Uses the final saved model to predict rows in `ToBeScreened.csv`. |
| `shap_feature_importance_cv.py` | Performs repeated-CV feature importance, XGBoost SHAP summaries, SHAP stability, and feature-correlation analyses. |
| `data_sufficiency_curve.py` | Evaluates data sufficiency / learning curves using random training-set subsets. |

## Data and Features

Training target:

```text
Luciferase expression (RLU)
```

Provided features:

```text
Molecular weight (Da)
Solubility (mg/mL)
Glass transition temperature (Tg, degC)
Viscosity (mPa.s)
Concentration (% w/v)
```

Feature sets compared by default:

```text
provided_only
provided_plus_rdkit_descriptors
rdkit_descriptors_only
```

Models compared by default:

```text
XGBoost
GBoost
RandomForest
DecisionTree
KNN
```

The modeling pipeline includes median imputation, zero-variance feature removal, optional `SelectKBest`, and the estimator. RDKit descriptors are generated from the `SMILES` column.

## Recommended Execution Order

### 1. Model and Feature-Set Selection

Run repeated nested cross-validation:

```powershell
conda run -n sugar python .\model_feature_nested_cv.py
```

Default settings:

```text
data: dataset.csv
outdir: results/model_feature_nested_cv
seed: 42
repeats: 10
outer-folds: 5
inner-folds: 3
search-iter: 80
n-jobs: -1
models: all
feature-sets: all
param-profile: anti_overfit
selection-rule: one_standard_error
one-se-multiplier: 1.0
feature-selection: auto
```

Important outputs:

| Output | Description |
| --- | --- |
| `results/model_feature_nested_cv/cv_summary.csv` | Main model/feature-set comparison summary. |
| `results/model_feature_nested_cv/cv_summary_selection_rules.csv` | Comparison between `best_rmse` and `one_standard_error`. |
| `results/model_feature_nested_cv/fold_metrics.csv` | Fold-level outer-CV metrics for the selected rule. |
| `results/model_feature_nested_cv/fold_metrics_selection_rules.csv` | Fold-level metrics for both selection rules. |
| `results/model_feature_nested_cv/best_hyperparameters.csv` | Consensus hyperparameters for each model/feature-set scenario. |
| `results/model_feature_nested_cv/selected_model.json` | Selected downstream model, feature set, and hyperparameters. |
| `results/model_feature_nested_cv/report.md` | Human-readable summary report. |

The default hyperparameter rule is `one_standard_error`: the inner-CV candidate within one standard error of the best inner RMSE and with lower complexity is selected. This is used to reduce overfitting from noisy hyperparameter choices.

### 2. Train Final Full-Data Model

Train the selected model on the complete training set:

```powershell
conda run -n sugar python .\train_full_models_save_weights.py
```

Default settings:

```text
data: dataset.csv
model-cv-dir: results/model_feature_nested_cv
outdir: results/full_models
model-dir: models
seed: 42
```

Important outputs:

| Output | Description |
| --- | --- |
| `models/<model>__<feature_set>.joblib` | Final saved model package. |
| `results/full_models/full_model_metadata.csv` | Metadata for the single final model. |
| `results/full_models/data_profile.json` | Final training profile and selected configuration. |

This script does not recompute performance metrics. It only consumes the selected configuration from `selected_model.json` and refits the final model on all training rows.

### 3. Predict Candidate Sugars

Predict the samples in `ToBeScreened.csv` using the final model:

```powershell
conda run -n sugar python .\predict_new_samples.py
```

Default settings:

```text
input: ToBeScreened.csv
model-dir: models
metadata: results/full_models/full_model_metadata.csv
outdir: results/predictions
```

Important outputs:

| Output | Description |
| --- | --- |
| `results/predictions/predictions.csv` | Prediction results in input-row order. |
| `results/predictions/predictions_ranked.csv` | Predictions ranked by predicted RLU. |
| `results/predictions/smiles_cleaning_report.csv` | SMILES strings changed by whitespace/fix rules. |
| `results/predictions/prediction_profile.json` | Prediction run metadata. |
| `results/predictions/report.md` | Top predictions and output summary. |

The script reads `full_model_metadata.csv` and uses only the final selected model. It does not scan all `.joblib` files in `models`.

### 4. Feature Importance, SHAP, and Correlation Analysis

Run repeated-CV feature analysis:

```powershell
conda run -n sugar python .\shap_feature_importance_cv.py
```

Default settings:

```text
data: dataset.csv
model-cv-dir: results/model_feature_nested_cv
outdir: results/feature_analysis
seed: 42
repeats: 10
folds: 5
permutation-repeats: 5
correlation-threshold: 0.8
```

Important outputs:

| Output | Description |
| --- | --- |
| `results/feature_analysis/feature_importance_summary.csv` | Mean feature importance across CV folds. |
| `results/feature_analysis/shap_summary.csv` | Mean XGBoost SHAP summaries. |
| `results/feature_analysis/shap_stability_summary.csv` | Fold-level SHAP rank and magnitude stability. |
| `results/feature_analysis/feature_correlation_summary.csv` | Feature-correlation summary after model pipeline transforms. |
| `results/feature_analysis/feature_correlation_high_pairs.csv` | High-correlation feature pairs using the default absolute Pearson threshold of 0.8. |
| `results/feature_analysis/feature_correlation_matrices/` | Per-feature-set Pearson correlation matrices. |
| `results/feature_analysis/xgboost_all_feature_values_wide.csv` | Per-sample values for all XGBoost full-feature inputs. |
| `results/feature_analysis/xgboost_all_feature_values_long.csv` | Long-format sample-feature table. |
| `results/feature_analysis/xgboost_all_feature_target_correlations.csv` | Feature-wise Pearson/Spearman correlation with RLU. |
| `results/feature_analysis/xgboost_all_feature_pearson_matrix.csv` | Full 2D Pearson matrix including RLU and all XGBoost input features. |
| `results/feature_analysis/xgboost_all_feature_spearman_matrix.csv` | Full 2D Spearman matrix including RLU and all XGBoost input features. |
| `results/feature_analysis/report.md` | Human-readable feature-analysis report. |

The XGBoost full-feature correlation outputs use `provided_plus_rdkit_descriptors` with `select__k=all`, and include the raw provided/RDKit input features before pipeline transforms. The column `used_by_xgboost_after_pipeline` marks whether a feature remains after imputation, variance filtering, and feature selection.

### 5. Data Sufficiency / Learning Curve

Run the data sufficiency curve:

```powershell
conda run -n sugar python .\data_sufficiency_curve.py
```

Default settings:

```text
data: dataset.csv
model-cv-dir: results/model_feature_nested_cv
outdir: results/data_sufficiency_curve
seed: 42
start-size: 30
max-size: None
step-size: 5
subset-repeats: 10
folds: 5
evaluation-mode: nested_1se
inner-folds: 3
search-iter: 12
n-jobs: 1
param-profile: anti_overfit
feature-selection: auto
one-se-multiplier: 1.0
```

Important outputs:

| Output | Description |
| --- | --- |
| `results/data_sufficiency_curve/learning_curve_summary.csv` | Mean metrics by subset size. |
| `results/data_sufficiency_curve/subset_repeat_summary.csv` | Repeat-level summary by subset size. |
| `results/data_sufficiency_curve/fold_metrics.csv` | Fold-level metrics. |
| `results/data_sufficiency_curve/cv_predictions.csv` | Out-of-fold predictions. |
| `results/data_sufficiency_curve/subset_membership.csv` | Sample membership for each random subset. |
| `results/data_sufficiency_curve/report.md` | Human-readable data-sufficiency report. |

`nested_1se` repeats inner tuning within each learning-curve fold. It is slower but methodologically closer to the main model-selection procedure.

## Typical Full Workflow

```powershell
conda run -n sugar python .\model_feature_nested_cv.py
conda run -n sugar python .\train_full_models_save_weights.py
conda run -n sugar python .\predict_new_samples.py
conda run -n sugar python .\shap_feature_importance_cv.py
conda run -n sugar python .\data_sufficiency_curve.py
```

The first three commands are required for prediction. The last two commands are analysis/reporting steps.

## Notes on Interpretation

- `one_standard_error` may have slightly worse inner-CV RMSE than `best_rmse` but better outer-CV RMSE. This is expected when the simpler candidate generalizes better.
- Highly correlated RDKit descriptors should be interpreted as descriptor clusters. A high SHAP score for one descriptor, such as `EState_VSA10`, may represent a broader correlated feature group rather than a unique causal feature.

## Environment

```text
Python 3.13.14
pandas 3.0.3
numpy 2.5.0
scikit-learn 1.9.0
scipy 1.18.0
joblib 1.5.3
xgboost 3.2.0
rdkit 2025.09.6
```

Install the Python dependencies with:

```powershell
pip install -r requirements.txt
```

RDKit is often easier to install with conda:

```powershell
conda install -c conda-forge rdkit
```
