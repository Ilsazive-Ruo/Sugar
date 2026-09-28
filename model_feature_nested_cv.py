from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, ParameterGrid, RandomizedSearchCV

from ml_common import (
    FEATURE_SET_NAMES,
    MODEL_NAMES,
    TARGET_COL,
    build_feature_sets,
    dataframe_to_markdown,
    json_dumps,
    make_estimator,
    read_training_data,
    regression_metrics,
    smiles_cleaning_report,
)


def parse_csv_choices(raw: str, allowed: list[str], option_name: str) -> list[str]:
    if raw.strip().lower() == "all":
        return allowed
    values = [part.strip() for part in raw.split(",") if part.strip()]
    invalid = [value for value in values if value not in allowed]
    if invalid:
        raise ValueError(
            f"Invalid {option_name}: {invalid}. Allowed: {', '.join(allowed)} or all."
        )
    return values


def prefix_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def baseline_param_grid(model_name: str) -> dict[str, list[Any]]:
    if model_name == "KNN":
        return {
            "model__n_neighbors": [3, 5, 7, 9, 11, 15],
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2],
        }
    if model_name == "DecisionTree":
        return {
            "model__max_depth": [None, 3, 5, 8, 12],
            "model__min_samples_split": [2, 4, 8, 12],
            "model__min_samples_leaf": [1, 2, 4, 8],
            "model__max_features": [None, "sqrt", 0.5, 1.0],
        }
    if model_name == "GBoost":
        return {
            "model__n_estimators": [100, 200, 300, 500],
            "model__learning_rate": [0.02, 0.05, 0.08, 0.1],
            "model__max_depth": [2, 3, 5, 8],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__min_samples_leaf": [1, 2, 4, 8],
        }
    if model_name == "RandomForest":
        return {
            "model__n_estimators": [100, 200, 300, 500],
            "model__max_depth": [None, 5, 8, 12, 16],
            "model__min_samples_split": [2, 4, 8, 12],
            "model__min_samples_leaf": [1, 2, 4, 8],
            "model__max_features": [1.0, "sqrt", 0.5],
        }
    if model_name == "XGBoost":
        return {
            "model__n_estimators": [100, 200, 300, 500],
            "model__learning_rate": [0.02, 0.05, 0.08, 0.1],
            "model__max_depth": [2, 3, 5, 8],
            "model__min_child_weight": [1, 2, 4],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__colsample_bytree": [0.7, 0.85, 1.0],
            "model__reg_lambda": [0.5, 1.0, 2.0],
        }
    raise ValueError(f"Unsupported model: {model_name}")


def effective_baseline_param_grid(
    model_name: str,
    n_train: int,
    inner_folds: int,
) -> dict[str, list[Any]]:
    grid = {key: list(value) for key, value in baseline_param_grid(model_name).items()}
    if model_name == "KNN" and "model__n_neighbors" in grid:
        max_inner_train = n_train - int(np.ceil(n_train / inner_folds))
        grid["model__n_neighbors"] = [
            value for value in grid["model__n_neighbors"] if int(value) <= max_inner_train
        ]
        if not grid["model__n_neighbors"]:
            raise ValueError(f"KNN cannot run with n_train={n_train} and inner_folds={inner_folds}.")
    return grid


def anti_overfit_param_grid(model_name: str) -> dict[str, list[Any]]:
    """Regularized spaces with shared complexity axes where model APIs allow it."""
    tree_depth = [2, 3, 4, 5]
    min_samples_split = [2, 8, 16]
    min_leaf_like = [1, 4, 8]
    feature_fraction = [0.6, 0.8, 1.0]
    sample_fraction = [0.7, 0.85, 1.0]
    n_estimators = [100, 200, 300]
    learning_rate = [0.03, 0.06, 0.1]
    pruning_alpha = [0.0, 1e-4, 1e-3]

    if model_name == "KNN":
        return {
            "model__n_neighbors": [5, 9, 15, 21, 31],
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2],
        }
    if model_name == "DecisionTree":
        return {
            "model__max_depth": tree_depth,
            "model__min_samples_split": min_samples_split,
            "model__min_samples_leaf": min_leaf_like,
            "model__max_features": feature_fraction,
            "model__ccp_alpha": pruning_alpha,
        }
    if model_name == "GBoost":
        return {
            "model__n_estimators": n_estimators,
            "model__learning_rate": learning_rate,
            "model__max_depth": tree_depth,
            "model__subsample": sample_fraction,
            "model__min_samples_split": min_samples_split,
            "model__min_samples_leaf": min_leaf_like,
            "model__max_features": feature_fraction,
            "model__ccp_alpha": pruning_alpha,
        }
    if model_name == "RandomForest":
        return {
            "model__n_estimators": n_estimators,
            "model__max_depth": tree_depth,
            "model__min_samples_split": min_samples_split,
            "model__min_samples_leaf": min_leaf_like,
            "model__max_features": feature_fraction,
            "model__bootstrap": [True],
            "model__max_samples": sample_fraction,
            "model__ccp_alpha": pruning_alpha,
        }
    if model_name == "XGBoost":
        return {
            "model__n_estimators": n_estimators,
            "model__learning_rate": learning_rate,
            "model__max_depth": tree_depth,
            "model__min_child_weight": min_leaf_like,
            "model__subsample": sample_fraction,
            "model__colsample_bytree": feature_fraction,
            "model__reg_lambda": [1.0, 3.0, 10.0],
            "model__reg_alpha": [0.0, 0.1, 1.0],
            "model__gamma": [0.0, 0.1, 1.0],
        }
    raise ValueError(f"Unsupported model: {model_name}")


def add_feature_selection_grid(
    grid: dict[str, list[Any]],
    feature_set_name: str,
    n_features: int,
    n_train: int,
    inner_folds: int,
    feature_selection: str,
) -> dict[str, list[Any]]:
    grid = {key: list(value) for key, value in grid.items()}
    if feature_selection == "off":
        return grid
    if feature_set_name == "provided_only" or n_features <= 20:
        return grid
    max_inner_train = n_train - int(np.ceil(n_train / inner_folds))
    max_k = min(n_features, max_inner_train)
    k_values = [value for value in [20, 40, 80, 120] if value < max_k]
    k_values.append("all")
    grid["select__k"] = k_values
    return grid


def effective_search_grid(
    model_name: str,
    feature_set_name: str,
    n_features: int,
    n_train: int,
    inner_folds: int,
    param_profile: str,
    feature_selection: str,
) -> dict[str, list[Any]]:
    if param_profile == "baseline":
        grid = effective_baseline_param_grid(model_name, n_train, inner_folds)
    elif param_profile == "anti_overfit":
        grid = anti_overfit_param_grid(model_name)
        if model_name == "KNN" and "model__n_neighbors" in grid:
            max_inner_train = n_train - int(np.ceil(n_train / inner_folds))
            grid["model__n_neighbors"] = [
                value for value in grid["model__n_neighbors"] if int(value) <= max_inner_train
            ]
            if not grid["model__n_neighbors"]:
                raise ValueError(
                    f"KNN cannot run with n_train={n_train} and inner_folds={inner_folds}."
                )
    else:
        raise ValueError(f"Unsupported --param-profile: {param_profile}")
    if param_profile == "anti_overfit":
        grid = add_feature_selection_grid(
            grid,
            feature_set_name,
            n_features,
            n_train,
            inner_folds,
            feature_selection,
        )
    return grid


def as_float_param(params: dict[str, Any], key: str, default: float) -> float:
    value = params.get(key, default)
    if value is None:
        return default
    if isinstance(value, str):
        if value == "sqrt":
            return 0.5
        if value == "log2":
            return 0.5
        return default
    return float(value)


def selector_complexity(params: dict[str, Any], n_features: int) -> float:
    k_value = params.get("select__k", "all")
    if k_value == "all":
        return 10.0
    return 10.0 * float(k_value) / max(float(n_features), 1.0)


def model_complexity_score(model_name: str, params: dict[str, Any], n_features: int) -> float:
    score = selector_complexity(params, n_features)
    if model_name == "KNN":
        n_neighbors = as_float_param(params, "model__n_neighbors", 5.0)
        score += 20.0 / max(n_neighbors, 1.0)
        score += 1.0 if params.get("model__weights") == "distance" else 0.0
        return float(score)
    if model_name == "DecisionTree":
        max_depth = params.get("model__max_depth")
        depth = 20.0 if max_depth is None else float(max_depth)
        min_leaf = as_float_param(params, "model__min_samples_leaf", 1.0)
        min_split = as_float_param(params, "model__min_samples_split", 2.0)
        max_features = as_float_param(params, "model__max_features", 1.0)
        ccp_alpha = as_float_param(params, "model__ccp_alpha", 0.0)
        return float(score + depth * 2.0 + max_features * 2.0 + 20.0 / min_leaf + 10.0 / min_split - ccp_alpha * 100.0)
    if model_name == "GBoost":
        depth = as_float_param(params, "model__max_depth", 3.0)
        n_estimators = as_float_param(params, "model__n_estimators", 100.0)
        min_leaf = as_float_param(params, "model__min_samples_leaf", 1.0)
        subsample = as_float_param(params, "model__subsample", 1.0)
        max_features = as_float_param(params, "model__max_features", 1.0)
        ccp_alpha = as_float_param(params, "model__ccp_alpha", 0.0)
        return float(score + depth * 3.0 + np.log1p(n_estimators) + subsample + max_features + 12.0 / min_leaf - ccp_alpha * 100.0)
    if model_name == "RandomForest":
        max_depth = params.get("model__max_depth")
        depth = 20.0 if max_depth is None else float(max_depth)
        n_estimators = as_float_param(params, "model__n_estimators", 200.0)
        min_leaf = as_float_param(params, "model__min_samples_leaf", 1.0)
        max_features = as_float_param(params, "model__max_features", 1.0)
        max_samples = as_float_param(params, "model__max_samples", 1.0)
        ccp_alpha = as_float_param(params, "model__ccp_alpha", 0.0)
        return float(score + depth * 2.0 + np.log1p(n_estimators) + max_features * 2.0 + max_samples + 15.0 / min_leaf - ccp_alpha * 1000.0)
    if model_name == "XGBoost":
        depth = as_float_param(params, "model__max_depth", 3.0)
        n_estimators = as_float_param(params, "model__n_estimators", 100.0)
        min_child = as_float_param(params, "model__min_child_weight", 1.0)
        subsample = as_float_param(params, "model__subsample", 1.0)
        colsample = as_float_param(params, "model__colsample_bytree", 1.0)
        reg_lambda = as_float_param(params, "model__reg_lambda", 1.0)
        reg_alpha = as_float_param(params, "model__reg_alpha", 0.0)
        gamma = as_float_param(params, "model__gamma", 0.0)
        return float(
            score
            + depth * 3.0
            + np.log1p(n_estimators)
            + subsample
            + colsample
            + 10.0 / min_child
            - np.log1p(reg_lambda)
            - np.log1p(reg_alpha)
            - np.log1p(gamma)
        )
    raise ValueError(f"Unsupported model: {model_name}")


def select_search_result(
    search: RandomizedSearchCV,
    model_name: str,
    n_features: int,
    inner_folds: int,
    selection_rule: str,
    one_se_multiplier: float,
) -> dict[str, Any]:
    cv_results = pd.DataFrame(search.cv_results_)
    cv_results["mean_inner_rmse"] = -cv_results["mean_test_score"].astype(float)
    cv_results["std_inner_rmse"] = cv_results["std_test_score"].astype(float)
    cv_results["inner_rmse_se"] = cv_results["std_inner_rmse"] / np.sqrt(inner_folds)
    if "mean_train_score" in cv_results.columns:
        cv_results["mean_inner_train_rmse"] = -cv_results["mean_train_score"].astype(float)
        cv_results["inner_train_test_rmse_gap"] = (
            cv_results["mean_inner_rmse"] - cv_results["mean_inner_train_rmse"]
        )
    else:
        cv_results["mean_inner_train_rmse"] = np.nan
        cv_results["inner_train_test_rmse_gap"] = np.nan
    cv_results["complexity_score"] = [
        model_complexity_score(model_name, params, n_features)
        for params in cv_results["params"]
    ]

    best_idx = int(cv_results["mean_inner_rmse"].idxmin())
    best = cv_results.loc[best_idx]
    if selection_rule == "best_rmse":
        selected = best
        threshold = float(best["mean_inner_rmse"])
        n_candidates = 1
    elif selection_rule == "one_standard_error":
        threshold = float(best["mean_inner_rmse"] + one_se_multiplier * best["inner_rmse_se"])
        candidates = cv_results.loc[cv_results["mean_inner_rmse"] <= threshold].copy()
        n_candidates = int(len(candidates))
        selected = candidates.sort_values(
            ["complexity_score", "inner_train_test_rmse_gap", "mean_inner_rmse"],
            ascending=[True, True, True],
        ).iloc[0]
    else:
        raise ValueError(f"Unsupported --selection-rule: {selection_rule}")

    return {
        "selected_params": dict(selected["params"]),
        "selected_inner_rmse": float(selected["mean_inner_rmse"]),
        "selected_inner_rmse_se": float(selected["inner_rmse_se"]),
        "selected_inner_train_rmse": float(selected["mean_inner_train_rmse"]),
        "selected_inner_train_test_rmse_gap": float(selected["inner_train_test_rmse_gap"]),
        "selected_complexity_score": float(selected["complexity_score"]),
        "inner_cv_min_rmse": float(best["mean_inner_rmse"]),
        "inner_cv_min_rmse_se": float(best["inner_rmse_se"]),
        "inner_cv_min_params": dict(best["params"]),
        "one_se_threshold": threshold,
        "one_se_candidate_count": n_candidates,
    }


def run_nested_cv(
    model_name: str,
    feature_set_name: str,
    x: pd.DataFrame,
    y: pd.Series,
    repeat_seeds: list[int],
    outer_folds: int,
    inner_folds: int,
    search_iter: int,
    n_jobs: int,
    param_profile: str,
    selection_rule: str,
    feature_selection: str,
    one_se_multiplier: float,
    collect_selection_comparison: bool = False,
) -> tuple[pd.DataFrame, ...]:
    fold_rows = []
    pred_frames = []
    train_pred_frames = []
    comparison_fold_rows = []
    comparison_pred_frames = []
    comparison_train_pred_frames = []

    for repeat_idx, repeat_seed in enumerate(repeat_seeds, start=1):
        outer_cv = KFold(n_splits=outer_folds, shuffle=True, random_state=repeat_seed)
        for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(x), start=1):
            x_train = x.iloc[train_idx]
            x_val = x.iloc[val_idx]
            y_train = y.iloc[train_idx].to_numpy(dtype=float)
            y_val = y.iloc[val_idx].to_numpy(dtype=float)

            grid = effective_search_grid(
                model_name=model_name,
                feature_set_name=feature_set_name,
                n_features=int(x.shape[1]),
                n_train=len(train_idx),
                inner_folds=inner_folds,
                param_profile=param_profile,
                feature_selection=feature_selection,
            )
            total_grid_size = len(ParameterGrid(grid))
            n_iter = min(search_iter, total_grid_size)
            search_seed = repeat_seed + fold_idx
            search = RandomizedSearchCV(
                estimator=make_estimator(model_name, repeat_seed),
                param_distributions=grid,
                n_iter=n_iter,
                scoring="neg_root_mean_squared_error",
                cv=KFold(n_splits=inner_folds, shuffle=True, random_state=search_seed),
                random_state=search_seed,
                n_jobs=n_jobs,
                refit=False,
                error_score="raise",
                return_train_score=True,
            )
            search.fit(x_train, y_train)
            scenario_run = f"{model_name}__{feature_set_name}"

            selection_cache: dict[str, dict[str, Any]] = {}

            def get_selection(rule_name: str) -> dict[str, Any]:
                if rule_name not in selection_cache:
                    selection_cache[rule_name] = select_search_result(
                        search=search,
                        model_name=model_name,
                        n_features=int(x.shape[1]),
                        inner_folds=inner_folds,
                        selection_rule=rule_name,
                        one_se_multiplier=one_se_multiplier,
                    )
                return selection_cache[rule_name]

            def evaluate_selection(
                selected: dict[str, Any],
                rule_name: str,
            ) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
                estimator = make_estimator(model_name, search_seed)
                estimator.set_params(**selected["selected_params"])
                estimator.fit(x_train, y_train)
                y_pred = np.asarray(estimator.predict(x_val), dtype=float)
                y_train_pred = np.asarray(estimator.predict(x_train), dtype=float)

                metrics = regression_metrics(y_val, y_pred)
                train_metrics = prefix_metrics(regression_metrics(y_train, y_train_pred), "train")
                metrics.update(train_metrics)
                metrics["train_val_rmse_gap"] = metrics["rmse"] - metrics["train_rmse"]
                metrics["train_val_rmse_ratio"] = (
                    metrics["rmse"] / metrics["train_rmse"] if metrics["train_rmse"] else np.nan
                )
                metrics["rmse_ratio"] = metrics["train_val_rmse_ratio"]
                metrics["train_val_r2_gap"] = metrics["train_r2"] - metrics["r2"]
                metrics.update(
                    {
                        "scenario_run": scenario_run,
                        "model": model_name,
                        "feature_set": feature_set_name,
                        "repeat": repeat_idx,
                        "repeat_seed": repeat_seed,
                        "fold": fold_idx,
                        "n_features_input": int(x.shape[1]),
                        "n_total_samples": int(len(x)),
                        "n_train": int(len(train_idx)),
                        "n_val": int(len(val_idx)),
                        "outer_folds": outer_folds,
                        "inner_folds": inner_folds,
                        "search_n_iter": int(n_iter),
                        "search_total_grid_size": int(total_grid_size),
                        "param_profile": param_profile,
                        "feature_selection": feature_selection,
                        "selection_rule": rule_name,
                        "one_se_multiplier": float(one_se_multiplier),
                        "inner_best_rmse": float(selected["inner_cv_min_rmse"]),
                        "inner_selected_rmse": float(selected["selected_inner_rmse"]),
                        "inner_selected_rmse_gap_vs_min": float(
                            selected["selected_inner_rmse"] - selected["inner_cv_min_rmse"]
                        ),
                        "inner_selected_rmse_se": float(selected["selected_inner_rmse_se"]),
                        "inner_selected_train_rmse": float(selected["selected_inner_train_rmse"]),
                        "inner_selected_train_test_rmse_gap": float(
                            selected["selected_inner_train_test_rmse_gap"]
                        ),
                        "inner_selected_complexity_score": float(
                            selected["selected_complexity_score"]
                        ),
                        "inner_cv_min_rmse": float(selected["inner_cv_min_rmse"]),
                        "inner_cv_min_rmse_se": float(selected["inner_cv_min_rmse_se"]),
                        "inner_cv_min_params": json_dumps(selected["inner_cv_min_params"]),
                        "one_se_threshold": float(selected["one_se_threshold"]),
                        "one_se_candidate_count": int(selected["one_se_candidate_count"]),
                        "best_params": json_dumps(selected["selected_params"]),
                    }
                )

                pred_frame = pd.DataFrame(
                    {
                        "scenario_run": scenario_run,
                        "model": model_name,
                        "feature_set": feature_set_name,
                        "selection_rule": rule_name,
                        "repeat": repeat_idx,
                        "repeat_seed": repeat_seed,
                        "fold": fold_idx,
                        "row_index": val_idx,
                        "y_true": y_val,
                        "y_pred": y_pred,
                        "abs_error": np.abs(y_pred - y_val),
                        "squared_error": np.square(y_pred - y_val),
                    }
                )
                train_pred_frame = pd.DataFrame(
                    {
                        "scenario_run": scenario_run,
                        "model": model_name,
                        "feature_set": feature_set_name,
                        "selection_rule": rule_name,
                        "repeat": repeat_idx,
                        "repeat_seed": repeat_seed,
                        "fold": fold_idx,
                        "row_index": train_idx,
                        "y_true": y_train,
                        "y_pred": y_train_pred,
                        "abs_error": np.abs(y_train_pred - y_train),
                        "squared_error": np.square(y_train_pred - y_train),
                    }
                )
                return metrics, pred_frame, train_pred_frame

            primary_selection = get_selection(selection_rule)
            metrics, pred_frame, train_pred_frame = evaluate_selection(
                primary_selection,
                selection_rule,
            )
            fold_rows.append(metrics)
            pred_frames.append(pred_frame)
            train_pred_frames.append(train_pred_frame)

            if collect_selection_comparison:
                for comparison_rule in ["best_rmse", "one_standard_error"]:
                    if comparison_rule == selection_rule:
                        comparison_metrics = metrics
                        comparison_pred_frame = pred_frame
                        comparison_train_pred_frame = train_pred_frame
                    else:
                        comparison_selection = get_selection(comparison_rule)
                        (
                            comparison_metrics,
                            comparison_pred_frame,
                            comparison_train_pred_frame,
                        ) = evaluate_selection(comparison_selection, comparison_rule)
                    comparison_fold_rows.append(comparison_metrics)
                    comparison_pred_frames.append(comparison_pred_frame)
                    comparison_train_pred_frames.append(comparison_train_pred_frame)

    primary_outputs = (
        pd.DataFrame(fold_rows),
        pd.concat(pred_frames, ignore_index=True),
        pd.concat(train_pred_frames, ignore_index=True),
    )
    if not collect_selection_comparison:
        return primary_outputs
    return primary_outputs + (
        pd.DataFrame(comparison_fold_rows),
        pd.concat(comparison_pred_frames, ignore_index=True),
        pd.concat(comparison_train_pred_frames, ignore_index=True),
    )


def summarize_cv(
    fold_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    train_predictions: pd.DataFrame,
    extra_group_cols: list[str] | None = None,
) -> pd.DataFrame:
    metric_cols = [
        "rmse",
        "mae",
        "median_ae",
        "r2",
        "explained_variance",
        "mape",
        "max_error",
        "mean_error",
        "std_error",
        "mean_absolute_log_error",
        "rmse_log1p",
        "pearson",
        "spearman",
        "inner_best_rmse",
        "inner_selected_rmse",
        "inner_selected_rmse_gap_vs_min",
        "one_se_threshold",
        "one_se_candidate_count",
        "train_val_rmse_gap",
        "train_val_rmse_ratio",
        "rmse_ratio",
        "train_val_r2_gap",
    ]
    train_metric_cols = [
        f"train_{metric}"
        for metric in metric_cols
        if metric != "inner_best_rmse" and f"train_{metric}" in fold_metrics.columns
    ]
    group_cols = ["scenario_run", "model", "feature_set"] + (extra_group_cols or [])

    def key_dict(keys: object) -> dict[str, object]:
        if not isinstance(keys, tuple):
            keys = (keys,)
        return dict(zip(group_cols, keys))

    summary = fold_metrics.groupby(group_cols)[
        metric_cols + train_metric_cols + ["n_train", "n_val"]
    ].agg(
        ["mean", "std"]
    )
    summary.columns = [
        "_".join(str(part) for part in col if part) if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    summary = summary.reset_index()

    pooled_rows = []
    for keys, group in predictions.groupby(group_cols):
        pooled = regression_metrics(group["y_true"].to_numpy(), group["y_pred"].to_numpy())
        row = key_dict(keys)
        row.update(
            {
                "pooled_oof_rmse": pooled["rmse"],
                "pooled_oof_mae": pooled["mae"],
                "pooled_oof_r2": pooled["r2"],
                "pooled_oof_pearson": pooled["pearson"],
                "pooled_oof_spearman": pooled["spearman"],
                "n_predictions": int(len(group)),
                "n_unique_predicted_rows": int(group["row_index"].nunique()),
            }
        )
        pooled_rows.append(row)
    summary = summary.merge(pd.DataFrame(pooled_rows), on=group_cols, how="left")

    train_pooled_rows = []
    for keys, group in train_predictions.groupby(group_cols):
        pooled = regression_metrics(group["y_true"].to_numpy(), group["y_pred"].to_numpy())
        row = key_dict(keys)
        row.update(
            {
                "pooled_train_rmse": pooled["rmse"],
                "pooled_train_mae": pooled["mae"],
                "pooled_train_r2": pooled["r2"],
                "pooled_train_pearson": pooled["pearson"],
                "pooled_train_spearman": pooled["spearman"],
                "n_train_predictions": int(len(group)),
                "n_unique_train_rows": int(group["row_index"].nunique()),
            }
        )
        train_pooled_rows.append(row)
    summary = summary.merge(pd.DataFrame(train_pooled_rows), on=group_cols, how="left")
    summary = summary.rename(
        columns={
            "rmse_mean": "mean_fold_rmse",
            "rmse_std": "std_fold_rmse",
            "mae_mean": "mean_fold_mae",
            "mae_std": "std_fold_mae",
            "r2_mean": "mean_fold_r2",
            "r2_std": "std_fold_r2",
            "mape_mean": "mean_fold_mape",
            "mape_std": "std_fold_mape",
            "inner_best_rmse_mean": "mean_inner_best_rmse",
            "inner_best_rmse_std": "std_inner_best_rmse",
            "inner_selected_rmse_mean": "mean_inner_selected_rmse",
            "inner_selected_rmse_std": "std_inner_selected_rmse",
            "inner_selected_rmse_gap_vs_min_mean": "mean_inner_selected_rmse_gap_vs_min",
            "inner_selected_rmse_gap_vs_min_std": "std_inner_selected_rmse_gap_vs_min",
            "one_se_threshold_mean": "mean_one_se_threshold",
            "one_se_threshold_std": "std_one_se_threshold",
            "one_se_candidate_count_mean": "mean_one_se_candidate_count",
            "one_se_candidate_count_std": "std_one_se_candidate_count",
            "train_val_rmse_gap_mean": "mean_train_val_rmse_gap",
            "train_val_rmse_gap_std": "std_train_val_rmse_gap",
            "train_val_rmse_ratio_mean": "mean_train_val_rmse_ratio",
            "train_val_rmse_ratio_std": "std_train_val_rmse_ratio",
            "rmse_ratio_mean": "mean_rmse_ratio",
            "rmse_ratio_std": "std_rmse_ratio",
            "train_val_r2_gap_mean": "mean_train_val_r2_gap",
            "train_val_r2_gap_std": "std_train_val_r2_gap",
            "train_rmse_mean": "mean_train_rmse",
            "train_rmse_std": "std_train_rmse",
            "train_mae_mean": "mean_train_mae",
            "train_mae_std": "std_train_mae",
            "train_r2_mean": "mean_train_r2",
            "train_r2_std": "std_train_r2",
            "train_pearson_mean": "mean_train_pearson",
            "train_pearson_std": "std_train_pearson",
            "train_spearman_mean": "mean_train_spearman",
            "train_spearman_std": "std_train_spearman",
            "n_train_mean": "mean_n_train",
            "n_val_mean": "mean_n_val",
        }
    )
    sort_cols = ["mean_fold_rmse", "mean_fold_mae"]
    if extra_group_cols:
        sort_cols = extra_group_cols + sort_cols
    return summary.sort_values(sort_cols).reset_index(drop=True)


def summarize_best_hyperparameters(
    fold_metrics: pd.DataFrame,
    extra_group_cols: list[str] | None = None,
) -> pd.DataFrame:
    rows = []
    group_cols = ["scenario_run", "model", "feature_set"] + (extra_group_cols or [])
    for keys, group in fold_metrics.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        group_key = dict(zip(group_cols, keys))
        counts = (
            group.groupby("best_params")
            .agg(
                selection_count=("best_params", "size"),
                mean_outer_val_rmse=("rmse", "mean"),
                std_outer_val_rmse=("rmse", "std"),
                mean_outer_train_rmse=("train_rmse", "mean"),
                std_outer_train_rmse=("train_rmse", "std"),
                mean_outer_train_val_rmse_gap=("train_val_rmse_gap", "mean"),
                mean_outer_train_val_r2_gap=("train_val_r2_gap", "mean"),
                mean_inner_best_rmse=("inner_best_rmse", "mean"),
                std_inner_best_rmse=("inner_best_rmse", "std"),
                mean_inner_selected_train_rmse=("inner_selected_train_rmse", "mean"),
                mean_inner_selected_train_test_rmse_gap=(
                    "inner_selected_train_test_rmse_gap",
                    "mean",
                ),
                mean_inner_selected_complexity_score=(
                    "inner_selected_complexity_score",
                    "mean",
                ),
            )
            .reset_index()
            .sort_values(
                ["selection_count", "mean_outer_val_rmse", "mean_inner_best_rmse"],
                ascending=[False, True, True],
            )
            .reset_index(drop=True)
        )
        counts.insert(0, "rank", np.arange(1, len(counts) + 1))
        for col in reversed(group_cols):
            counts.insert(0, col, group_key[col])
        counts["selection_fraction"] = counts["selection_count"] / len(group)
        rows.append(counts)
    ranked = pd.concat(rows, ignore_index=True)
    return ranked.loc[ranked["rank"] == 1].sort_values(group_cols).reset_index(drop=True)


def summarize_repeats(
    fold_metrics: pd.DataFrame,
    extra_group_cols: list[str] | None = None,
) -> pd.DataFrame:
    group_cols = [
        "scenario_run",
        "model",
        "feature_set",
        *(extra_group_cols or []),
        "repeat",
        "repeat_seed",
    ]
    metric_cols = [
        col
        for col in [
            "rmse",
            "mae",
            "median_ae",
            "r2",
            "explained_variance",
            "mape",
            "max_error",
            "mean_error",
            "std_error",
            "mean_absolute_log_error",
            "rmse_log1p",
            "pearson",
            "spearman",
            "train_rmse",
            "train_mae",
            "train_r2",
            "train_val_rmse_gap",
            "train_val_rmse_ratio",
            "rmse_ratio",
            "train_val_r2_gap",
            "inner_best_rmse",
            "inner_selected_rmse",
            "inner_selected_rmse_gap_vs_min",
            "one_se_threshold",
            "one_se_candidate_count",
            "inner_selected_complexity_score",
        ]
        if col in fold_metrics.columns
    ]
    repeat_metrics = (
        fold_metrics.groupby(group_cols)[metric_cols + ["n_train", "n_val"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    repeat_metrics.columns = [
        "_".join(str(part) for part in col if part) if isinstance(col, tuple) else col
        for col in repeat_metrics.columns
    ]
    repeat_metrics = repeat_metrics.rename(
        columns={
            "rmse_mean": "mean_fold_rmse",
            "mae_mean": "mean_fold_mae",
            "r2_mean": "mean_fold_r2",
            "train_rmse_mean": "mean_train_rmse",
            "train_r2_mean": "mean_train_r2",
            "train_val_rmse_gap_mean": "mean_train_val_rmse_gap",
            "train_val_rmse_ratio_mean": "mean_train_val_rmse_ratio",
            "rmse_ratio_mean": "mean_rmse_ratio",
            "train_val_r2_gap_mean": "mean_train_val_r2_gap",
        }
    )
    sort_cols = ["feature_set", "model", *(extra_group_cols or []), "repeat"]
    return repeat_metrics.sort_values(sort_cols).reset_index(drop=True)


def write_metric_wide_tables(
    repeat_metrics: pd.DataFrame,
    outdir: Path,
    model_names: list[str],
    feature_set_names: list[str],
) -> list[str]:
    outdir.mkdir(parents=True, exist_ok=True)
    metric_cols = [
        "mean_fold_r2",
        "mean_fold_rmse",
        "mean_fold_mae",
        "mean_train_r2",
        "mean_train_rmse",
        "mean_train_val_rmse_gap",
        "mean_train_val_rmse_ratio",
        "mean_rmse_ratio",
        "mean_train_val_r2_gap",
        "inner_selected_rmse_mean",
        "inner_selected_rmse_gap_vs_min_mean",
        "one_se_candidate_count_mean",
    ]
    written = []
    repeats = sorted(int(value) for value in repeat_metrics["repeat"].dropna().unique())
    filename_map = {
        "mean_fold_r2": "r2",
        "mean_fold_rmse": "rmse",
        "mean_fold_mae": "mae",
    }
    for metric in [col for col in metric_cols if col in repeat_metrics.columns]:
        rows = []
        for feature_set in feature_set_names:
            row: dict[str, object] = {"feature_set": feature_set}
            for model_name in model_names:
                for repeat in repeats:
                    col_name = f"{model_name}__repeat_{repeat:02d}"
                    match = repeat_metrics.loc[
                        (repeat_metrics["feature_set"] == feature_set)
                        & (repeat_metrics["model"] == model_name)
                        & (repeat_metrics["repeat"] == repeat)
                    ]
                    row[col_name] = np.nan if match.empty else float(match.iloc[0][metric])
            rows.append(row)
        path = outdir / f"{filename_map.get(metric, metric.replace('mean_', ''))}.csv"
        pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
        written.append(str(path))
    return written


def build_selected_model_payload(
    summary: pd.DataFrame,
    best_params: pd.DataFrame,
    feature_set_names: list[str],
) -> dict[str, Any]:
    best_row = summary.sort_values(["mean_fold_rmse", "mean_fold_mae"]).iloc[0]
    selected_model = str(best_row["model"])
    params_by_feature_set = {}
    for feature_set in feature_set_names:
        row = best_params.loc[
            (best_params["model"] == selected_model)
            & (best_params["feature_set"] == feature_set)
        ]
        if row.empty:
            continue
        params_by_feature_set[feature_set] = json.loads(str(row.iloc[0]["best_params"]))
    return {
        "selection_basis": "best scenario by repeated outer-CV mean RMSE after inner 1-SE tuning",
        "selected_model": selected_model,
        "selected_feature_set": str(best_row["feature_set"]),
        "selected_scenario": str(best_row["scenario_run"]),
        "params_by_feature_set": params_by_feature_set,
        "best_summary": {
            key: (float(value) if isinstance(value, (float, np.floating)) else value)
            for key, value in best_row.to_dict().items()
            if key
            in {
                "scenario_run",
                "model",
                "feature_set",
                "mean_fold_rmse",
                "mean_fold_mae",
                "mean_fold_r2",
                "pooled_oof_rmse",
                "pooled_oof_r2",
                "mean_train_rmse",
                "mean_train_r2",
                "mean_train_val_rmse_gap",
                "mean_train_val_rmse_ratio",
                "mean_train_val_r2_gap",
            }
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Repeated nested cross-validation for model and feature-set comparison."
        )
    )
    parser.add_argument("--data", default="dataset.csv", type=Path)
    parser.add_argument("--outdir", default="results/model_feature_nested_cv", type=Path)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--repeats", default=10, type=int)
    parser.add_argument("--outer-folds", default=5, type=int)
    parser.add_argument("--inner-folds", default=3, type=int)
    parser.add_argument("--search-iter", default=80, type=int)
    parser.add_argument("--n-jobs", default=-1, type=int)
    parser.add_argument("--models", default="all")
    parser.add_argument("--feature-sets", default="all")
    parser.add_argument(
        "--param-profile",
        default="anti_overfit",
        choices=["anti_overfit", "baseline"],
        help="anti_overfit uses balanced regularized grids; baseline uses legacy broad grids.",
    )
    parser.add_argument(
        "--selection-rule",
        default="one_standard_error",
        choices=["one_standard_error", "best_rmse"],
        help="one_standard_error chooses the simplest candidate within one SE of the best inner RMSE.",
    )
    parser.add_argument(
        "--one-se-multiplier",
        default=1.0,
        type=float,
        help="SE width used by one_standard_error selection. 1.0 is the standard 1-SE default.",
    )
    parser.add_argument(
        "--feature-selection",
        default="auto",
        choices=["auto", "off"],
        help="auto tunes select__k for high-dimensional descriptor feature sets.",
    )
    args = parser.parse_args()
    if args.one_se_multiplier < 0:
        raise ValueError("--one-se-multiplier must be non-negative.")

    model_names = parse_csv_choices(args.models, MODEL_NAMES, "--models")
    feature_set_names = parse_csv_choices(args.feature_sets, FEATURE_SET_NAMES, "--feature-sets")
    repeat_seeds = [args.seed + idx for idx in range(args.repeats)]

    args.outdir.mkdir(parents=True, exist_ok=True)
    df = read_training_data(args.data)
    y = df[TARGET_COL]
    feature_sets = build_feature_sets(df)
    cleaning_report = smiles_cleaning_report(df)

    fold_frames = []
    pred_frames = []
    train_pred_frames = []
    selection_fold_frames = []
    selection_pred_frames = []
    selection_train_pred_frames = []
    for model_name in model_names:
        for feature_set_name in feature_set_names:
            print(f"Running {model_name} / {feature_set_name}...")
            (
                fold_metrics,
                predictions,
                train_predictions,
                selection_fold_metrics,
                selection_predictions,
                selection_train_predictions,
            ) = run_nested_cv(
                model_name=model_name,
                feature_set_name=feature_set_name,
                x=feature_sets[feature_set_name],
                y=y,
                repeat_seeds=repeat_seeds,
                outer_folds=args.outer_folds,
                inner_folds=args.inner_folds,
                search_iter=args.search_iter,
                n_jobs=args.n_jobs,
                param_profile=args.param_profile,
                selection_rule=args.selection_rule,
                feature_selection=args.feature_selection,
                one_se_multiplier=args.one_se_multiplier,
                collect_selection_comparison=True,
            )
            fold_frames.append(fold_metrics)
            pred_frames.append(predictions)
            train_pred_frames.append(train_predictions)
            selection_fold_frames.append(selection_fold_metrics)
            selection_pred_frames.append(selection_predictions)
            selection_train_pred_frames.append(selection_train_predictions)

    fold_metrics = pd.concat(fold_frames, ignore_index=True)
    predictions = pd.concat(pred_frames, ignore_index=True)
    train_predictions = pd.concat(train_pred_frames, ignore_index=True)
    selection_fold_metrics = pd.concat(selection_fold_frames, ignore_index=True)
    selection_predictions = pd.concat(selection_pred_frames, ignore_index=True)
    selection_train_predictions = pd.concat(selection_train_pred_frames, ignore_index=True)
    summary = summarize_cv(fold_metrics, predictions, train_predictions)
    repeat_metrics = summarize_repeats(fold_metrics)
    best_params = summarize_best_hyperparameters(fold_metrics)
    summary = summary.merge(
        best_params[["scenario_run", "best_params", "selection_count", "selection_fraction"]],
        on="scenario_run",
        how="left",
    ).rename(
        columns={
            "best_params": "consensus_best_params",
            "selection_count": "consensus_best_params_count",
            "selection_fraction": "consensus_best_params_fraction",
        }
    )
    selection_summary = summarize_cv(
        selection_fold_metrics,
        selection_predictions,
        selection_train_predictions,
        extra_group_cols=["selection_rule"],
    )
    selection_repeat_metrics = summarize_repeats(
        selection_fold_metrics,
        extra_group_cols=["selection_rule"],
    )
    selection_best_params = summarize_best_hyperparameters(
        selection_fold_metrics,
        extra_group_cols=["selection_rule"],
    )
    selection_summary = selection_summary.merge(
        selection_best_params[
            [
                "scenario_run",
                "model",
                "feature_set",
                "selection_rule",
                "best_params",
                "selection_count",
                "selection_fraction",
            ]
        ],
        on=["scenario_run", "model", "feature_set", "selection_rule"],
        how="left",
    ).rename(
        columns={
            "best_params": "consensus_best_params",
            "selection_count": "consensus_best_params_count",
            "selection_fraction": "consensus_best_params_fraction",
        }
    )
    selected_model = build_selected_model_payload(summary, best_params, feature_set_names)
    profile_n_train = len(df) - int(np.ceil(len(df) / args.outer_folds))
    search_grids_by_feature_set = {
        model: {
            feature_set: effective_search_grid(
                model_name=model,
                feature_set_name=feature_set,
                n_features=int(feature_sets[feature_set].shape[1]),
                n_train=profile_n_train,
                inner_folds=args.inner_folds,
                param_profile=args.param_profile,
                feature_selection=args.feature_selection,
            )
            for feature_set in feature_set_names
        }
        for model in model_names
    }

    paths = {
        "summary": args.outdir / "cv_summary.csv",
        "selection_summary": args.outdir / "cv_summary_selection_rules.csv",
        "repeat_metrics": args.outdir / "repeat_metrics.csv",
        "selection_repeat_metrics": args.outdir / "repeat_metrics_selection_rules.csv",
        "fold_metrics": args.outdir / "fold_metrics.csv",
        "selection_fold_metrics": args.outdir / "fold_metrics_selection_rules.csv",
        "predictions": args.outdir / "cv_predictions.csv",
        "selection_predictions": args.outdir / "cv_predictions_selection_rules.csv",
        "train_predictions": args.outdir / "cv_train_predictions.csv",
        "selection_train_predictions": args.outdir / "cv_train_predictions_selection_rules.csv",
        "best_params": args.outdir / "best_hyperparameters.csv",
        "selection_best_params": args.outdir / "best_hyperparameters_selection_rules.csv",
        "selected_model": args.outdir / "selected_model.json",
        "metric_wide_dir": args.outdir / "metric_wide" / "repeat_mean",
        "cleaning": args.outdir / "smiles_cleaning_report.csv",
        "profile": args.outdir / "data_profile.json",
        "report": args.outdir / "report.md",
    }
    summary.to_csv(paths["summary"], index=False, encoding="utf-8-sig")
    selection_summary.to_csv(paths["selection_summary"], index=False, encoding="utf-8-sig")
    repeat_metrics.to_csv(paths["repeat_metrics"], index=False, encoding="utf-8-sig")
    selection_repeat_metrics.to_csv(
        paths["selection_repeat_metrics"],
        index=False,
        encoding="utf-8-sig",
    )
    fold_metrics.to_csv(paths["fold_metrics"], index=False, encoding="utf-8-sig")
    selection_fold_metrics.to_csv(
        paths["selection_fold_metrics"],
        index=False,
        encoding="utf-8-sig",
    )
    predictions.to_csv(paths["predictions"], index=False, encoding="utf-8-sig")
    selection_predictions.to_csv(
        paths["selection_predictions"],
        index=False,
        encoding="utf-8-sig",
    )
    train_predictions.to_csv(paths["train_predictions"], index=False, encoding="utf-8-sig")
    selection_train_predictions.to_csv(
        paths["selection_train_predictions"],
        index=False,
        encoding="utf-8-sig",
    )
    best_params.to_csv(paths["best_params"], index=False, encoding="utf-8-sig")
    selection_best_params.to_csv(paths["selection_best_params"], index=False, encoding="utf-8-sig")
    metric_wide_files = write_metric_wide_tables(
        repeat_metrics=repeat_metrics,
        outdir=paths["metric_wide_dir"],
        model_names=model_names,
        feature_set_names=feature_set_names,
    )
    paths["selected_model"].write_text(
        json.dumps(selected_model, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    cleaning_report.to_csv(paths["cleaning"], index=False, encoding="utf-8-sig")

    profile = {
        "data": str(args.data),
        "target": TARGET_COL,
        "n_samples": int(len(df)),
        "models": model_names,
        "feature_sets": feature_set_names,
        "feature_set_sizes": {name: int(feature_sets[name].shape[1]) for name in feature_set_names},
        "repeats": args.repeats,
        "repeat_seeds": repeat_seeds,
        "outer_folds": args.outer_folds,
        "inner_folds": args.inner_folds,
        "search_iter": args.search_iter,
        "param_profile": args.param_profile,
        "selection_rule": args.selection_rule,
        "primary_outputs_use_selection_rule": args.selection_rule,
        "selection_rules_compared": ["best_rmse", "one_standard_error"],
        "one_se_multiplier": args.one_se_multiplier,
        "feature_selection": args.feature_selection,
        "overfit_controls": [
            "nested cross-validation",
            "balanced regularized anti-overfit parameter grids",
            "one-standard-error hyperparameter selection by default",
            "train metrics and train-validation gaps per outer fold",
            "SelectKBest feature-count tuning for high-dimensional descriptor feature sets",
            "subsampling, column sampling, stronger tree depth and leaf-size constraints",
        ],
        "fold_metrics_include_train_metrics": True,
        "repeat_metrics_file": str(paths["repeat_metrics"]),
        "selection_rule_comparison_files": {
            "summary": str(paths["selection_summary"]),
            "repeat_metrics": str(paths["selection_repeat_metrics"]),
            "fold_metrics": str(paths["selection_fold_metrics"]),
            "predictions": str(paths["selection_predictions"]),
            "train_predictions": str(paths["selection_train_predictions"]),
            "best_hyperparameters": str(paths["selection_best_params"]),
        },
        "metric_wide_files": metric_wide_files,
        "selected_model_file": str(paths["selected_model"]),
        "train_predictions_file": str(paths["train_predictions"]),
        "baseline_search_grids": {model: baseline_param_grid(model) for model in model_names},
        "effective_search_grids_by_feature_set": search_grids_by_feature_set,
    }
    paths["profile"].write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")

    report_cols = [
        "scenario_run",
        "model",
        "feature_set",
        "mean_fold_rmse",
        "std_fold_rmse",
        "mean_fold_mae",
        "mean_fold_r2",
        "std_fold_r2",
        "mean_train_rmse",
        "mean_train_mae",
        "mean_train_r2",
        "pooled_train_rmse",
        "pooled_train_r2",
        "mean_train_val_rmse_gap",
        "mean_train_val_rmse_ratio",
        "mean_train_val_r2_gap",
        "mean_inner_selected_rmse_gap_vs_min",
        "mean_one_se_candidate_count",
        "pooled_oof_rmse",
        "pooled_oof_r2",
        "pooled_oof_pearson",
        "pooled_oof_spearman",
        "n_predictions",
        "consensus_best_params_count",
        "consensus_best_params_fraction",
    ]
    selection_report_cols = [
        "selection_rule",
        "scenario_run",
        "model",
        "feature_set",
        "mean_fold_rmse",
        "std_fold_rmse",
        "mean_fold_mae",
        "mean_fold_r2",
        "std_fold_r2",
        "mean_train_rmse",
        "mean_train_r2",
        "pooled_train_rmse",
        "pooled_train_r2",
        "mean_train_val_rmse_gap",
        "mean_train_val_rmse_ratio",
        "mean_train_val_r2_gap",
        "mean_inner_selected_rmse_gap_vs_min",
        "mean_one_se_candidate_count",
        "pooled_oof_rmse",
        "pooled_oof_r2",
        "pooled_oof_pearson",
        "pooled_oof_spearman",
        "n_predictions",
        "consensus_best_params_count",
        "consensus_best_params_fraction",
    ]
    best = summary.iloc[0]
    report = [
        "# Model and feature nested CV evaluation",
        "",
        f"- Data: `{args.data}`",
        f"- Samples: {len(df)}",
        f"- Repeated CV: {args.repeats} x {args.outer_folds}-fold",
        f"- Inner tuning: {args.inner_folds}-fold RandomizedSearchCV in every outer fold, n_iter={args.search_iter}",
        f"- Parameter profile: `{args.param_profile}`",
        f"- Hyperparameter selection rule: `{args.selection_rule}`",
        f"- One-SE multiplier: {args.one_se_multiplier}",
        f"- Feature selection: `{args.feature_selection}`",
        f"- Selected model for downstream analyses: `{selected_model['selected_model']}`",
        f"- Best scenario: `{best['scenario_run']}`",
        f"- Best mean fold RMSE: {best['mean_fold_rmse']:.3f}",
        f"- Best mean fold R2: {best['mean_fold_r2']:.6f}",
        f"- Metric wide tables: `{paths['metric_wide_dir']}`",
        "",
        "## CV Summary",
        "",
        dataframe_to_markdown(summary[report_cols]),
        "",
        "## Best RMSE vs 1-SE Selection Rule Comparison",
        "",
        dataframe_to_markdown(selection_summary[selection_report_cols]),
        "",
        "## Output Files",
        "",
    ]
    report.extend(f"- `{path}`: {name}" for name, path in paths.items() if name != "report")
    report.append("")
    paths["report"].write_text("\n".join(report), encoding="utf-8")

    print(summary[report_cols].to_string(index=False))
    print("\nSelection rule comparison:")
    print(selection_summary[selection_report_cols].to_string(index=False))
    print(f"\nWrote outputs to: {args.outdir}")


if __name__ == "__main__":
    main()
