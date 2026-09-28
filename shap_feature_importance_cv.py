from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold

from ml_common import (
    FEATURE_SET_NAMES,
    ID_COL,
    NAME_COL,
    SMILES_COL,
    TARGET_COL,
    build_feature_sets,
    dataframe_to_markdown,
    feature_names_after_variance,
    make_estimator,
    read_training_data,
    regression_metrics,
    transformed_after_variance,
)


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return cleaned or "feature_set"


def load_params_for_model(best_params_path: Path, model_name: str) -> dict[str, dict[str, Any]]:
    best_params = pd.read_csv(best_params_path)
    params = {}
    for feature_set in FEATURE_SET_NAMES:
        row = best_params.loc[
            (best_params["model"] == model_name) & (best_params["feature_set"] == feature_set)
        ]
        if row.empty:
            raise ValueError(f"No best params found for {model_name}/{feature_set}.")
        params[feature_set] = json.loads(str(row.iloc[0]["best_params"]))
    return params


def load_model_cv_selection(model_cv_dir: Path) -> tuple[str, dict[str, dict[str, Any]], list[str]]:
    selected_path = model_cv_dir / "selected_model.json"
    if selected_path.exists():
        payload = json.loads(selected_path.read_text(encoding="utf-8"))
        params = {
            str(feature_set): dict(values)
            for feature_set, values in payload["params_by_feature_set"].items()
        }
        return str(payload["selected_model"]), params, list(params)

    summary = pd.read_csv(model_cv_dir / "cv_summary.csv")
    model_name = str(summary.sort_values(["mean_fold_rmse", "mean_fold_mae"]).iloc[0]["model"])
    params = load_params_for_model(model_cv_dir / "best_hyperparameters.csv", model_name)
    return model_name, params, list(params)


def prefix_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def model_feature_importance(
    estimator,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    feature_names: list[str],
    seed: int,
    permutation_repeats: int,
) -> pd.DataFrame:
    model = estimator.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        return pd.DataFrame(
            {
                "feature": feature_names,
                "importance": np.asarray(model.feature_importances_, dtype=float),
                "importance_type": "model_feature_importances",
            }
        )

    result = permutation_importance(
        estimator,
        x_val,
        y_val,
        n_repeats=permutation_repeats,
        random_state=seed,
        scoring="neg_root_mean_squared_error",
    )
    return pd.DataFrame(
        {
            "feature": list(x_val.columns),
            "importance": result.importances_mean,
            "importance_std": result.importances_std,
            "importance_type": "permutation_neg_rmse_drop",
        }
    )


def xgboost_shap_summary(estimator, x_val: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    x_transformed = transformed_after_variance(estimator, x_val)
    booster = estimator.named_steps["model"].get_booster()
    dmatrix = xgb.DMatrix(x_transformed, feature_names=feature_names)
    contrib = booster.predict(dmatrix, pred_contribs=True)
    shap_values = contrib[:, :-1]
    return pd.DataFrame(
        {
            "feature": feature_names,
            "mean_shap": np.mean(shap_values, axis=0),
            "mean_abs_shap": np.mean(np.abs(shap_values), axis=0),
            "std_abs_shap": np.std(np.abs(shap_values), axis=0, ddof=1),
        }
    )


def run_cv_importance(
    df: pd.DataFrame,
    feature_sets: dict[str, pd.DataFrame],
    y: pd.Series,
    model_name: str,
    params_by_feature_set: dict[str, dict[str, object]],
    feature_set_names: list[str],
    repeats: int,
    folds: int,
    seed: int,
    permutation_repeats: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    repeat_seeds = [seed + idx for idx in range(repeats)]
    metric_rows = []
    pred_frames = []
    train_pred_frames = []
    importance_frames = []
    shap_frames = []

    for feature_set in feature_set_names:
        x = feature_sets[feature_set]
        params = params_by_feature_set[feature_set]
        for repeat_idx, repeat_seed in enumerate(repeat_seeds, start=1):
            cv = KFold(n_splits=folds, shuffle=True, random_state=repeat_seed)
            for fold_idx, (train_idx, val_idx) in enumerate(cv.split(x), start=1):
                fold_seed = repeat_seed + fold_idx
                estimator = make_estimator(model_name, fold_seed)
                estimator.set_params(**params)
                x_train = x.iloc[train_idx]
                x_val = x.iloc[val_idx]
                y_train = y.iloc[train_idx].to_numpy(dtype=float)
                y_val = y.iloc[val_idx].to_numpy(dtype=float)
                estimator.fit(x_train, y_train)
                y_pred = np.asarray(estimator.predict(x_val), dtype=float)
                y_train_pred = np.asarray(estimator.predict(x_train), dtype=float)

                metrics = regression_metrics(y_val, y_pred)
                metrics.update(prefix_metrics(regression_metrics(y_train, y_train_pred), "train"))
                metrics.update(
                    {
                        "model": model_name,
                        "feature_set": feature_set,
                        "repeat": repeat_idx,
                        "repeat_seed": repeat_seed,
                        "fold": fold_idx,
                        "n_train": int(len(train_idx)),
                        "n_val": int(len(val_idx)),
                        "n_features_input": int(x.shape[1]),
                        "best_params": json.dumps(params, ensure_ascii=False, sort_keys=True),
                    }
                )
                metric_rows.append(metrics)
                pred_frames.append(
                    pd.DataFrame(
                        {
                            "model": model_name,
                            "feature_set": feature_set,
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
                )
                train_pred_frames.append(
                    pd.DataFrame(
                        {
                            "model": model_name,
                            "feature_set": feature_set,
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
                )

                selected_feature_names = feature_names_after_variance(estimator, list(x.columns))
                importance = model_feature_importance(
                    estimator,
                    x_val,
                    y_val,
                    selected_feature_names,
                    fold_seed,
                    permutation_repeats,
                )
                importance["model"] = model_name
                importance["feature_set"] = feature_set
                importance["repeat"] = repeat_idx
                importance["fold"] = fold_idx
                importance_frames.append(importance)

                if model_name == "XGBoost":
                    shap_summary = xgboost_shap_summary(estimator, x_val, selected_feature_names)
                    shap_summary["model"] = model_name
                    shap_summary["feature_set"] = feature_set
                    shap_summary["repeat"] = repeat_idx
                    shap_summary["fold"] = fold_idx
                    shap_frames.append(shap_summary)

    shap_df = pd.concat(shap_frames, ignore_index=True) if shap_frames else pd.DataFrame()
    return (
        pd.DataFrame(metric_rows),
        pd.concat(pred_frames, ignore_index=True),
        pd.concat(train_pred_frames, ignore_index=True),
        pd.concat(importance_frames, ignore_index=True),
        shap_df,
    )


def summarize_feature_tables(importance: pd.DataFrame, shap_values: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    importance_summary = (
        importance.groupby(["model", "feature_set", "feature", "importance_type"])["importance"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_importance", "std": "std_importance", "count": "n_folds"})
        .sort_values(["feature_set", "mean_importance"], ascending=[True, False])
    )
    if shap_values.empty:
        shap_summary = pd.DataFrame()
    else:
        shap_summary = (
            shap_values.groupby(["model", "feature_set", "feature"])[
                ["mean_abs_shap", "mean_shap", "std_abs_shap"]
            ]
            .agg(["mean", "std"])
            .reset_index()
        )
        shap_summary.columns = [
            "_".join(str(part) for part in col if part) if isinstance(col, tuple) else col
            for col in shap_summary.columns
        ]
        shap_summary = shap_summary.rename(
            columns={
                "mean_abs_shap_mean": "mean_abs_shap",
                "mean_abs_shap_std": "std_mean_abs_shap",
                "mean_shap_mean": "mean_shap",
                "std_abs_shap_mean": "mean_std_abs_shap",
            }
        ).sort_values(["feature_set", "mean_abs_shap"], ascending=[True, False])
    return importance_summary.reset_index(drop=True), shap_summary.reset_index(drop=True)


def compute_feature_correlations(
    feature_sets: dict[str, pd.DataFrame],
    y: pd.Series,
    model_name: str,
    params_by_feature_set: dict[str, dict[str, Any]],
    feature_set_names: list[str],
    seed: int,
    abs_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    pair_frames = []
    summary_frames = []
    matrices = {}

    for feature_set in feature_set_names:
        x = feature_sets[feature_set]
        estimator = make_estimator(model_name, seed)
        estimator.set_params(**params_by_feature_set[feature_set])
        estimator.fit(x, y.to_numpy(dtype=float))
        feature_names = feature_names_after_variance(estimator, list(x.columns))
        transformed = transformed_after_variance(estimator, x)
        x_model_space = pd.DataFrame(transformed, columns=feature_names)
        corr = x_model_space.corr(method="pearson")
        matrices[feature_set] = corr

        if len(feature_names) < 2:
            summary_frames.append(
                pd.DataFrame(
                    {
                        "model": [model_name],
                        "feature_set": [feature_set],
                        "feature": feature_names,
                        "n_correlated_features": [0 for _ in feature_names],
                        "max_abs_pearson_r": [np.nan for _ in feature_names],
                        "mean_abs_pearson_r": [np.nan for _ in feature_names],
                        "n_abs_pearson_ge_threshold": [0 for _ in feature_names],
                    }
                )
            )
            continue

        corr_values = corr.to_numpy(dtype=float)
        upper_i, upper_j = np.triu_indices_from(corr_values, k=1)
        pairs = pd.DataFrame(
            {
                "model": model_name,
                "feature_set": feature_set,
                "feature_1": [feature_names[idx] for idx in upper_i],
                "feature_2": [feature_names[idx] for idx in upper_j],
                "pearson_r": corr_values[upper_i, upper_j],
            }
        ).dropna(subset=["pearson_r"])
        pairs["abs_pearson_r"] = pairs["pearson_r"].abs()
        pairs = pairs.loc[pairs["abs_pearson_r"] >= abs_threshold].sort_values(
            "abs_pearson_r",
            ascending=False,
        )
        pairs.insert(0, "pair_rank", np.arange(1, len(pairs) + 1))
        pair_frames.append(pairs)

        abs_corr = np.abs(corr_values)
        np.fill_diagonal(abs_corr, np.nan)
        summary = pd.DataFrame(
            {
                "model": model_name,
                "feature_set": feature_set,
                "feature": feature_names,
                "n_correlated_features": np.sum(~np.isnan(abs_corr), axis=1).astype(int),
                "max_abs_pearson_r": np.nanmax(abs_corr, axis=1),
                "mean_abs_pearson_r": np.nanmean(abs_corr, axis=1),
                "n_abs_pearson_ge_threshold": np.nansum(
                    abs_corr >= abs_threshold,
                    axis=1,
                ).astype(int),
            }
        ).sort_values(
            ["n_abs_pearson_ge_threshold", "max_abs_pearson_r"],
            ascending=[False, False],
        )
        summary_frames.append(summary)

    pairs_df = pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame(
        columns=[
            "pair_rank",
            "model",
            "feature_set",
            "feature_1",
            "feature_2",
            "pearson_r",
            "abs_pearson_r",
        ]
    )
    summary_df = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    return pairs_df, summary_df, matrices


def safe_series_corr(a: pd.Series, b: pd.Series, method: str) -> float:
    paired = pd.concat([a, b], axis=1).dropna()
    if len(paired) < 2:
        return np.nan
    if paired.iloc[:, 0].nunique(dropna=True) < 2 or paired.iloc[:, 1].nunique(dropna=True) < 2:
        return np.nan
    return float(paired.iloc[:, 0].corr(paired.iloc[:, 1], method=method))


def xgboost_all_feature_values_and_target_correlations(
    df: pd.DataFrame,
    feature_sets: dict[str, pd.DataFrame],
    y: pd.Series,
    params_by_feature_set: dict[str, dict[str, Any]],
    seed: int,
    feature_set: str = "provided_plus_rdkit_descriptors",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if feature_set not in feature_sets:
        raise ValueError(f"Feature set is unavailable: {feature_set}")
    if feature_set not in params_by_feature_set:
        raise ValueError(f"No XGBoost parameters found for feature set: {feature_set}")

    x = feature_sets[feature_set]
    params = dict(params_by_feature_set[feature_set])
    params["select__k"] = "all"
    estimator = make_estimator("XGBoost", seed)
    estimator.set_params(**params)
    estimator.fit(x, y.to_numpy(dtype=float))
    input_feature_names = list(x.columns)
    model_feature_names = set(feature_names_after_variance(estimator, input_feature_names))
    feature_values = x.reset_index(drop=True).copy()

    id_cols = {"row_index": np.arange(len(df)), TARGET_COL: y.to_numpy(dtype=float)}
    if ID_COL in df.columns:
        id_cols[ID_COL] = df[ID_COL].to_numpy()
    if NAME_COL in df.columns:
        id_cols[NAME_COL] = df[NAME_COL].to_numpy()
    if SMILES_COL in df.columns:
        id_cols[SMILES_COL] = df[SMILES_COL].to_numpy()
    id_df = pd.DataFrame(id_cols)
    ordered_id_cols = [
        col for col in ["row_index", ID_COL, NAME_COL, TARGET_COL, SMILES_COL] if col in id_df.columns
    ]
    id_df = id_df[ordered_id_cols]

    wide = pd.concat([id_df.reset_index(drop=True), feature_values.reset_index(drop=True)], axis=1)
    correlations = pd.DataFrame(
        {
            "model": "XGBoost",
            "feature_set": feature_set,
            "feature": input_feature_names,
            "used_by_xgboost_after_pipeline": [
                feature in model_feature_names for feature in input_feature_names
            ],
            "n_samples": [
                int(pd.concat([feature_values[feature], y], axis=1).dropna().shape[0])
                for feature in input_feature_names
            ],
            "pearson_r_with_target": [
                safe_series_corr(feature_values[feature], y, "pearson") for feature in input_feature_names
            ],
            "spearman_r_with_target": [
                safe_series_corr(feature_values[feature], y, "spearman") for feature in input_feature_names
            ],
        }
    )
    correlations["abs_pearson_r_with_target"] = correlations["pearson_r_with_target"].abs()
    correlations["abs_spearman_r_with_target"] = correlations["spearman_r_with_target"].abs()
    correlations = correlations.sort_values(
        ["abs_pearson_r_with_target", "abs_spearman_r_with_target"],
        ascending=[False, False],
    ).reset_index(drop=True)
    matrix_input = pd.concat(
        [
            pd.Series(y.to_numpy(dtype=float), name=TARGET_COL),
            feature_values,
        ],
        axis=1,
    )
    pearson_matrix = matrix_input.corr(method="pearson")
    spearman_matrix = matrix_input.corr(method="spearman")

    long = feature_values.copy()
    long.insert(0, TARGET_COL, y.to_numpy(dtype=float))
    if NAME_COL in df.columns:
        long.insert(0, NAME_COL, df[NAME_COL].to_numpy())
    if ID_COL in df.columns:
        long.insert(0, ID_COL, df[ID_COL].to_numpy())
    long.insert(0, "row_index", np.arange(len(df)))
    long = long.melt(
        id_vars=[col for col in ["row_index", ID_COL, NAME_COL, TARGET_COL] if col in long.columns],
        var_name="feature",
        value_name="feature_value",
    )
    long = long.merge(
        correlations[
            [
                "feature",
                "used_by_xgboost_after_pipeline",
                "pearson_r_with_target",
                "spearman_r_with_target",
                "abs_pearson_r_with_target",
                "abs_spearman_r_with_target",
            ]
        ],
        on="feature",
        how="left",
    )
    long.insert(0, "model", "XGBoost")
    long.insert(1, "feature_set", feature_set)
    return wide, long, correlations, pearson_matrix, spearman_matrix


def summarize_shap_stability(
    shap_values: pd.DataFrame,
    top_k_values: tuple[int, ...] = (5, 10, 20),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if shap_values.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    ranked = shap_values.copy()
    ranked["fold_id"] = (
        "repeat_"
        + ranked["repeat"].astype(int).astype(str).str.zfill(2)
        + "__fold_"
        + ranked["fold"].astype(int).astype(str).str.zfill(2)
    )
    ranked["shap_rank"] = ranked.groupby(["model", "feature_set", "fold_id"])[
        "mean_abs_shap"
    ].rank(method="average", ascending=False)

    total_folds = (
        ranked[["model", "feature_set", "fold_id"]]
        .drop_duplicates()
        .groupby(["model", "feature_set"])
        .size()
        .rename("total_folds")
        .reset_index()
    )

    grouped = (
        ranked.groupby(["model", "feature_set", "feature"])
        .agg(
            n_folds_observed=("fold_id", "nunique"),
            mean_abs_shap=("mean_abs_shap", "mean"),
            std_abs_shap_across_folds=("mean_abs_shap", "std"),
            median_abs_shap=("mean_abs_shap", "median"),
            q25_abs_shap=("mean_abs_shap", lambda s: float(s.quantile(0.25))),
            q75_abs_shap=("mean_abs_shap", lambda s: float(s.quantile(0.75))),
            mean_shap=("mean_shap", "mean"),
            std_mean_shap_across_folds=("mean_shap", "std"),
            mean_rank=("shap_rank", "mean"),
            std_rank=("shap_rank", "std"),
            median_rank=("shap_rank", "median"),
            best_rank=("shap_rank", "min"),
            worst_rank=("shap_rank", "max"),
            positive_mean_shap_fraction=("mean_shap", lambda s: float((s > 0).mean())),
            negative_mean_shap_fraction=("mean_shap", lambda s: float((s < 0).mean())),
        )
        .reset_index()
        .merge(total_folds, on=["model", "feature_set"], how="left")
    )
    grouped["presence_fraction"] = grouped["n_folds_observed"] / grouped["total_folds"]
    grouped["cv_abs_shap"] = (
        grouped["std_abs_shap_across_folds"] / grouped["mean_abs_shap"].replace(0, np.nan)
    )
    grouped["iqr_abs_shap"] = grouped["q75_abs_shap"] - grouped["q25_abs_shap"]
    grouped["sign_consistency"] = grouped[
        ["positive_mean_shap_fraction", "negative_mean_shap_fraction"]
    ].max(axis=1)

    topk_frames = []
    for top_k in top_k_values:
        topk = (
            ranked.assign(in_top_k=ranked["shap_rank"] <= top_k)
            .groupby(["model", "feature_set", "feature"])["in_top_k"]
            .mean()
            .rename(f"top_{top_k}_frequency")
            .reset_index()
        )
        topk_frames.append(topk)
    for topk in topk_frames:
        grouped = grouped.merge(topk, on=["model", "feature_set", "feature"], how="left")

    stability = grouped.sort_values(
        ["feature_set", "mean_abs_shap", "presence_fraction"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    pair_rows = []
    for (model, feature_set), group in ranked.groupby(["model", "feature_set"]):
        pivot = group.pivot_table(
            index="feature",
            columns="fold_id",
            values="mean_abs_shap",
            aggfunc="first",
        ).fillna(0.0)
        if pivot.shape[1] < 2 or pivot.shape[0] < 2:
            continue
        ranks = pivot.rank(axis=0, method="average", ascending=False)
        top10 = {
            fold_id: set(pivot[fold_id].nlargest(min(10, len(pivot))).index)
            for fold_id in pivot.columns
        }
        for fold_a, fold_b in combinations(pivot.columns, 2):
            corr = ranks[fold_a].corr(ranks[fold_b], method="spearman")
            union = top10[fold_a] | top10[fold_b]
            intersection = top10[fold_a] & top10[fold_b]
            pair_rows.append(
                {
                    "model": model,
                    "feature_set": feature_set,
                    "fold_a": fold_a,
                    "fold_b": fold_b,
                    "spearman_rank_corr": corr,
                    "top10_jaccard": len(intersection) / len(union) if union else np.nan,
                    "n_features_ranked": int(pivot.shape[0]),
                }
            )
    rank_pairs = pd.DataFrame(pair_rows)
    if rank_pairs.empty:
        rank_summary = pd.DataFrame()
    else:
        rank_summary = (
            rank_pairs.groupby(["model", "feature_set"])[
                ["spearman_rank_corr", "top10_jaccard", "n_features_ranked"]
            ]
            .agg(["mean", "std", "median", "min", "max", "count"])
            .reset_index()
        )
        rank_summary.columns = [
            "_".join(str(part) for part in col if part) if isinstance(col, tuple) else col
            for col in rank_summary.columns
        ]
        rank_summary = rank_summary.rename(
            columns={
                "spearman_rank_corr_mean": "mean_spearman_rank_corr",
                "spearman_rank_corr_std": "std_spearman_rank_corr",
                "spearman_rank_corr_median": "median_spearman_rank_corr",
                "spearman_rank_corr_min": "min_spearman_rank_corr",
                "spearman_rank_corr_max": "max_spearman_rank_corr",
                "spearman_rank_corr_count": "n_fold_pairs",
                "top10_jaccard_mean": "mean_top10_jaccard",
                "top10_jaccard_std": "std_top10_jaccard",
                "top10_jaccard_median": "median_top10_jaccard",
                "top10_jaccard_min": "min_top10_jaccard",
                "top10_jaccard_max": "max_top10_jaccard",
                "n_features_ranked_mean": "mean_n_features_ranked",
            }
        )
    return stability, rank_pairs, rank_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Feature importance, SHAP summary, feature correlation, and SHAP stability."
        )
    )
    parser.add_argument("--data", default="dataset.csv", type=Path)
    parser.add_argument("--model-cv-dir", default="results/model_feature_nested_cv", type=Path)
    parser.add_argument("--outdir", default="results/feature_analysis", type=Path)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--repeats", default=10, type=int)
    parser.add_argument("--folds", default=5, type=int)
    parser.add_argument("--permutation-repeats", default=5, type=int)
    parser.add_argument(
        "--correlation-threshold",
        default=0.8,
        type=float,
        help="Absolute Pearson r threshold for reporting high-correlation feature pairs.",
    )
    args = parser.parse_args()
    if not 0 <= args.correlation_threshold <= 1:
        raise ValueError("--correlation-threshold must be between 0 and 1.")

    args.outdir.mkdir(parents=True, exist_ok=True)
    model_name, params_by_feature_set, feature_set_names = load_model_cv_selection(args.model_cv_dir)

    df = read_training_data(args.data)
    y = df[TARGET_COL]
    feature_sets = build_feature_sets(df)
    metrics, predictions, train_predictions, importance, shap_values = run_cv_importance(
        df=df,
        feature_sets=feature_sets,
        y=y,
        model_name=model_name,
        params_by_feature_set=params_by_feature_set,
        feature_set_names=feature_set_names,
        repeats=args.repeats,
        folds=args.folds,
        seed=args.seed,
        permutation_repeats=args.permutation_repeats,
    )
    importance_summary, shap_summary = summarize_feature_tables(importance, shap_values)
    correlation_pairs, correlation_summary, correlation_matrices = compute_feature_correlations(
        feature_sets=feature_sets,
        y=y,
        model_name=model_name,
        params_by_feature_set=params_by_feature_set,
        feature_set_names=feature_set_names,
        seed=args.seed,
        abs_threshold=args.correlation_threshold,
    )
    shap_stability, shap_rank_pairs, shap_rank_summary = summarize_shap_stability(shap_values)
    xgb_all_feature_values_wide = pd.DataFrame()
    xgb_all_feature_values_long = pd.DataFrame()
    xgb_all_feature_target_correlations = pd.DataFrame()
    xgb_all_feature_pearson_matrix = pd.DataFrame()
    xgb_all_feature_spearman_matrix = pd.DataFrame()
    if model_name == "XGBoost" and "provided_plus_rdkit_descriptors" in params_by_feature_set:
        (
            xgb_all_feature_values_wide,
            xgb_all_feature_values_long,
            xgb_all_feature_target_correlations,
            xgb_all_feature_pearson_matrix,
            xgb_all_feature_spearman_matrix,
        ) = xgboost_all_feature_values_and_target_correlations(
            df=df,
            feature_sets=feature_sets,
            y=y,
            params_by_feature_set=params_by_feature_set,
            seed=args.seed,
            feature_set="provided_plus_rdkit_descriptors",
        )

    paths = {
        "metrics": args.outdir / "fold_metrics.csv",
        "predictions": args.outdir / "cv_predictions.csv",
        "train_predictions": args.outdir / "cv_train_predictions.csv",
        "importance_fold": args.outdir / "feature_importance_fold.csv",
        "importance_summary": args.outdir / "feature_importance_summary.csv",
        "shap_fold": args.outdir / "shap_fold_feature_summary.csv",
        "shap_summary": args.outdir / "shap_summary.csv",
        "shap_stability": args.outdir / "shap_stability_summary.csv",
        "shap_rank_stability_pairs": args.outdir / "shap_rank_stability_fold_pairs.csv",
        "shap_rank_stability_summary": args.outdir / "shap_rank_stability_summary.csv",
        "correlation_summary": args.outdir / "feature_correlation_summary.csv",
        "correlation_pairs": args.outdir / "feature_correlation_high_pairs.csv",
        "correlation_matrix_dir": args.outdir / "feature_correlation_matrices",
        "xgboost_all_feature_values_wide": args.outdir / "xgboost_all_feature_values_wide.csv",
        "xgboost_all_feature_values_long": args.outdir / "xgboost_all_feature_values_long.csv",
        "xgboost_all_feature_target_correlations": args.outdir / "xgboost_all_feature_target_correlations.csv",
        "xgboost_all_feature_pearson_matrix": args.outdir / "xgboost_all_feature_pearson_matrix.csv",
        "xgboost_all_feature_spearman_matrix": args.outdir / "xgboost_all_feature_spearman_matrix.csv",
        "profile": args.outdir / "data_profile.json",
        "report": args.outdir / "report.md",
    }
    paths["correlation_matrix_dir"].mkdir(parents=True, exist_ok=True)
    metrics.to_csv(paths["metrics"], index=False, encoding="utf-8-sig")
    predictions.to_csv(paths["predictions"], index=False, encoding="utf-8-sig")
    train_predictions.to_csv(paths["train_predictions"], index=False, encoding="utf-8-sig")
    importance.to_csv(paths["importance_fold"], index=False, encoding="utf-8-sig")
    importance_summary.to_csv(paths["importance_summary"], index=False, encoding="utf-8-sig")
    shap_values.to_csv(paths["shap_fold"], index=False, encoding="utf-8-sig")
    shap_summary.to_csv(paths["shap_summary"], index=False, encoding="utf-8-sig")
    shap_stability.to_csv(paths["shap_stability"], index=False, encoding="utf-8-sig")
    shap_rank_pairs.to_csv(paths["shap_rank_stability_pairs"], index=False, encoding="utf-8-sig")
    shap_rank_summary.to_csv(paths["shap_rank_stability_summary"], index=False, encoding="utf-8-sig")
    correlation_summary.to_csv(paths["correlation_summary"], index=False, encoding="utf-8-sig")
    correlation_pairs.to_csv(paths["correlation_pairs"], index=False, encoding="utf-8-sig")
    xgb_all_feature_values_wide.to_csv(
        paths["xgboost_all_feature_values_wide"],
        index=False,
        encoding="utf-8-sig",
    )
    xgb_all_feature_values_long.to_csv(
        paths["xgboost_all_feature_values_long"],
        index=False,
        encoding="utf-8-sig",
    )
    xgb_all_feature_target_correlations.to_csv(
        paths["xgboost_all_feature_target_correlations"],
        index=False,
        encoding="utf-8-sig",
    )
    xgb_all_feature_pearson_matrix.to_csv(
        paths["xgboost_all_feature_pearson_matrix"],
        encoding="utf-8-sig",
    )
    xgb_all_feature_spearman_matrix.to_csv(
        paths["xgboost_all_feature_spearman_matrix"],
        encoding="utf-8-sig",
    )
    correlation_matrix_files = []
    for feature_set, matrix in correlation_matrices.items():
        matrix_path = paths["correlation_matrix_dir"] / f"{safe_filename(feature_set)}_pearson.csv"
        matrix.to_csv(matrix_path, encoding="utf-8-sig")
        correlation_matrix_files.append(str(matrix_path))

    profile = {
        "data": str(args.data),
        "model_cv_dir": str(args.model_cv_dir),
        "model": model_name,
        "feature_sets": feature_set_names,
        "feature_set_sizes": {name: int(feature_sets[name].shape[1]) for name in feature_set_names},
        "best_params_by_feature_set": params_by_feature_set,
        "repeats": args.repeats,
        "folds": args.folds,
        "seed": args.seed,
        "fold_metrics_include_train_metrics": True,
        "train_predictions_file": str(paths["train_predictions"]),
        "shap_method": "xgboost pred_contribs" if model_name == "XGBoost" else "requires shap package",
        "shap_stability_method": "fold-level mean_abs_shap rank and magnitude stability across repeated CV",
        "feature_correlation_method": "Pearson correlation after the same imputation, variance filtering, and SelectKBest feature selection used by the model",
        "correlation_threshold": args.correlation_threshold,
        "correlation_matrix_files": correlation_matrix_files,
        "xgboost_all_feature_target_correlation_method": (
            "XGBoost with provided_plus_rdkit_descriptors fitted on the full training set; "
            "feature values are the raw provided/RDKit input features before pipeline transforms; "
            "used_by_xgboost_after_pipeline marks features retained after the model pipeline imputer, "
            "variance filter, and SelectKBest; Pearson and Spearman correlations are computed against "
            "Luciferase expression (RLU). Full Pearson and Spearman two-dimensional matrices include "
            "Luciferase expression (RLU) plus all input features."
            if not xgb_all_feature_target_correlations.empty
            else "not generated because the selected model is not XGBoost or all-feature XGBoost params are unavailable"
        ),
    }
    paths["profile"].write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")

    top_importance = (
        importance_summary.groupby("feature_set", group_keys=False)
        .head(20)
        .reset_index(drop=True)
    )
    if shap_summary.empty:
        top_shap = pd.DataFrame()
        top_shap_stability = pd.DataFrame()
    else:
        top_shap = shap_summary.groupby("feature_set", group_keys=False).head(20).reset_index(drop=True)
        top_shap_stability = (
            shap_stability.groupby("feature_set", group_keys=False)
            .head(20)
            .reset_index(drop=True)
        )
    top_correlation_pairs = (
        correlation_pairs.groupby("feature_set", group_keys=False)
        .head(20)
        .reset_index(drop=True)
        if not correlation_pairs.empty
        else pd.DataFrame()
    )
    top_xgb_target_correlations = (
        xgb_all_feature_target_correlations.head(30).reset_index(drop=True)
        if not xgb_all_feature_target_correlations.empty
        else pd.DataFrame()
    )

    report = [
        "# SHAP and feature importance under repeated CV",
        "",
        f"- Data: `{args.data}`",
        f"- Best model type from model comparison: `{model_name}`",
        f"- CV: {args.repeats} x {args.folds}-fold",
        f"- Feature sets: {', '.join(feature_set_names)}",
        f"- SHAP method: {'XGBoost pred_contribs' if model_name == 'XGBoost' else 'not available without shap package'}",
        "",
        "## Top Feature Importance",
        "",
        dataframe_to_markdown(
            top_importance[["feature_set", "feature", "mean_importance", "std_importance", "n_folds"]]
        ),
        "",
    ]
    if not top_shap.empty:
        report.extend(
            [
                "## Top SHAP",
                "",
                dataframe_to_markdown(top_shap[["feature_set", "feature", "mean_abs_shap", "mean_shap"]]),
                "",
            ]
        )
    if not top_shap_stability.empty:
        report.extend(
            [
                "## SHAP Stability",
                "",
                dataframe_to_markdown(
                    top_shap_stability[
                        [
                            "feature_set",
                            "feature",
                            "mean_abs_shap",
                            "cv_abs_shap",
                            "mean_rank",
                            "std_rank",
                            "presence_fraction",
                            "top_10_frequency",
                            "sign_consistency",
                        ]
                    ]
                ),
                "",
            ]
        )
    if not shap_rank_summary.empty:
        report.extend(
            [
                "## SHAP Rank Stability By Feature Set",
                "",
                dataframe_to_markdown(
                    shap_rank_summary[
                        [
                            "feature_set",
                            "mean_spearman_rank_corr",
                            "std_spearman_rank_corr",
                            "mean_top10_jaccard",
                            "std_top10_jaccard",
                            "n_fold_pairs",
                        ]
                    ]
                ),
                "",
            ]
        )
    if not top_correlation_pairs.empty:
        report.extend(
            [
                "## High Feature Correlations",
                "",
                dataframe_to_markdown(
                    top_correlation_pairs[
                        [
                            "feature_set",
                            "feature_1",
                            "feature_2",
                            "pearson_r",
                            "abs_pearson_r",
                        ]
                    ]
                ),
                "",
            ]
        )
    if not top_xgb_target_correlations.empty:
        report.extend(
            [
                "## XGBoost All-Feature Values vs Target Correlations",
                "",
                f"- Full Pearson matrix: `{paths['xgboost_all_feature_pearson_matrix']}`",
                f"- Full Spearman matrix: `{paths['xgboost_all_feature_spearman_matrix']}`",
                "",
                dataframe_to_markdown(
                    top_xgb_target_correlations[
                        [
                            "feature",
                            "used_by_xgboost_after_pipeline",
                            "pearson_r_with_target",
                            "spearman_r_with_target",
                            "abs_pearson_r_with_target",
                        ]
                    ]
                ),
                "",
            ]
        )
    report.extend(["## Output Files", ""])
    report.extend(f"- `{path}`: {name}" for name, path in paths.items() if name != "report")
    report.append("")
    paths["report"].write_text("\n".join(report), encoding="utf-8")

    print(top_importance[["feature_set", "feature", "mean_importance", "std_importance"]].to_string(index=False))
    if not top_shap.empty:
        print("\nTop SHAP:")
        print(top_shap[["feature_set", "feature", "mean_abs_shap", "mean_shap"]].to_string(index=False))
    if not shap_rank_summary.empty:
        print("\nSHAP rank stability:")
        print(
            shap_rank_summary[
                [
                    "feature_set",
                    "mean_spearman_rank_corr",
                    "mean_top10_jaccard",
                    "n_fold_pairs",
                ]
            ].to_string(index=False)
        )
    if not top_correlation_pairs.empty:
        print("\nHigh feature correlations:")
        print(
            top_correlation_pairs[
                ["feature_set", "feature_1", "feature_2", "pearson_r", "abs_pearson_r"]
            ]
            .head(30)
            .to_string(index=False)
        )
    if not top_xgb_target_correlations.empty:
        print("\nXGBoost all-feature target correlations:")
        print(
            top_xgb_target_correlations[
                [
                    "feature",
                    "used_by_xgboost_after_pipeline",
                    "pearson_r_with_target",
                    "spearman_r_with_target",
                    "abs_pearson_r_with_target",
                ]
            ].to_string(index=False)
        )
    print(f"\nWrote outputs to: {args.outdir}")


if __name__ == "__main__":
    main()
