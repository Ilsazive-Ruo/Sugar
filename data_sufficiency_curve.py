from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from ml_common import (
    NAME_COL,
    TARGET_COL,
    build_feature_sets,
    dataframe_to_markdown,
    json_dumps,
    make_estimator,
    read_training_data,
    regression_metrics,
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def subset_sizes(n_samples: int, start: int, step: int, max_size: int | None) -> list[int]:
    if start > n_samples:
        raise ValueError(f"start_size={start} exceeds n_samples={n_samples}.")
    end = n_samples if max_size is None else min(max_size, n_samples)
    if end < start:
        raise ValueError(f"max_size={max_size} is smaller than start_size={start}.")
    sizes = list(range(start, end + 1, step))
    if sizes[-1] != end:
        sizes.append(end)
    return sizes


def load_selected_scenario(model_cv_dir: Path) -> tuple[str, str, dict[str, Any], dict[str, Any]]:
    selected_path = model_cv_dir / "selected_model.json"
    if selected_path.exists():
        payload = json.loads(selected_path.read_text(encoding="utf-8"))
        model_name = str(payload["selected_model"])
        feature_set = str(payload["selected_feature_set"])
        params = payload["params_by_feature_set"][feature_set]
        return model_name, feature_set, params, payload

    summary = pd.read_csv(model_cv_dir / "cv_summary.csv")
    best = summary.sort_values(["mean_fold_rmse", "mean_fold_mae"]).iloc[0]
    model_name = str(best["model"])
    feature_set = str(best["feature_set"])
    params = json.loads(str(best["consensus_best_params"]))
    return model_name, feature_set, params, {"selected_scenario": str(best["scenario_run"])}


def prefix_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def load_model_cv_module() -> Any:
    model_cv_path = Path(__file__).with_name("model_feature_nested_cv.py")
    spec = importlib.util.spec_from_file_location("model_feature_nested_cv", model_cv_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load model comparison module from {model_cv_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def seed_for_subset_repeat(
    n_subset: int,
    n_samples: int,
    subset_repeat: int,
    seed: int,
) -> int:
    if n_subset == n_samples:
        return seed + subset_repeat - 1
    return seed + n_subset * 100_000 + subset_repeat * 1_000


def remap_prediction_indices(frame: pd.DataFrame, subset_idx: np.ndarray) -> pd.DataFrame:
    frame = frame.copy()
    subset_positions = frame["row_index"].to_numpy(dtype=int)
    frame["subset_row_index"] = subset_positions
    frame["row_index"] = subset_idx[subset_positions]
    return frame


def run_fixed_param_curve(
    df: pd.DataFrame,
    x: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    feature_set_name: str,
    best_params: dict[str, Any],
    sizes: list[int],
    subset_repeats: int,
    folds: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_indices = np.arange(len(df))
    metric_rows = []
    pred_frames = []
    membership_frames = []
    scenario_run = f"{model_name}__{feature_set_name}"

    for n_subset in sizes:
        print(f"Running subset size {n_subset}...")
        for subset_repeat in range(1, subset_repeats + 1):
            subset_seed = seed_for_subset_repeat(n_subset, len(df), subset_repeat, seed)
            rng = np.random.default_rng(subset_seed)
            subset_idx = np.sort(rng.choice(all_indices, size=n_subset, replace=False))
            membership = pd.DataFrame(
                {
                    "n_subset": n_subset,
                    "subset_repeat": subset_repeat,
                    "subset_seed": subset_seed,
                    "row_index": subset_idx,
                }
            )
            if NAME_COL in df.columns:
                membership["sugar_name"] = df.iloc[subset_idx][NAME_COL].to_numpy()
            membership_frames.append(membership)

            x_subset = x.iloc[subset_idx].reset_index(drop=True)
            y_subset = y.iloc[subset_idx].reset_index(drop=True)
            original_idx = pd.Series(subset_idx)
            cv = KFold(n_splits=folds, shuffle=True, random_state=subset_seed)
            for fold_idx, (train_pos, val_pos) in enumerate(cv.split(x_subset), start=1):
                estimator = make_estimator(model_name, subset_seed + fold_idx)
                estimator.set_params(**best_params)
                x_train = x_subset.iloc[train_pos]
                x_val = x_subset.iloc[val_pos]
                y_train = y_subset.iloc[train_pos].to_numpy(dtype=float)
                y_val = y_subset.iloc[val_pos].to_numpy(dtype=float)
                estimator.fit(x_train, y_train)
                y_pred = np.asarray(estimator.predict(x_val), dtype=float)
                y_train_pred = np.asarray(estimator.predict(x_train), dtype=float)

                metrics = regression_metrics(y_val, y_pred)
                metrics.update(prefix_metrics(regression_metrics(y_train, y_train_pred), "train"))
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
                        "n_subset": n_subset,
                        "subset_repeat": subset_repeat,
                        "subset_seed": subset_seed,
                        "fold": fold_idx,
                        "n_train": int(len(train_pos)),
                        "n_val": int(len(val_pos)),
                        "n_features_input": int(x.shape[1]),
                        "best_params": json_dumps(best_params),
                    }
                )
                metric_rows.append(metrics)

                val_original_idx = original_idx.iloc[val_pos].to_numpy()
                pred_frames.append(
                    pd.DataFrame(
                        {
                            "scenario_run": scenario_run,
                            "model": model_name,
                            "feature_set": feature_set_name,
                            "n_subset": n_subset,
                            "subset_repeat": subset_repeat,
                            "subset_seed": subset_seed,
                            "fold": fold_idx,
                            "row_index": val_original_idx,
                            "y_true": y_val,
                            "y_pred": y_pred,
                            "abs_error": np.abs(y_pred - y_val),
                            "squared_error": np.square(y_pred - y_val),
                        }
                    )
                )

    return (
        pd.DataFrame(metric_rows),
        pd.concat(pred_frames, ignore_index=True),
        pd.concat(membership_frames, ignore_index=True),
    )


def run_nested_1se_curve(
    df: pd.DataFrame,
    x: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    feature_set_name: str,
    sizes: list[int],
    subset_repeats: int,
    folds: int,
    inner_folds: int,
    search_iter: int,
    n_jobs: int,
    seed: int,
    param_profile: str,
    feature_selection: str,
    one_se_multiplier: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_cv = load_model_cv_module()
    all_indices = np.arange(len(df))
    metric_frames = []
    pred_frames = []
    membership_frames = []

    for n_subset in sizes:
        print(f"Running subset size {n_subset} with nested 1-SE tuning...")
        for subset_repeat in range(1, subset_repeats + 1):
            subset_seed = seed_for_subset_repeat(n_subset, len(df), subset_repeat, seed)
            rng = np.random.default_rng(subset_seed)
            subset_idx = np.sort(rng.choice(all_indices, size=n_subset, replace=False))
            membership = pd.DataFrame(
                {
                    "n_subset": n_subset,
                    "subset_repeat": subset_repeat,
                    "subset_seed": subset_seed,
                    "row_index": subset_idx,
                }
            )
            if NAME_COL in df.columns:
                membership["sugar_name"] = df.iloc[subset_idx][NAME_COL].to_numpy()
            membership_frames.append(membership)

            fold_metrics, predictions, _train_predictions = model_cv.run_nested_cv(
                model_name=model_name,
                feature_set_name=feature_set_name,
                x=x.iloc[subset_idx].reset_index(drop=True),
                y=y.iloc[subset_idx].reset_index(drop=True),
                repeat_seeds=[subset_seed],
                outer_folds=folds,
                inner_folds=inner_folds,
                search_iter=search_iter,
                n_jobs=n_jobs,
                param_profile=param_profile,
                selection_rule="one_standard_error",
                feature_selection=feature_selection,
                one_se_multiplier=one_se_multiplier,
            )
            fold_metrics = fold_metrics.copy()
            fold_metrics["n_subset"] = n_subset
            fold_metrics["subset_repeat"] = subset_repeat
            fold_metrics["subset_seed"] = subset_seed
            fold_metrics["repeat"] = subset_repeat
            fold_metrics["repeat_seed"] = subset_seed
            metric_frames.append(fold_metrics)

            predictions = remap_prediction_indices(predictions, subset_idx)
            predictions["n_subset"] = n_subset
            predictions["subset_repeat"] = subset_repeat
            predictions["subset_seed"] = subset_seed
            predictions["repeat"] = subset_repeat
            predictions["repeat_seed"] = subset_seed
            pred_frames.append(predictions)

    return (
        pd.concat(metric_frames, ignore_index=True),
        pd.concat(pred_frames, ignore_index=True),
        pd.concat(membership_frames, ignore_index=True),
    )


def summarize(metrics: pd.DataFrame, predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        if col in metrics.columns
    ]
    repeat_summary = (
        metrics.groupby(["scenario_run", "n_subset", "subset_repeat"])[
            metric_cols + ["n_train", "n_val"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    repeat_summary.columns = [
        "_".join(str(part) for part in col if part) if isinstance(col, tuple) else col
        for col in repeat_summary.columns
    ]

    curve = (
        metrics.groupby(["scenario_run", "model", "feature_set", "n_subset"])[
            metric_cols + ["n_train", "n_val"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    curve.columns = [
        "_".join(str(part) for part in col if part) if isinstance(col, tuple) else col
        for col in curve.columns
    ]
    curve = curve.rename(
        columns={
            "rmse_mean": "mean_fold_rmse",
            "rmse_std": "std_fold_rmse",
            "mae_mean": "mean_fold_mae",
            "mae_std": "std_fold_mae",
            "r2_mean": "mean_fold_r2",
            "r2_std": "std_fold_r2",
            "train_rmse_mean": "mean_train_rmse",
            "train_r2_mean": "mean_train_r2",
            "train_val_rmse_gap_mean": "mean_train_val_rmse_gap",
            "train_val_rmse_ratio_mean": "mean_train_val_rmse_ratio",
            "rmse_ratio_mean": "mean_rmse_ratio",
            "train_val_r2_gap_mean": "mean_train_val_r2_gap",
            "n_train_mean": "mean_n_train",
            "n_val_mean": "mean_n_val",
        }
    )

    pooled_rows = []
    for (scenario_run, model_name, feature_set, n_subset), group in predictions.groupby(
        ["scenario_run", "model", "feature_set", "n_subset"]
    ):
        pooled = regression_metrics(group["y_true"].to_numpy(), group["y_pred"].to_numpy())
        pooled_rows.append(
            {
                "scenario_run": scenario_run,
                "model": model_name,
                "feature_set": feature_set,
                "n_subset": n_subset,
                "pooled_cv_rmse": pooled["rmse"],
                "pooled_cv_mae": pooled["mae"],
                "pooled_cv_r2": pooled["r2"],
                "pooled_cv_pearson": pooled["pearson"],
                "pooled_cv_spearman": pooled["spearman"],
                "n_predictions": int(len(group)),
            }
        )
    curve = curve.merge(
        pd.DataFrame(pooled_rows),
        on=["scenario_run", "model", "feature_set", "n_subset"],
        how="left",
    ).sort_values("n_subset")

    full_rmse = float(curve.iloc[-1]["pooled_cv_rmse"])
    curve["pooled_cv_rmse_gap_vs_full"] = curve["pooled_cv_rmse"] - full_rmse
    curve["pooled_cv_rmse_pct_gap_vs_full"] = curve["pooled_cv_rmse_gap_vs_full"] / full_rmse * 100
    curve["pooled_cv_rmse_improvement_vs_previous"] = -curve["pooled_cv_rmse"].diff()
    curve["pooled_cv_r2_change_vs_previous"] = curve["pooled_cv_r2"].diff()
    return curve.reset_index(drop=True), repeat_summary.reset_index(drop=True)


def write_repeat_wide_tables(repeat_summary: pd.DataFrame, outdir: Path) -> list[str]:
    outdir.mkdir(parents=True, exist_ok=True)
    metric_cols = [
        "rmse_mean",
        "mae_mean",
        "r2_mean",
        "train_rmse_mean",
        "train_r2_mean",
        "train_val_rmse_gap_mean",
        "train_val_rmse_ratio_mean",
        "rmse_ratio_mean",
        "train_val_r2_gap_mean",
        "inner_selected_rmse_mean",
        "inner_selected_rmse_gap_vs_min_mean",
        "one_se_candidate_count_mean",
    ]
    filename_map = {
        "rmse_mean": "rmse",
        "mae_mean": "mae",
        "r2_mean": "r2",
    }
    written = []
    repeats = sorted(int(value) for value in repeat_summary["subset_repeat"].dropna().unique())
    for metric in [col for col in metric_cols if col in repeat_summary.columns]:
        rows = []
        for n_subset, group in repeat_summary.groupby("n_subset"):
            row: dict[str, object] = {"n_subset": int(n_subset)}
            for repeat in repeats:
                match = group.loc[group["subset_repeat"] == repeat]
                row[f"repeat_{repeat:02d}"] = np.nan if match.empty else float(match.iloc[0][metric])
            rows.append(row)
        path = outdir / f"{filename_map.get(metric, metric.replace('_mean', ''))}.csv"
        pd.DataFrame(rows).sort_values("n_subset").to_csv(path, index=False, encoding="utf-8-sig")
        written.append(str(path))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Data sufficiency curve using random row subsets."
        )
    )
    parser.add_argument("--data", default="dataset.csv", type=Path)
    parser.add_argument("--model-cv-dir", default="results/model_feature_nested_cv", type=Path)
    parser.add_argument("--outdir", default="results/data_sufficiency_curve", type=Path)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--start-size", default=30, type=positive_int)
    parser.add_argument("--max-size", default=None, type=positive_int)
    parser.add_argument("--step-size", default=5, type=positive_int)
    parser.add_argument("--subset-repeats", default=10, type=positive_int)
    parser.add_argument("--folds", default=5, type=positive_int)
    parser.add_argument(
        "--evaluation-mode",
        default="nested_1se",
        choices=["nested_1se", "fixed_params"],
        help="nested_1se repeats the model-comparison inner tuning rule in each fold; fixed_params is faster.",
    )
    parser.add_argument("--inner-folds", default=3, type=positive_int)
    parser.add_argument("--search-iter", default=12, type=positive_int)
    parser.add_argument("--n-jobs", default=1, type=int)
    parser.add_argument(
        "--param-profile",
        default="anti_overfit",
        choices=["anti_overfit", "baseline"],
    )
    parser.add_argument(
        "--feature-selection",
        default="auto",
        choices=["auto", "off"],
    )
    parser.add_argument("--one-se-multiplier", default=1.0, type=float)
    args = parser.parse_args()

    if args.folds > args.start_size:
        raise ValueError("--folds cannot exceed --start-size.")
    if args.inner_folds > args.start_size:
        raise ValueError("--inner-folds cannot exceed --start-size.")
    if args.one_se_multiplier < 0:
        raise ValueError("--one-se-multiplier must be non-negative.")

    args.outdir.mkdir(parents=True, exist_ok=True)
    model_name, feature_set_name, best_params, selected_payload = load_selected_scenario(args.model_cv_dir)

    df = read_training_data(args.data)
    y = df[TARGET_COL]
    feature_sets = build_feature_sets(df)
    sizes = subset_sizes(len(df), args.start_size, args.step_size, args.max_size)

    if args.evaluation_mode == "nested_1se":
        metrics, predictions, membership = run_nested_1se_curve(
            df=df,
            x=feature_sets[feature_set_name],
            y=y,
            model_name=model_name,
            feature_set_name=feature_set_name,
            sizes=sizes,
            subset_repeats=args.subset_repeats,
            folds=args.folds,
            inner_folds=args.inner_folds,
            search_iter=args.search_iter,
            n_jobs=args.n_jobs,
            seed=args.seed,
            param_profile=args.param_profile,
            feature_selection=args.feature_selection,
            one_se_multiplier=args.one_se_multiplier,
        )
    else:
        metrics, predictions, membership = run_fixed_param_curve(
            df=df,
            x=feature_sets[feature_set_name],
            y=y,
            model_name=model_name,
            feature_set_name=feature_set_name,
            best_params=best_params,
            sizes=sizes,
            subset_repeats=args.subset_repeats,
            folds=args.folds,
            seed=args.seed,
        )
    curve, repeat_summary = summarize(metrics, predictions)

    paths = {
        "curve": args.outdir / "learning_curve_summary.csv",
        "repeat_summary": args.outdir / "subset_repeat_summary.csv",
        "metrics": args.outdir / "fold_metrics.csv",
        "predictions": args.outdir / "cv_predictions.csv",
        "membership": args.outdir / "subset_membership.csv",
        "metric_wide_dir": args.outdir / "metric_wide" / "repeat_mean",
        "profile": args.outdir / "data_profile.json",
        "report": args.outdir / "report.md",
    }
    curve.to_csv(paths["curve"], index=False, encoding="utf-8-sig")
    repeat_summary.to_csv(paths["repeat_summary"], index=False, encoding="utf-8-sig")
    metrics.to_csv(paths["metrics"], index=False, encoding="utf-8-sig")
    predictions.to_csv(paths["predictions"], index=False, encoding="utf-8-sig")
    membership.to_csv(paths["membership"], index=False, encoding="utf-8-sig")
    metric_wide_files = write_repeat_wide_tables(repeat_summary, paths["metric_wide_dir"])

    profile = {
        "data": str(args.data),
        "model_cv_dir": str(args.model_cv_dir),
        "selected_model": model_name,
        "selected_feature_set": feature_set_name,
        "selected_scenario": selected_payload.get("selected_scenario"),
        "feature_set_size": int(feature_sets[feature_set_name].shape[1]),
        "model_cv_consensus_best_params": best_params,
        "method": (
            "random row subsets; nested row-level KFold CV with per-fold inner 1-SE tuning"
            if args.evaluation_mode == "nested_1se"
            else "random row subsets; row-level KFold CV; fixed model-comparison 1-SE consensus hyperparameters"
        ),
        "evaluation_mode": args.evaluation_mode,
        "n_total_samples": int(len(df)),
        "subset_sizes": sizes,
        "subset_repeats": args.subset_repeats,
        "folds": args.folds,
        "inner_folds": args.inner_folds if args.evaluation_mode == "nested_1se" else None,
        "search_iter": args.search_iter if args.evaluation_mode == "nested_1se" else None,
        "param_profile": args.param_profile if args.evaluation_mode == "nested_1se" else None,
        "feature_selection": args.feature_selection if args.evaluation_mode == "nested_1se" else None,
        "one_se_multiplier": args.one_se_multiplier if args.evaluation_mode == "nested_1se" else None,
        "seed": args.seed,
        "metric_wide_files": metric_wide_files,
    }
    paths["profile"].write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")

    report_cols = [
        "n_subset",
        "mean_n_train",
        "mean_n_val",
        "mean_fold_rmse",
        "std_fold_rmse",
        "pooled_cv_rmse",
        "mean_fold_mae",
        "pooled_cv_mae",
        "mean_fold_r2",
        "pooled_cv_r2",
        "mean_train_rmse",
        "mean_train_r2",
        "mean_rmse_ratio",
        "mean_train_val_r2_gap",
        "pooled_cv_rmse_pct_gap_vs_full",
        "pooled_cv_rmse_improvement_vs_previous",
        "n_predictions",
    ]
    report = [
        "# Data sufficiency learning curve",
        "",
        f"- Data: `{args.data}`",
        f"- Selected scenario from model comparison: `{selected_payload.get('selected_scenario')}`",
        f"- Model: `{model_name}`",
        f"- Feature set: `{feature_set_name}`",
        f"- Evaluation mode: `{args.evaluation_mode}`",
        f"- Random subset sizes: {sizes[0]} to {sizes[-1]}, step={args.step_size} samples",
        f"- Subset repeats per size: {args.subset_repeats}",
        f"- CV: {args.folds}-fold row-level KFold",
        "- Subsets are random rows, not sugar-level groups.",
        (
            f"- Hyperparameters: inner {args.inner_folds}-fold tuning with model-comparison 1-SE logic, "
            f"n_iter={args.search_iter}."
            if args.evaluation_mode == "nested_1se"
            else "- Hyperparameters: fixed to model-comparison consensus best params."
        ),
        f"- Metric wide tables: `{paths['metric_wide_dir']}`",
        "",
        "## Learning Curve Summary",
        "",
        dataframe_to_markdown(curve[report_cols]),
        "",
        "## Output Files",
        "",
    ]
    report.extend(f"- `{path}`: {name}" for name, path in paths.items() if name != "report")
    report.append("")
    paths["report"].write_text("\n".join(report), encoding="utf-8")

    print(curve[report_cols].to_string(index=False))
    print(f"\nWrote outputs to: {args.outdir}")


if __name__ == "__main__":
    main()
