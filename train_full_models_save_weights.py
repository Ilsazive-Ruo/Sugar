from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from ml_common import (
    TARGET_COL,
    build_feature_sets,
    make_estimator,
    read_training_data,
)


def load_params_for_model(best_params_path: Path, model_name: str) -> dict[str, dict[str, Any]]:
    from ml_common import FEATURE_SET_NAMES

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


def load_model_cv_selection(model_cv_dir: Path) -> tuple[str, str, dict[str, Any], dict[str, Any]]:
    selected_path = model_cv_dir / "selected_model.json"
    if selected_path.exists():
        payload = json.loads(selected_path.read_text(encoding="utf-8"))
        model_name = str(payload["selected_model"])
        feature_set = str(payload["selected_feature_set"])
        params = dict(payload["params_by_feature_set"][feature_set])
        return model_name, feature_set, params, payload

    summary = pd.read_csv(model_cv_dir / "cv_summary.csv")
    best = summary.sort_values(["mean_fold_rmse", "mean_fold_mae"]).iloc[0]
    model_name = str(best["model"])
    feature_set = str(best["feature_set"])
    if "consensus_best_params" in best.index:
        params = json.loads(str(best["consensus_best_params"]))
    else:
        params = load_params_for_model(model_cv_dir / "best_hyperparameters.csv", model_name)[
            feature_set
        ]
    return model_name, feature_set, params, {"selected_scenario": str(best["scenario_run"])}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train final models and save model packages."
        )
    )
    parser.add_argument("--data", default="dataset.csv", type=Path)
    parser.add_argument("--model-cv-dir", default="results/model_feature_nested_cv", type=Path)
    parser.add_argument("--outdir", default="results/full_models", type=Path)
    parser.add_argument("--model-dir", default="models", type=Path)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    model_name, feature_set, params, selection_payload = load_model_cv_selection(args.model_cv_dir)

    df = read_training_data(args.data)
    y = df[TARGET_COL]
    feature_sets = build_feature_sets(df)

    if feature_set not in feature_sets:
        raise ValueError(f"Selected feature set is unavailable: {feature_set}")

    x = feature_sets[feature_set]
    estimator = make_estimator(model_name, args.seed)
    estimator.set_params(**params)
    estimator.fit(x, y.to_numpy(dtype=float))

    model_path = args.model_dir / f"{model_name}__{feature_set}.joblib"
    package = {
        "pipeline": estimator,
        "model": model_name,
        "feature_set": feature_set,
        "input_feature_columns": list(x.columns),
        "target": TARGET_COL,
        "best_params": params,
        "training_data": str(args.data),
        "n_training_samples": int(len(df)),
        "selection": selection_payload,
    }
    joblib.dump(package, model_path)

    metadata_row = {
        "model": model_name,
        "feature_set": feature_set,
        "model_path": str(model_path),
        "n_training_samples": int(len(df)),
        "n_features_input": int(x.shape[1]),
        "best_params": json.dumps(params, ensure_ascii=False, sort_keys=True),
    }
    metadata = pd.DataFrame([metadata_row])

    paths = {
        "metadata": args.outdir / "full_model_metadata.csv",
        "profile": args.outdir / "data_profile.json",
    }
    metadata.to_csv(paths["metadata"], index=False, encoding="utf-8-sig")
    profile = {
        "data": str(args.data),
        "model_cv_dir": str(args.model_cv_dir),
        "model_dir": str(args.model_dir),
        "model": model_name,
        "feature_set": feature_set,
        "n_samples": int(len(df)),
        "model_path": str(model_path),
        "n_features_input": int(x.shape[1]),
        "best_params": params,
        "selection": selection_payload,
    }
    paths["profile"].write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        metadata[
            ["model", "feature_set", "model_path", "n_training_samples", "n_features_input"]
        ].to_string(index=False)
    )
    print(f"\nWrote models to: {args.model_dir}")
    print(f"Wrote outputs to: {args.outdir}")


if __name__ == "__main__":
    main()
