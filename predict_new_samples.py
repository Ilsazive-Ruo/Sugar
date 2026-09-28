from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml_common import (
    NAME_COL,
    PROVIDED_FEATURES,
    SMILES_COL,
    align_features,
    build_feature_sets,
    dataframe_to_markdown,
    read_prediction_data,
    smiles_cleaning_report,
)


PREDICTION_COL = "predicted_Luciferase expression (RLU)"


def load_best_model_package(model_dir: Path, metadata_path: Path) -> dict[str, object]:
    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path)
        if len(metadata) != 1:
            raise ValueError(
                f"Expected exactly one final model in {metadata_path}, found {len(metadata)}."
            )
        model_path = Path(str(metadata.iloc[0]["model_path"]))
        if not model_path.is_absolute():
            model_path = metadata_path.parent.parent.parent / model_path
    else:
        paths = sorted(model_dir.glob("*.joblib"))
        if len(paths) != 1:
            raise FileNotFoundError(
                f"Could not find one final model. Expected {metadata_path}, or exactly one "
                f".joblib package in {model_dir}; found {len(paths)}."
            )
        model_path = paths[0]

    if not model_path.exists():
        raise FileNotFoundError(f"Final model package not found: {model_path}")

    package = joblib.load(model_path)
    package["model_path"] = str(model_path)
    return package


def predict(package: dict[str, object], df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_sets = build_feature_sets(df)
    base_cols = [
        "prediction_row",
        "No.",
        NAME_COL,
        *PROVIDED_FEATURES,
        SMILES_COL,
    ]
    base_cols = list(dict.fromkeys(col for col in base_cols if col in df.columns))
    feature_set = str(package["feature_set"])
    if feature_set not in feature_sets:
        raise ValueError(f"Model feature set is unavailable for prediction data: {feature_set}")

    x = align_features(
        feature_sets[feature_set],
        [str(col) for col in package["input_feature_columns"]],
    )
    estimator = package["pipeline"]
    y_pred = np.asarray(estimator.predict(x), dtype=float)
    predictions = df[base_cols].copy()
    predictions["model"] = str(package["model"])
    predictions["feature_set"] = feature_set
    predictions["model_path"] = str(package["model_path"])
    predictions[PREDICTION_COL] = y_pred
    predictions["predicted_target_nonnegative"] = np.clip(y_pred, 0, None)
    predictions["best_params"] = json.dumps(
        package["best_params"], ensure_ascii=False, sort_keys=True
    )

    ranked = predictions.sort_values(PREDICTION_COL, ascending=False).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    return predictions, ranked


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Predict new samples with saved model packages."
        )
    )
    parser.add_argument("--input", default="ToBeScreened.csv", type=Path)
    parser.add_argument("--model-dir", default="models", type=Path)
    parser.add_argument("--metadata", default="results/full_models/full_model_metadata.csv", type=Path)
    parser.add_argument("--outdir", default="results/predictions", type=Path)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    df = read_prediction_data(args.input)
    package = load_best_model_package(args.model_dir, args.metadata)
    predictions, ranked = predict(package, df)
    cleaning_report = smiles_cleaning_report(df)

    paths = {
        "predictions": args.outdir / "predictions.csv",
        "predictions_ranked": args.outdir / "predictions_ranked.csv",
        "cleaning": args.outdir / "smiles_cleaning_report.csv",
        "profile": args.outdir / "prediction_profile.json",
        "report": args.outdir / "report.md",
    }
    predictions.to_csv(paths["predictions"], index=False, encoding="utf-8-sig")
    ranked.to_csv(paths["predictions_ranked"], index=False, encoding="utf-8-sig")
    cleaning_report.to_csv(paths["cleaning"], index=False, encoding="utf-8-sig")

    profile = {
        "input": str(args.input),
        "model_dir": str(args.model_dir),
        "metadata": str(args.metadata),
        "n_input_rows": int(len(df)),
        "n_predictions": int(len(predictions)),
        "model": str(package["model"]),
        "feature_set": str(package["feature_set"]),
        "model_path": str(package["model_path"]),
    }
    paths["profile"].write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")

    top = ranked.loc[ranked["rank"] <= 10].reset_index(drop=True)
    report_cols = [
        "feature_set",
        "rank",
        NAME_COL,
        "Concentration (% w/v)",
        PREDICTION_COL,
    ]
    report = [
        "# New sample predictions",
        "",
        f"- Input: `{args.input}`",
        f"- Model directory: `{args.model_dir}`",
        f"- Input rows: {len(df)}",
        f"- Predictions: {len(predictions)}",
        "",
        "## Top Predictions By Feature Set",
        "",
        dataframe_to_markdown(top[report_cols]),
        "",
        "## Output Files",
        "",
    ]
    report.extend(f"- `{path}`: {name}" for name, path in paths.items() if name != "report")
    report.append("")
    paths["report"].write_text("\n".join(report), encoding="utf-8")

    print(top[report_cols].to_string(index=False))
    print(f"\nWrote outputs to: {args.outdir}")


if __name__ == "__main__":
    main()
