from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    explained_variance_score,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


TARGET_COL = "Luciferase expression (RLU)"
SMILES_COL = "SMILES"
ID_COL = "ID"
NAME_COL = "Sugar name"
PROVIDED_FEATURES = [
    "Molecular weight (Da)",
    "Solubility (mg/mL)",
    "Glass transition temperature (Tg, degC)",
    "Viscosity (mPa.s)",
    "Concentration (% w/v)",
]
FEATURE_SET_NAMES = [
    "provided_only",
    "provided_plus_rdkit_descriptors",
    "rdkit_descriptors_only",
]
MODEL_NAMES = ["XGBoost", "GBoost", "RandomForest", "DecisionTree", "KNN"]

LEGACY_COLUMN_MAP = {
    "Name": NAME_COL,
    "Sugar": NAME_COL,
    "Molecular weight": "Molecular weight (Da)",
    "Solubility": "Solubility (mg/mL)",
    "Tg": "Glass transition temperature (Tg, degC)",
    "Viscosity": "Viscosity (mPa.s)",
    "Concentration": "Concentration (% w/v)",
    "Transfection": TARGET_COL,
}

SMILES_FIXES: list[tuple[str, str]] = [
    ("[2 H2O]", "O.O"),
    ("[2H2O]", "O.O"),
    ("-;@", ""),
]


def normalize_column_name(name: object) -> str:
    cleaned = (
        str(name)
        .replace("\n", " ")
        .replace("\r", " ")
        .replace("°C", "degC")
        .replace("℃", "degC")
        .replace("掳C", "degC")
        .replace("·", ".")
        .replace("路", ".")
        .strip()
    )
    return re.sub(r"\s+", " ", cleaned)


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported data file type: {path}")


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_column_name(col) for col in df.columns]
    df = df.rename(columns={k: v for k, v in LEGACY_COLUMN_MAP.items() if k in df.columns})
    df = df.drop(columns=[col for col in df.columns if col.startswith("Unnamed")], errors="ignore")
    return df


def read_training_data(path: Path) -> pd.DataFrame:
    df = canonicalize_columns(read_table(path))
    required = PROVIDED_FEATURES + [TARGET_COL, SMILES_COL]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required training columns in {path}: {missing}")

    df = df.dropna(subset=[SMILES_COL, TARGET_COL]).reset_index(drop=True)
    for col in PROVIDED_FEATURES + [TARGET_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    return df


def read_prediction_data(path: Path) -> pd.DataFrame:
    df = canonicalize_columns(read_table(path))
    required = PROVIDED_FEATURES + [SMILES_COL]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required prediction columns in {path}: {missing}")

    if NAME_COL not in df.columns:
        df[NAME_COL] = ""
    df = df.dropna(subset=[SMILES_COL]).reset_index(drop=True)
    for col in PROVIDED_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["prediction_row"] = np.arange(1, len(df) + 1)
    return df


def clean_smiles(smiles: object) -> tuple[str, bool]:
    cleaned = re.sub(r"\s+", "", str(smiles))
    original = cleaned
    for old, new in SMILES_FIXES:
        cleaned = cleaned.replace(old, new)
    return cleaned, cleaned != original


def smiles_to_mols(smiles: pd.Series) -> list[Chem.Mol]:
    mols = []
    bad = []
    for idx, smi in smiles.items():
        cleaned, _ = clean_smiles(smi)
        mol = Chem.MolFromSmiles(cleaned)
        if mol is None:
            bad.append((idx, smi))
        mols.append(mol)
    if bad:
        examples = ", ".join(f"{idx}:{smi}" for idx, smi in bad[:5])
        raise ValueError(f"Invalid SMILES found, examples: {examples}")
    return mols


def smiles_cleaning_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for idx, smi in df[SMILES_COL].items():
        cleaned, changed = clean_smiles(smi)
        if changed:
            rows.append(
                {
                    "row_index": idx,
                    "ID": df.loc[idx, ID_COL] if ID_COL in df.columns else np.nan,
                    "name": df.loc[idx, NAME_COL] if NAME_COL in df.columns else "",
                    "original_smiles": smi,
                    "cleaned_smiles": cleaned,
                }
            )
    return pd.DataFrame(rows)


def make_rdkit_descriptors(mols: list[Chem.Mol]) -> pd.DataFrame:
    rows = []
    names = [name for name, _ in Descriptors.descList]
    for mol in mols:
        values = []
        for _, func in Descriptors.descList:
            try:
                values.append(func(mol))
            except Exception:
                values.append(np.nan)
        rows.append(values)
    return pd.DataFrame(rows, columns=[f"rdkit_{name}" for name in names]).replace(
        [np.inf, -np.inf], np.nan
    )


def build_feature_sets(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    mols = smiles_to_mols(df[SMILES_COL])
    provided = df[PROVIDED_FEATURES].copy()
    rdkit_desc = make_rdkit_descriptors(mols)
    feature_sets = {
        "provided_only": provided,
        "provided_plus_rdkit_descriptors": pd.concat([provided, rdkit_desc], axis=1),
        "rdkit_descriptors_only": rdkit_desc,
    }
    cleaned = {}
    for name, x in feature_sets.items():
        x = x.replace([np.inf, -np.inf], np.nan)
        cleaned[name] = x.mask(x.abs() > 1e20, np.nan)
    return cleaned


def make_estimator(model_name: str, seed: int) -> Pipeline:
    common_steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold(threshold=0.0)),
        ("select", SelectKBest(score_func=f_regression, k="all")),
    ]
    if model_name == "KNN":
        return Pipeline(common_steps + [("scaler", StandardScaler()), ("model", KNeighborsRegressor())])
    if model_name == "DecisionTree":
        return Pipeline(common_steps + [("model", DecisionTreeRegressor(random_state=seed))])
    if model_name == "GBoost":
        return Pipeline(common_steps + [("model", GradientBoostingRegressor(random_state=seed))])
    if model_name == "RandomForest":
        return Pipeline(
            common_steps
            + [("model", RandomForestRegressor(random_state=seed, n_jobs=1))]
        )
    if model_name == "XGBoost":
        return Pipeline(
            common_steps
            + [
                (
                    "model",
                    XGBRegressor(
                        objective="reg:squarederror",
                        random_state=seed,
                        n_jobs=1,
                    ),
                )
            ]
        )
    raise ValueError(f"Unsupported model: {model_name}")


def json_dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def safe_corr(a: np.ndarray, b: np.ndarray, method: str) -> float:
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    if method == "pearson":
        return float(stats.pearsonr(a, b).statistic)
    if method == "spearman":
        return float(stats.spearmanr(a, b).statistic)
    raise ValueError(method)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred_nonnegative = np.clip(y_pred, 0, None)
    residuals = y_pred - y_true
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "median_ae": float(median_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "explained_variance": float(explained_variance_score(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "max_error": float(max_error(y_true, y_pred)),
        "mean_error": float(np.mean(residuals)),
        "std_error": float(np.std(residuals, ddof=1)) if len(residuals) > 1 else np.nan,
        "mean_absolute_log_error": float(
            mean_absolute_error(np.log1p(y_true), np.log1p(y_pred_nonnegative))
        ),
        "rmse_log1p": float(
            np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred_nonnegative)))
        ),
        "pearson": safe_corr(y_true, y_pred, "pearson"),
        "spearman": safe_corr(y_true, y_pred, "spearman"),
    }


def feature_names_after_variance(pipeline: Pipeline, input_columns: list[str]) -> list[str]:
    support = pipeline.named_steps["variance"].get_support()
    names = pd.Index(input_columns)[support]
    if "select" in pipeline.named_steps:
        select_support = pipeline.named_steps["select"].get_support()
        names = names[select_support]
    return list(names)


def transformed_after_variance(pipeline: Pipeline, x: pd.DataFrame) -> np.ndarray:
    values = pipeline.named_steps["imputer"].transform(x)
    values = pipeline.named_steps["variance"].transform(values)
    if "select" in pipeline.named_steps:
        values = pipeline.named_steps["select"].transform(values)
    return values


def align_features(x: pd.DataFrame, expected_columns: list[str]) -> pd.DataFrame:
    aligned = x.copy()
    for col in expected_columns:
        if col not in aligned.columns:
            aligned[col] = np.nan
    return aligned[expected_columns]


def format_report_value(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    return str(value)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(format_report_value(row[col]) for col in headers) + " |")
    return "\n".join(lines)
