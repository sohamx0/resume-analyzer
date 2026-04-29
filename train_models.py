#!/usr/bin/env python3
"""Train regression and classification models for synthetic resume analysis."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, StandardScaler


BASE_NUMERIC_FEATURES = [
    "num_skills",
    "num_projects",
    "num_experiences",
    "num_courses",
    "num_languages",
    "word_count",
    "avg_sentence_length",
    "section_count",
    "multi_domain_flag",
]

EXTRA_NUMERIC_FEATURES = [
    "skill_density",
    "metric_count",
    "metric_ratio",
    "avg_bullet_length",
    "has_summary",
    "has_projects",
    "has_experience",
]

NUMERIC_FEATURES = BASE_NUMERIC_FEATURES + EXTRA_NUMERIC_FEATURES

TEXT_FEATURE = "resume_text"

TARGET_FINAL_SCORE = "final_score"
TARGET_QUALITY = "classification_label"
TARGET_DOMAIN = "domain_label"

REQUIRED_COLUMNS = BASE_NUMERIC_FEATURES + [
    TEXT_FEATURE,
    TARGET_FINAL_SCORE,
    TARGET_QUALITY,
    TARGET_DOMAIN,
]
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train resume scoring and classification models.")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/datasets/new/synthetic_resumes.csv",
        help="Path to synthetic_resumes.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/trained",
        help="Directory to save trained models and artifacts",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split ratio",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    # Keep only required columns to avoid accidental leakage from engineered labels.
    data = df[REQUIRED_COLUMNS].copy()

    data = _add_missing_features(data)

    # Handle text nulls safely before vectorization.
    data[TEXT_FEATURE] = data[TEXT_FEATURE].fillna("").astype(str)

    # Coerce numeric columns and keep NaNs for imputer handling.
    for col in NUMERIC_FEATURES + [TARGET_FINAL_SCORE]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    # Drop rows without mandatory supervised targets.
    data = data.dropna(subset=[TARGET_FINAL_SCORE, TARGET_QUALITY, TARGET_DOMAIN]).reset_index(drop=True)

    return data


def _metric_count(text: str) -> int:
    return len(re.findall(r"\b\d+(?:\.\d+)?%?\b", text))


def _avg_bullet_length(text: str) -> float:
    bullets = [
        re.sub(r"^[-*\u2022\s]+", "", line).strip()
        for line in text.splitlines()
        if re.match(r"^\s*[-*\u2022]", line)
    ]
    if not bullets:
        return 0.0
    lengths = [len(re.findall(r"\b\w+\b", bullet)) for bullet in bullets if bullet]
    return round(sum(lengths) / len(lengths), 2) if lengths else 0.0


def _add_missing_features(data: pd.DataFrame) -> pd.DataFrame:
    if not EXTRA_NUMERIC_FEATURES:
        return data

    text_series = data[TEXT_FEATURE].fillna("").astype(str)
    word_counts = data.get("word_count")
    if word_counts is None:
        word_counts = text_series.apply(lambda text: len(re.findall(r"\b\w+\b", text)))

    metric_counts = text_series.apply(_metric_count)
    metric_ratio = metric_counts / word_counts.replace(0, 1)
    avg_bullet_length = text_series.apply(_avg_bullet_length)

    if "skill_density" not in data.columns:
        data["skill_density"] = data["num_skills"] / word_counts.replace(0, 1)
    if "metric_count" not in data.columns:
        data["metric_count"] = metric_counts
    if "metric_ratio" not in data.columns:
        data["metric_ratio"] = metric_ratio
    if "avg_bullet_length" not in data.columns:
        data["avg_bullet_length"] = avg_bullet_length
    if "has_summary" not in data.columns:
        data["has_summary"] = text_series.str.contains(r"\bsummary\b", case=False, regex=True).astype(int)
    if "has_projects" not in data.columns:
        data["has_projects"] = text_series.str.contains(r"\bprojects?\b", case=False, regex=True).astype(int)
    if "has_experience" not in data.columns:
        data["has_experience"] = text_series.str.contains(r"\bexperience\b", case=False, regex=True).astype(int)

    return data


def build_preprocessor() -> ColumnTransformer:
    text_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="")),
            ("flatten", FunctionTransformer(np.ravel, validate=False)),
            ("tfidf", TfidfVectorizer(max_features=5000)),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("text", text_pipeline, [TEXT_FEATURE]),
            ("num", numeric_pipeline, NUMERIC_FEATURES),
        ]
    )

    return preprocessor


def split_data(
    data: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    X = data[[TEXT_FEATURE] + NUMERIC_FEATURES]
    y_final = data[TARGET_FINAL_SCORE]
    y_quality = data[TARGET_QUALITY]
    y_domain = data[TARGET_DOMAIN]

    return train_test_split(
        X,
        y_final,
        y_quality,
        y_domain,
        test_size=test_size,
        random_state=random_state,
        stratify=y_quality,
    )


def encode_targets(
    y_quality_train: pd.Series,
    y_quality_test: pd.Series,
    y_domain_train: pd.Series,
    y_domain_test: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, LabelEncoder, LabelEncoder]:
    quality_encoder = LabelEncoder()
    domain_encoder = LabelEncoder()

    y_quality_train_enc = quality_encoder.fit_transform(y_quality_train)
    y_quality_test_enc = quality_encoder.transform(y_quality_test)

    y_domain_train_enc = domain_encoder.fit_transform(y_domain_train)
    y_domain_test_enc = domain_encoder.transform(y_domain_test)

    return (
        y_quality_train_enc,
        y_quality_test_enc,
        y_domain_train_enc,
        y_domain_test_enc,
        quality_encoder,
        domain_encoder,
    )


def train_models(
    X_train_transformed: Any,
    y_final_train: pd.Series,
    y_quality_train_enc: pd.Series,
    y_domain_train_enc: pd.Series,
    random_state: int,
) -> tuple[RandomForestRegressor, RandomForestClassifier, RandomForestClassifier, CalibratedClassifierCV, CalibratedClassifierCV]:
    rf_regressor = RandomForestRegressor(
        n_estimators=350,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=random_state,
        n_jobs=-1,
    )

    quality_classifier = RandomForestClassifier(
        n_estimators=420,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )

    domain_classifier = RandomForestClassifier(
        n_estimators=350,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=random_state,
        n_jobs=-1,
    )

    rf_regressor.fit(X_train_transformed, y_final_train)
    quality_classifier.fit(X_train_transformed, y_quality_train_enc)
    domain_classifier.fit(X_train_transformed, y_domain_train_enc)

    quality_calibrator = CalibratedClassifierCV(quality_classifier, method="sigmoid", cv=3)
    domain_calibrator = CalibratedClassifierCV(domain_classifier, method="sigmoid", cv=3)
    quality_calibrator.fit(X_train_transformed, y_quality_train_enc)
    domain_calibrator.fit(X_train_transformed, y_domain_train_enc)

    return rf_regressor, quality_classifier, domain_classifier, quality_calibrator, domain_calibrator


def derive_confidence_thresholds(probabilities: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    if probabilities.size == 0:
        return {"min_confidence": 0.58, "min_margin": 0.08}

    top_idx = np.argmax(probabilities, axis=1)
    top_probs = probabilities[np.arange(probabilities.shape[0]), top_idx]
    sorted_probs = np.sort(probabilities, axis=1)
    margins = top_probs - sorted_probs[:, -2]
    correct_mask = top_idx == y_true

    if np.any(correct_mask):
        min_conf = float(np.quantile(top_probs[correct_mask], 0.2))
        min_margin = float(np.quantile(margins[correct_mask], 0.2))
    else:
        min_conf = 0.58
        min_margin = 0.08

    return {
        "min_confidence": round(max(0.4, min_conf), 3),
        "min_margin": round(max(0.05, min_margin), 3),
    }


def evaluate_models(
    X_test_transformed: Any,
    y_final_test: pd.Series,
    y_quality_test_enc: pd.Series,
    y_domain_test_enc: pd.Series,
    rf_regressor: RandomForestRegressor,
    quality_classifier: LogisticRegression,
    domain_classifier: RandomForestClassifier,
) -> dict[str, float]:
    y_pred_final = rf_regressor.predict(X_test_transformed)
    y_pred_quality = quality_classifier.predict(X_test_transformed)
    y_pred_domain = domain_classifier.predict(X_test_transformed)

    mae = mean_absolute_error(y_final_test, y_pred_final)
    rmse = math.sqrt(mean_squared_error(y_final_test, y_pred_final))

    quality_acc = accuracy_score(y_quality_test_enc, y_pred_quality)
    quality_f1 = f1_score(y_quality_test_enc, y_pred_quality, average="weighted")

    domain_acc = accuracy_score(y_domain_test_enc, y_pred_domain)
    domain_f1 = f1_score(y_domain_test_enc, y_pred_domain, average="weighted")

    return {
        "regression_mae": mae,
        "regression_rmse": rmse,
        "quality_accuracy": quality_acc,
        "quality_f1": quality_f1,
        "domain_accuracy": domain_acc,
        "domain_f1": domain_f1,
    }


def print_metrics(metrics: dict[str, float]) -> None:
    print("\n=== Evaluation Results ===")
    print("Regression (final_score):")
    print(f"  MAE : {metrics['regression_mae']:.4f}")
    print(f"  RMSE: {metrics['regression_rmse']:.4f}")

    print("\nClassification (classification_label):")
    print(f"  Accuracy: {metrics['quality_accuracy']:.4f}")
    print(f"  F1 Score: {metrics['quality_f1']:.4f}")

    print("\nClassification (domain_label):")
    print(f"  Accuracy: {metrics['domain_accuracy']:.4f}")
    print(f"  F1 Score: {metrics['domain_f1']:.4f}")


def print_top_feature_importance(
    model: RandomForestRegressor | RandomForestClassifier,
    feature_names: list[str],
    title: str,
    top_n: int = 15,
) -> None:
    importances = model.feature_importances_
    pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:top_n]

    print(f"\n=== Top {top_n} Feature Importances: {title} ===")
    for name, score in pairs:
        print(f"  {name}: {score:.6f}")


def build_transformed_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    text_pipeline = preprocessor.named_transformers_["text"]
    tfidf_vectorizer = text_pipeline.named_steps["tfidf"]
    tfidf_features = [f"text__{name}" for name in tfidf_vectorizer.get_feature_names_out()]
    numeric_features = [f"num__{name}" for name in NUMERIC_FEATURES]
    return tfidf_features + numeric_features


def save_artifacts(
    output_dir: Path,
    preprocessor: ColumnTransformer,
    rf_regressor: RandomForestRegressor,
    quality_classifier: RandomForestClassifier,
    domain_classifier: RandomForestClassifier,
    quality_calibrator: CalibratedClassifierCV,
    domain_calibrator: CalibratedClassifierCV,
    quality_encoder: LabelEncoder,
    domain_encoder: LabelEncoder,
    confidence_thresholds: dict[str, dict[str, float]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(rf_regressor, output_dir / "rf_regressor.pkl")
    joblib.dump(quality_classifier, output_dir / "quality_classifier.pkl")
    joblib.dump(domain_classifier, output_dir / "domain_classifier.pkl")
    joblib.dump(quality_calibrator, output_dir / "quality_calibrator.pkl")
    joblib.dump(domain_calibrator, output_dir / "domain_calibrator.pkl")

    # Save full preprocessor and standalone TF-IDF vectorizer for inference compatibility.
    joblib.dump(preprocessor, output_dir / "preprocessor.pkl")
    tfidf_vectorizer = preprocessor.named_transformers_["text"].named_steps["tfidf"]
    joblib.dump(tfidf_vectorizer, output_dir / "tfidf_vectorizer.pkl")

    joblib.dump(quality_encoder, output_dir / "quality_label_encoder.pkl")
    joblib.dump(domain_encoder, output_dir / "domain_label_encoder.pkl")

    (output_dir / "confidence_thresholds.json").write_text(
        json.dumps(confidence_thresholds, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)

    df = load_data(data_path)
    data = prepare_data(df)

    (
        X_train,
        X_test,
        y_final_train,
        y_final_test,
        y_quality_train,
        y_quality_test,
        y_domain_train,
        y_domain_test,
    ) = split_data(data, test_size=args.test_size, random_state=args.random_state)

    (
        y_quality_train_enc,
        y_quality_test_enc,
        y_domain_train_enc,
        y_domain_test_enc,
        quality_encoder,
        domain_encoder,
    ) = encode_targets(
        y_quality_train,
        y_quality_test,
        y_domain_train,
        y_domain_test,
    )

    preprocessor = build_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    (
        rf_regressor,
        quality_classifier,
        domain_classifier,
        quality_calibrator,
        domain_calibrator,
    ) = train_models(
        X_train_transformed=X_train_transformed,
        y_final_train=y_final_train,
        y_quality_train_enc=y_quality_train_enc,
        y_domain_train_enc=y_domain_train_enc,
        random_state=args.random_state,
    )

    metrics = evaluate_models(
        X_test_transformed=X_test_transformed,
        y_final_test=y_final_test,
        y_quality_test_enc=y_quality_test_enc,
        y_domain_test_enc=y_domain_test_enc,
        rf_regressor=rf_regressor,
        quality_classifier=quality_classifier,
        domain_classifier=domain_classifier,
    )
    print_metrics(metrics)

    quality_probs = quality_calibrator.predict_proba(X_test_transformed)
    domain_probs = domain_calibrator.predict_proba(X_test_transformed)
    confidence_thresholds = {
        "quality": derive_confidence_thresholds(quality_probs, y_quality_test_enc),
        "domain": derive_confidence_thresholds(domain_probs, y_domain_test_enc),
    }

    feature_names = build_transformed_feature_names(preprocessor)
    print_top_feature_importance(rf_regressor, feature_names, title="RandomForestRegressor (final_score)")
    print_top_feature_importance(domain_classifier, feature_names, title="RandomForestClassifier (domain_label)")

    save_artifacts(
        output_dir=output_dir,
        preprocessor=preprocessor,
        rf_regressor=rf_regressor,
        quality_classifier=quality_classifier,
        domain_classifier=domain_classifier,
        quality_calibrator=quality_calibrator,
        domain_calibrator=domain_calibrator,
        quality_encoder=quality_encoder,
        domain_encoder=domain_encoder,
        confidence_thresholds=confidence_thresholds,
    )

    print("\n=== Saved Artifacts ===")
    print(f"  Models and preprocessors saved to: {output_dir}")
    print("  - rf_regressor.pkl")
    print("  - quality_classifier.pkl")
    print("  - domain_classifier.pkl")
    print("  - preprocessor.pkl")
    print("  - tfidf_vectorizer.pkl")
    print("  - quality_label_encoder.pkl")
    print("  - domain_label_encoder.pkl")
    print("  - quality_calibrator.pkl")
    print("  - domain_calibrator.pkl")
    print("  - confidence_thresholds.json")


if __name__ == "__main__":
    main()
