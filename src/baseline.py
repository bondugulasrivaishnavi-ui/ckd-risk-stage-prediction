"""Quick baseline + leakage ablation.

Usage:
    python -m src.baseline

Compares Logistic Regression and Random Forest under three feature sets:
    all          every feature
    no_leakage   drops serum creatinine / blood urea / anemia / urine markers
    screening    only history, vitals and glucose (realistic early screening)
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_clean
from src.evaluate import compare_models
from src.features import LEAKAGE_BROAD, SCREENING_FEATURES, TARGET


def make_models() -> dict:
    return {
        "LogReg": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]),
        "RandomForest": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=300, class_weight="balanced", random_state=42)),
        ]),
    }


def main():
    df = load_clean()
    y = df[TARGET]
    all_features = [c for c in df.columns if c != TARGET]

    feature_sets = {
        "all": all_features,
        "no_leakage": [c for c in all_features if c not in LEAKAGE_BROAD],
        "screening": [c for c in SCREENING_FEATURES if c in df.columns],
    }

    for name, cols in feature_sets.items():
        print(f"\n=== Feature set: {name} ({len(cols)} features) ===")
        table = compare_models(make_models(), df[cols], y)
        print(table.to_string())


if __name__ == "__main__":
    main()
