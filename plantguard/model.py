"""
model.py — Feature engineering and predictive-maintenance model for PlantGuard AI.

Two complementary approaches, both included on purpose (good talking point
in an interview or essay about *why* you chose the approach you did):

1. IsolationForest (unsupervised anomaly detection) — flags readings that
   look statistically unusual vs. normal operation, with NO labeled failure
   data required. This is the realistic case: most plants don't have
   thousands of labeled failure events to train on.

2. RandomForestClassifier (supervised risk scoring) — once you DO have
   some labeled history (like our simulated `label` column), a supervised
   model gives a calibrated 0-100% risk score, which is what a dashboard
   actually wants to show an operator.

Both models share the same engineered features: rolling means/stds
capture *trend*, not just instantaneous value, which is how real
predictive-maintenance systems catch slow degradation before a trip.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

SENSOR_COLS = ["bearing_temp_f", "vibration_ips", "discharge_psi"]
ROLL_WINDOW = 15  # minutes


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling mean/std features that capture trend + volatility."""
    out = df.copy()
    for col in SENSOR_COLS:
        out[f"{col}_roll_mean"] = out[col].rolling(ROLL_WINDOW, min_periods=1).mean()
        out[f"{col}_roll_std"] = out[col].rolling(ROLL_WINDOW, min_periods=1).std().fillna(0)
        # rate of change vs. the rolling mean N steps ago = "creep" signal
        out[f"{col}_slope"] = out[f"{col}_roll_mean"].diff().fillna(0)
    return out


def feature_matrix(df_feat: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in df_feat.columns if c not in ("timestamp", "label")]
    return df_feat[feature_cols]


def train_isolation_forest(df: pd.DataFrame, contamination: float = 0.12) -> tuple:
    """Train unsupervised anomaly detector. Returns (model, feature_cols)."""
    df_feat = engineer_features(df)
    X = feature_matrix(df_feat)
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
    )
    model.fit(X)
    return model, list(X.columns)


def train_risk_classifier(df: pd.DataFrame) -> tuple:
    """Train supervised risk-score classifier. Returns (model, feature_cols, metrics)."""
    df_feat = engineer_features(df)
    X = feature_matrix(df_feat)
    y = df_feat["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    metrics = {
        "roc_auc": roc_auc_score(y_test, proba) if len(set(y_test)) > 1 else float("nan"),
        "report": classification_report(y_test, preds, zero_division=0),
    }
    return model, list(X.columns), metrics


def risk_score(model, feature_cols: list, df_feat: pd.DataFrame) -> np.ndarray:
    """Return a 0-100 risk score per row from a trained RandomForestClassifier."""
    X = df_feat[feature_cols]
    return (model.predict_proba(X)[:, 1] * 100).round(1)


def anomaly_flags(model, feature_cols: list, df_feat: pd.DataFrame) -> np.ndarray:
    """Return boolean anomaly flags per row from a trained IsolationForest."""
    X = df_feat[feature_cols]
    preds = model.predict(X)  # -1 = anomaly, 1 = normal
    return preds == -1


def save_models(iso_model, iso_cols, rf_model, rf_cols, path_prefix: str = "plantguard_model"):
    joblib.dump({"model": iso_model, "cols": iso_cols}, f"{path_prefix}_iso.joblib")
    joblib.dump({"model": rf_model, "cols": rf_cols}, f"{path_prefix}_rf.joblib")


if __name__ == "__main__":
    from data_sim import generate_asset_run

    print("Generating synthetic asset run...")
    df = generate_asset_run()

    print("\n--- Training IsolationForest (unsupervised) ---")
    iso_model, iso_cols = train_isolation_forest(df)
    df_feat = engineer_features(df)
    flags = anomaly_flags(iso_model, iso_cols, df_feat)
    print(f"Flagged {flags.sum()} / {len(df)} points as anomalous")
    print(f"Of the {int(df['label'].sum())} truly degraded points, "
          f"{int((flags & (df['label'] == 1)).sum())} were caught")

    print("\n--- Training RandomForestClassifier (supervised risk score) ---")
    rf_model, rf_cols, metrics = train_risk_classifier(df)
    print(f"ROC AUC: {metrics['roc_auc']:.3f}")
    print(metrics["report"])

    scores = risk_score(rf_model, rf_cols, df_feat)
    print(f"Final risk score (last reading): {scores[-1]}%")
    print(f"Risk score at start of run:      {scores[0]}%")

    save_models(iso_model, iso_cols, rf_model, rf_cols)
    print("\nModels saved to plantguard_model_iso.joblib / plantguard_model_rf.joblib")
