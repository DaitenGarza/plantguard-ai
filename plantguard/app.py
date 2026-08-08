"""
app.py — PlantGuard AI dashboard (Streamlit).

Run with:  streamlit run app.py

Simulates a plant asset (centrifugal pump) sensor feed, scores it for
failure risk with a trained model, and displays it the way an operator
console would: live trend charts, a risk gauge, and an anomaly alert log.
"""

import numpy as np
import pandas as pd
import streamlit as st

from data_sim import generate_asset_run
from model import (
    engineer_features,
    train_isolation_forest,
    train_risk_classifier,
    risk_score,
    anomaly_flags,
    SENSOR_COLS,
)

st.set_page_config(page_title="PlantGuard AI", layout="wide", page_icon="⚙️")

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ PlantGuard AI")
st.sidebar.caption("Predictive maintenance demo — centrifugal pump asset")

asset_name = st.sidebar.text_input("Asset tag", value="P-204A  (Charge Pump)")
seed = st.sidebar.number_input("Simulation seed", value=42, step=1)
n_points = st.sidebar.slider("Run length (minutes)", 500, 4000, 2000, step=100)
scrub = st.sidebar.slider(
    "Playhead — replay the run up to this point",
    0, n_points - 1, n_points - 1,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**How to read this:** healthy operation runs flat and quiet. "
    "Watch the risk score and vibration trend climb together in the last "
    "third of the run — that's the model catching a slow bearing failure "
    "*before* the pump trips, which is the entire point of predictive "
    "maintenance vs. run-to-failure."
)

# ---------------------------------------------------------------------------
# Data + model (cached so the sidebar sliders don't retrain every rerun)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_run(seed: int, n_points: int) -> pd.DataFrame:
    return generate_asset_run(n_points=n_points, seed=seed)


@st.cache_resource(show_spinner="Training models...")
def load_models(seed: int, n_points: int):
    df = load_run(seed, n_points)
    iso_model, iso_cols = train_isolation_forest(df)
    rf_model, rf_cols, metrics = train_risk_classifier(df)
    return iso_model, iso_cols, rf_model, rf_cols, metrics


df_full = load_run(seed, n_points)
iso_model, iso_cols, rf_model, rf_cols, metrics = load_models(seed, n_points)

df_feat_full = engineer_features(df_full)
scores_full = risk_score(rf_model, rf_cols, df_feat_full)
flags_full = anomaly_flags(iso_model, iso_cols, df_feat_full)

# Slice everything to the scrub playhead so the dashboard feels "live"
df = df_full.iloc[: scrub + 1]
scores = scores_full[: scrub + 1]
flags = flags_full[: scrub + 1]

latest_score = scores[-1]
latest_row = df.iloc[-1]

# ---------------------------------------------------------------------------
# Header + risk gauge
# ---------------------------------------------------------------------------
st.title(f"{asset_name}")
st.caption(f"Live sensor feed · {latest_row['timestamp']}")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Bearing Temp", f"{latest_row['bearing_temp_f']:.1f} °F")
col2.metric("Vibration", f"{latest_row['vibration_ips']:.3f} in/sec")
col3.metric("Discharge Pressure", f"{latest_row['discharge_psi']:.1f} psi")

if latest_score < 30:
    risk_label, risk_color = "LOW", "normal"
elif latest_score < 65:
    risk_label, risk_color = "ELEVATED", "off"
else:
    risk_label, risk_color = "HIGH — SCHEDULE MAINTENANCE", "inverse"

col4.metric("Failure Risk Score", f"{latest_score:.0f}%", risk_label, delta_color=risk_color)

if latest_score >= 65:
    st.error(
        f"⚠️ **{asset_name} risk score is {latest_score:.0f}%.** "
        "Sensor trend matches early-stage bearing degradation. "
        "Recommend scheduling inspection before next run."
    )
elif latest_score >= 30:
    st.warning(f"Risk trending up ({latest_score:.0f}%). Monitor closely.")
else:
    st.success(f"Operating normally. Risk score {latest_score:.0f}%.")

# ---------------------------------------------------------------------------
# Trend charts
# ---------------------------------------------------------------------------
st.subheader("Sensor Trends")
chart_df = df.set_index("timestamp")[SENSOR_COLS]
st.line_chart(chart_df, height=280)

st.subheader("Failure Risk Score Over Time")
risk_df = pd.DataFrame({"timestamp": df["timestamp"], "risk_score": scores}).set_index("timestamp")
st.area_chart(risk_df, height=220)

# ---------------------------------------------------------------------------
# Anomaly alert log
# ---------------------------------------------------------------------------
st.subheader("Anomaly Alert Log")
alert_df = df.loc[flags, ["timestamp", *SENSOR_COLS]].copy()
alert_df["risk_score"] = scores[flags]

if len(alert_df) == 0:
    st.info("No anomalies flagged yet in this window.")
else:
    st.dataframe(
        alert_df.sort_values("timestamp", ascending=False).reset_index(drop=True),
        use_container_width=True,
        height=250,
    )

with st.expander("Model performance (held-out test data)"):
    st.text(f"ROC AUC: {metrics['roc_auc']:.3f}")
    st.text(metrics["report"])
    st.caption(
        "Trained on this simulated run's own labels — in production you'd "
        "train on historical failure records across many pumps of the same "
        "class, not a single asset's simulated history."
    )
