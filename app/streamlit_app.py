"""Streamlit demo (stub). Run with:  streamlit run app/streamlit_app.py

Right now this trains a small screening model on the fly from the cleaned UCI
data. Later: load the final tuned model, add stage prediction and SHAP plots.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.data_loader import load_clean
from src.features import SCREENING_FEATURES, TARGET

st.set_page_config(page_title="CKD Risk Screening", page_icon="🩺")
st.title("CKD Risk Screening (demo)")
st.caption("Educational project. Not a medical device and not for clinical use.")


@st.cache_resource
def train():
    df = load_clean()
    cols = [c for c in SCREENING_FEATURES if c in df.columns]
    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42)),
    ])
    model.fit(df[cols], df[TARGET])
    return model, cols


model, cols = train()

st.subheader("Patient details")
c1, c2 = st.columns(2)
with c1:
    age = st.number_input("Age", 1, 100, 50)
    bp = st.number_input("Blood pressure (diastolic, mmHg)", 40, 180, 80)
    bgr = st.number_input("Random blood glucose (mg/dL)", 50, 500, 110)
with c2:
    htn = st.selectbox("Hypertension", ["no", "yes"])
    dm = st.selectbox("Diabetes mellitus", ["no", "yes"])
    cad = st.selectbox("Coronary artery disease", ["no", "yes"])
    appet = st.selectbox("Appetite", ["good", "poor"])
    pe = st.selectbox("Pedal edema", ["no", "yes"])
    ane = st.selectbox("Anemia", ["no", "yes"])

if st.button("Estimate risk"):
    yn = {"no": 0, "yes": 1}
    row = {
        "age": age, "bp": bp, "bgr": bgr,
        "htn": yn[htn], "dm": yn[dm], "cad": yn[cad],
        "appet": 0 if appet == "good" else 1,
        "pe": yn[pe], "ane": yn[ane],
    }
    X = pd.DataFrame([row])[cols]
    prob = float(model.predict_proba(X)[0, 1])
    st.metric("Estimated CKD risk", f"{prob:.0%}")
    st.progress(min(max(prob, 0.0), 1.0))
    st.info("TODO: add predicted stage and SHAP explanation for this patient.")
