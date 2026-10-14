# CKD Risk and Stage Prediction Using Data Mining Techniques

An end-to-end, leakage-aware data mining project on Chronic Kidney Disease (CKD):
classification, stage prediction, patient subtyping, association rule mining,
explainability (SHAP), external validation, and a deployable web app.

> Status: **work in progress.** Checked boxes below are done.

## Why this project is different from the usual UCI notebook

The UCI CKD dataset is small (400 rows) and almost perfectly separable, so
99% accuracy is easy and says little. This project focuses on:

- **Leakage analysis:** serum creatinine, hemoglobin, etc. nearly define the label.
  We compare models with and without them to build a real *early screening* model.
- **Rigorous evaluation:** stratified nested cross-validation, recall / precision /
  F1 / ROC-AUC / PR-AUC, calibration curves (not just accuracy).
- **External validation** on a larger population dataset (NHANES).
- **Multiple data mining techniques** beyond classification.
- **Explainability** and a **usable app**.

## Roadmap

- [ ] **Data:** UCI CKD baseline, NHANES external set, documented data issues
- [ ] **Preprocessing:** missingness analysis, simple vs KNN vs MICE imputation, outliers, scaling, encoding
- [ ] **Feature selection:** correlation filter, mutual information, RFE, L1 (compare what each picks)
- [ ] **Classification:** LR, Decision Tree, Random Forest, SVM, XGBoost, LightGBM with nested CV
- [ ] **Leakage ablation:** with vs without creatinine / eGFR-related features
- [ ] **Clustering:** k-means, GMM, hierarchical; silhouette + clinical interpretation
- [ ] **Association rules:** Apriori / FP-Growth on risk factor co-occurrence
- [ ] **Stage prediction:** multi-class G1-G5 (requires eGFR-labelled data, e.g. NHANES)
- [ ] **Explainability:** SHAP global + per-patient
- [ ] **Calibration and threshold analysis** for screening use
- [ ] **App:** Streamlit risk + stage + SHAP explanation
- [ ] **Report and README results tables**

## Project structure

```
ckd-risk-stage-prediction/
├── data/raw/            # downloaded datasets (git-ignored)
├── data/processed/      # cleaned datasets (git-ignored)
├── notebooks/           # 01_eda.ipynb, 02_..., one per stage
├── src/
│   ├── data_loader.py   # download + clean UCI CKD
│   ├── evaluate.py      # metrics, CV, calibration helpers
│   └── features.py      # feature groups incl. leakage-prone columns
├── app/streamlit_app.py # demo app (stub)
├── reports/figures/     # saved plots
└── requirements.txt
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.data_loader        # downloads + cleans UCI CKD -> data/processed/ckd_clean.csv
python -m src.baseline           # quick baseline: LR + RF with CV
jupyter notebook notebooks/01_eda.ipynb
```

## Datasets

| Dataset | Use | Source |
|---|---|---|
| UCI Chronic Kidney Disease (400 rows, 24 features) | baseline, EDA | UCI ML Repository (id 336) |
| NHANES (CDC) | external validation, stage labels via eGFR + albuminuria | CDC NHANES |

## Results

_Add results tables here as you complete each stage._

## Disclaimer

Educational project. Not a medical device and not for clinical use.
