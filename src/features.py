"""Feature groups for the UCI CKD dataset.

Column names follow the UCI abbreviations:
age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc, sod, pot, hemo, pcv,
wc, rc, htn, dm, cad, appet, pe, ane, class
"""

TARGET = "class"

NUMERIC = [
    "age", "bp", "sg", "al", "su", "bgr", "bu", "sc",
    "sod", "pot", "hemo", "pcv", "wc", "rc",
]

CATEGORICAL = [
    "rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane",
]

# Binary categoricals and their value mappings (1 = abnormal / present / yes / poor)
BINARY_MAP = {
    "rbc": {"normal": 0, "abnormal": 1},
    "pc": {"normal": 0, "abnormal": 1},
    "pcc": {"notpresent": 0, "present": 1},
    "ba": {"notpresent": 0, "present": 1},
    "htn": {"no": 0, "yes": 1},
    "dm": {"no": 0, "yes": 1},
    "cad": {"no": 0, "yes": 1},
    "appet": {"good": 0, "poor": 1},
    "pe": {"no": 0, "yes": 1},
    "ane": {"no": 0, "yes": 1},
}

# Lab values that are direct kidney-function markers or near-direct consequences
# of CKD. They make the classification task almost trivial, so the project
# compares models WITH and WITHOUT them (leakage ablation).
#   sc   serum creatinine  (eGFR is computed from it)
#   bu   blood urea
#   hemo / pcv / rc  anemia markers, strongly tied to advanced CKD
#   sg / al  urine specific gravity and albumin (albuminuria defines CKD staging)
LEAKAGE_STRICT = ["sc", "bu"]
LEAKAGE_BROAD = ["sc", "bu", "hemo", "pcv", "rc", "sg", "al"]

# Features available in a realistic early-screening setting
# (history, vitals, simple urine/blood glucose checks).
SCREENING_FEATURES = [
    "age", "bp", "bgr", "htn", "dm", "cad", "appet", "pe", "ane",
]
